"""Graph construction for progressive multilevel mesh trajectories."""

from typing import List
from time import perf_counter

import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

from mesh_util import MeshWrapper, get_sizing_field

# Optional experiment instrumentation; inactive for normal training/inference.
graph_profile_hook = None


class ProgressiveData(Data):
    """PyG data with element-local facet indices batched correctly."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {
            "current_element_index",
            "pivot_interp_index",
            "current_interior_facet_elements",
            "current_interior_facet_vertices",
        }:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "level_index":
            # This is a categorical level id, not a node index.  PyG's
            # default treats every key containing "index" as offsettable.
            return 0
        if key == "current_interior_facet_elements":
            return self.current_element_index.shape[0]
        if key == "current_interior_facet_vertices":
            return self.num_nodes
        if key == "owner_pivot":
            return self.num_nodes
        if key == "pivot_interp_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def extract_poisson_features(mesh: MeshWrapper, load_fn,
                             use_fem_solution_input: bool = False) -> List[np.ndarray]:
    degree = np.unique(mesh.mesh_edges.flatten(), return_counts=True)[1].astype(np.float64)
    sizing = get_sizing_field(mesh)
    load = load_fn.evaluate(mesh.vertex_positions)
    coords = mesh.vertex_positions
    features = [coords[:, i] for i in range(coords.shape[1])] + [degree, sizing, load]
    if use_fem_solution_input:
        solution = mesh.__dict__.get("_poisson_solution_feature")
        if solution is None:
            from data_generator.poisson_data_generator import solve_poisson

            solution = np.asarray(solve_poisson(mesh.mesh, load_fn)).reshape(-1)
            if solution.shape[0] != mesh.num_vertices or not np.all(np.isfinite(solution)):
                raise ValueError("Invalid FEM solution feature on current mesh.")
            mesh._poisson_solution_feature = solution
        features.append(solution)
    return features


def vertex_edges(mesh: MeshWrapper, edge_feature_names: List[str], add_self_edges: bool = True):
    edges = torch.tensor(mesh.mesh_edges, dtype=torch.long)
    positions = torch.tensor(mesh.vertex_positions, dtype=torch.float32)
    src = torch.cat([edges[0], edges[1]], dim=0)
    dst = torch.cat([edges[1], edges[0]], dim=0)
    is_self = torch.zeros(src.numel(), dtype=torch.bool)

    if add_self_edges:
        n = mesh.num_vertices
        self_idx = torch.arange(n)
        src = torch.cat([src, self_idx])
        dst = torch.cat([dst, self_idx])
        is_self = torch.cat([is_self, torch.ones(n, dtype=torch.bool)])

    edge_feats = []
    if "euclidean_distance" in edge_feature_names:
        edge_feats.append(torch.norm(positions[dst] - positions[src], dim=1))
    if "edge_curvature" in edge_feature_names:
        edge_feats.append(_edge_curvature_feature(mesh, src, dst))

    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.stack(edge_feats, dim=1)
    return edge_index, edge_attr, is_self


def _edge_curvature_feature(mesh: MeshWrapper, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    try:
        boundary_curvatures = mesh.boundary_edge_curvatures
        boundary_e = mesh.boundary_edges
        boundary_map = {}
        for i in range(boundary_e.shape[1]):
            key = (min(boundary_e[0, i], boundary_e[1, i]), max(boundary_e[0, i], boundary_e[1, i]))
            boundary_map[key] = boundary_curvatures[i]
        curv = []
        for s, d in zip(src.tolist(), dst.tolist()):
            if s == d:
                curv.append(0.0)
            else:
                curv.append(boundary_map.get((min(s, d), max(s, d)), 0.0))
        return torch.tensor(curv, dtype=torch.float32)
    except Exception:
        return torch.zeros(src.numel(), dtype=torch.float32)


def _segments_inside_2d_mesh(mesh: MeshWrapper, edges: np.ndarray) -> np.ndarray:
    """Reject geometric shortcut edges crossing holes or L-shaped voids."""
    if mesh.mesh.dim() != 2 or len(edges) == 0:
        return np.ones(len(edges), dtype=bool)
    triangles = mesh.vertex_positions[mesh.element_indices]
    points = mesh.vertex_positions[edges]
    fractions = np.array([0.2, 0.5, 0.8])
    samples = points[:, 0, None, :] * (1.0 - fractions)[None, :, None] + points[:, 1, None, :] * fractions[None, :, None]
    # Query only nearby elements.  The previous implementation tested every
    # sample against every triangle, making graph construction quadratic for
    # the larger Laplace and beam meshes.
    centroids = triangles.mean(axis=1)
    candidate_k = min(32, len(triangles))
    candidates = cKDTree(centroids).query(samples.reshape(-1, 2), k=candidate_k)[1]
    candidates = np.asarray(candidates).reshape(len(edges), 3, candidate_k)
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    den = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    valid = np.abs(den) > 1e-14
    good = np.zeros((len(edges), 3), dtype=bool)
    for i in range(samples.shape[1]):
        sample_set = samples[:, i, :]
        tri = triangles[candidates[:, i]]
        aa = a[candidates[:, i]]
        bb = b[candidates[:, i]]
        dd = den[candidates[:, i]]
        p = sample_set[:, None, :] - tri[:, :, 0, :]
        u = (p[..., 0] * bb[..., 1] - p[..., 1] * bb[..., 0]) / dd
        v = (aa[..., 0] * p[..., 1] - aa[..., 1] * p[..., 0]) / dd
        good[:, i] = np.any((np.abs(dd) > 1e-14) & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8), axis=1)
    return good.all(axis=1)


def cross_level_edges(
    fine_mesh: MeshWrapper,
    coarse_mesh: MeshWrapper,
    fine_offset: int,
    coarse_offset: int,
    edge_feature_names: List[str],
):
    fine_idx = np.arange(fine_mesh.num_vertices)
    fine_pos = fine_mesh.vertex_positions
    coarse_idx = (coarse_mesh.fine_owner if hasattr(coarse_mesh, 'fine_owner')
                  else coarse_mesh.vertex_tree.query(fine_pos, k=1)[1].astype(np.int64))
    coarse_pos = coarse_mesh.vertex_positions[coarse_idx]

    src = np.concatenate([fine_idx + fine_offset, coarse_idx + coarse_offset])
    dst = np.concatenate([coarse_idx + coarse_offset, fine_idx + fine_offset])
    pos_src = np.concatenate([fine_pos, coarse_pos], axis=0)
    pos_dst = np.concatenate([coarse_pos, fine_pos], axis=0)

    feats = []
    if "euclidean_distance" in edge_feature_names:
        feats.append(np.linalg.norm(pos_dst - pos_src, axis=1))
    if "edge_curvature" in edge_feature_names:
        feats.append(np.zeros(src.shape[0], dtype=np.float64))
    return (
        torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        torch.tensor(np.array(feats).T, dtype=torch.float32),
    )


def build_multilevel_graph(original_mesh: MeshWrapper, current_mesh: MeshWrapper, load_fn,
                           current_level: int, max_levels: int,
                           edge_feature_names: List[str],
                           use_fem_solution_input: bool = False,
                           use_multilevel_graph: bool = True,
                           task_name: str = "poisson",
                           use_pivot_shortcuts: bool = False,
                           pivot_ratio: float = 0.10,
                           pivot_alpha: float = 1.0,
                           pivot_neighbors: int = 2,
                           use_hop_coarse_graph: bool = False,
                           use_pivot_residual_heads: bool = False,
                           use_pivot_graph: bool = False,
                           use_pivot_communication: bool = True,
                           pivot_communication_topology: str = "delaunay") -> Data:
    """Build a Mars-style current graph with an optional topology-only M0 graph."""
    profile_start = perf_counter()
    profile_pivot_seconds = 0.0
    profile_interpolation_seconds = 0.0
    profile_communication_seconds = 0.0
    if not 0 <= current_level < max_levels:
        raise ValueError(f"current_level={current_level} is outside [0, {max_levels}).")
    if pivot_communication_topology not in {"delaunay", "knn6"}:
        raise ValueError(f"Unknown pivot communication topology: {pivot_communication_topology}")
    include_original = use_multilevel_graph and current_level > 0
    if use_hop_coarse_graph and use_pivot_shortcuts:
        raise ValueError('Coarse replacement and shortcut experiments must be separate')
    if include_original and use_hop_coarse_graph:
        from hop_coarse import HopCoarseGraph
        section_start = perf_counter()
        original_mesh = HopCoarseGraph(current_mesh, get_sizing_field(current_mesh), pivot_ratio)
        profile_pivot_seconds += perf_counter() - section_start
    meshes = [current_mesh, original_mesh] if include_original else [current_mesh]
    true_levels = [current_level, 0] if include_original else [current_level]
    current_mesh_index = 0
    xs = []
    edge_indices = []
    edge_attrs = []
    level_index = []
    current_masks = []
    intra_current_masks = []
    offsets = []
    offset = 0
    num_base_features = None

    for local_level, (true_level, mesh) in enumerate(zip(true_levels, meshes)):
        offsets.append(offset)
        is_current = local_level == current_mesh_index
        if is_current:
            features = extract_poisson_features(
                mesh, load_fn, use_fem_solution_input=use_fem_solution_input
            )
            base_x = torch.tensor(np.stack(features, axis=1), dtype=torch.float32)
            num_base_features = base_x.shape[1]
        else:
            # As in Mars, M0 is structural context rather than a second set of
            # physical observations.  This keeps cross-level edges focused on
            # shortening message paths.
            base_x = torch.zeros(mesh.num_vertices, num_base_features, dtype=torch.float32)
        graph_type = torch.full(
            (mesh.num_vertices, 1), 0.0 if is_current else 1.0,
            dtype=torch.float32,
        )
        x = torch.cat([base_x, graph_type], dim=1)
        xs.append(x)

        edge_index, edge_attr, is_self = vertex_edges(mesh, edge_feature_names, add_self_edges=True)
        edge_indices.append(edge_index + offset)
        edge_attrs.append(edge_attr)
        intra_current_masks.append(
            torch.full((edge_index.shape[1],), is_current, dtype=torch.bool) & ~is_self
        )

        level_index.append(torch.full((mesh.num_vertices,), true_level, dtype=torch.long))
        current_masks.append(torch.full(
            (mesh.num_vertices,), is_current, dtype=torch.bool
        ))
        offset += mesh.num_vertices

    if len(meshes) == 2:
        # Connect Mk directly to M0 in both directions.
        edge_index, edge_attr = cross_level_edges(
            current_mesh, original_mesh, offsets[0], offsets[1], edge_feature_names
        )
        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
        intra_current_masks.append(torch.zeros(edge_index.shape[1], dtype=torch.bool))

    if use_pivot_shortcuts and current_mesh.num_vertices > 4:
        # Add sparse long-range edges without changing node semantics.  The
        # sizing-weighted FPS pivots are recomputed on every rollout mesh.
        pos = current_mesh.vertex_positions
        e = np.asarray(current_mesh.mesh_edges, dtype=np.int64)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import shortest_path
        adjacency = coo_matrix((np.ones(2 * e.shape[1]),
            (np.r_[e[0], e[1]], np.r_[e[1], e[0]])),
            shape=(current_mesh.num_vertices, current_mesh.num_vertices)).tocsr()
        distances = {}
        def distance_to(source):
            if source not in distances:
                distances[source] = shortest_path(adjacency, directed=False,
                    unweighted=True, indices=source)
            return distances[source]
        sizing = get_sizing_field(current_mesh).reshape(-1)
        w = 1.0 / (np.maximum(sizing, 1e-8) + 1e-8)
        k = max(1, min(current_mesh.num_vertices, int(np.ceil(pivot_ratio * current_mesh.num_vertices))))
        pivots = [int(np.argmax(w))]; dmin = distance_to(pivots[0])
        for _ in range(1, k):
            score = (w ** float(pivot_alpha)) * dmin
            score[pivots] = -np.inf
            p = int(np.argmax(score)); pivots.append(p); dmin = np.minimum(dmin, distance_to(p))
        pivot_dist = np.stack([distance_to(p) for p in pivots], axis=1)
        nearest = np.argsort(pivot_dist, axis=1)[:, :max(1, int(pivot_neighbors))]
        src = np.repeat(np.arange(current_mesh.num_vertices), nearest.shape[1]) + offsets[0]
        dst = np.asarray(pivots, dtype=np.int64)[nearest.reshape(-1)] + offsets[0]
        shortcut = np.vstack([np.r_[src, dst], np.r_[dst, src]])
        shortcut_attr = []
        if "euclidean_distance" in edge_feature_names:
            shortcut_attr.append(np.linalg.norm(pos[shortcut[1] - offsets[0]] - pos[shortcut[0] - offsets[0]], axis=1))
        if "edge_curvature" in edge_feature_names:
            shortcut_attr.append(np.zeros(shortcut.shape[1], dtype=np.float64))
        shortcut_attr = np.asarray(shortcut_attr).T
        edge_indices.append(torch.tensor(shortcut, dtype=torch.long))
        edge_attrs.append(torch.tensor(shortcut_attr, dtype=torch.float32))
        intra_current_masks.append(torch.ones(shortcut.shape[1], dtype=torch.bool))

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    positions = torch.tensor(np.concatenate([m.vertex_positions for m in meshes], axis=0), dtype=torch.float32)
    edge_length = torch.norm(positions[edge_index[1]] - positions[edge_index[0]], dim=1, keepdim=True)

    current_offset = offsets[current_mesh_index]
    current_elements = torch.tensor(
        current_mesh.element_indices + current_offset, dtype=torch.long
    )
    current_areas = torch.tensor(current_mesh.simplex_volumes, dtype=torch.float32).unsqueeze(-1)
    # Degree-two three-point rule matches the element-integrated residual term
    # used by skfem better than a single element-center sample.
    if current_mesh.mesh.dim() == 2:
        barycentric_points = np.array([[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                                       [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                                       [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]])
    else:
        barycentric_points = np.array([
            [0.58541020, 0.13819660, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.13819660, 0.58541020],
        ])
    element_vertices = current_mesh.vertex_positions[current_mesh.element_indices]
    quadrature_points = np.einsum("qv,evd->eqd", barycentric_points, element_vertices)
    load_values = load_fn.evaluate(quadrature_points.reshape(-1, current_mesh.mesh.dim())).reshape(
        current_mesh.num_elements, len(barycentric_points)
    )
    current_load_squared_mean = np.square(load_values).mean(axis=1)
    current_load = torch.tensor(current_load_squared_mean, dtype=torch.float32).unsqueeze(-1)
    facet_to_elements = current_mesh.mesh.f2t
    interior_facets = np.flatnonzero(facet_to_elements[1] != -1)
    current_interior_facet_elements = torch.tensor(
        facet_to_elements[:, interior_facets].T, dtype=torch.long
    )
    current_interior_facet_vertices = torch.tensor(
        current_mesh.mesh.facets[:, interior_facets].T + current_offset, dtype=torch.long
    )

    data = ProgressiveData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pos = positions
    data.edge_length = edge_length
    data.level_index = torch.cat(level_index, dim=0)
    data.current_level_mask = torch.cat(current_masks, dim=0)
    data.mask_output = data.current_level_mask
    data.intra_current_edge_mask = torch.cat(intra_current_masks, dim=0)
    data.current_level = torch.tensor(current_level, dtype=torch.long)
    data.current_element_index = current_elements
    data.current_element_area = current_areas
    data.current_element_load = current_load
    data.mesh_dim = int(current_mesh.mesh.dim())
    pivot_enabled = use_pivot_residual_heads or use_pivot_graph
    if pivot_enabled:
        from hop_coarse import HopCoarseGraph
        section_start = perf_counter()
        skeleton = HopCoarseGraph(current_mesh, get_sizing_field(current_mesh), pivot_ratio)
        profile_pivot_seconds += perf_counter() - section_start
        pivot_mask = np.zeros(current_mesh.num_vertices, dtype=bool)
        pivot_mask[skeleton.pivots] = True
        data.pivot_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.fine_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.owner_pivot = torch.full((data.num_nodes,), -1, dtype=torch.long)
        data.pivot_mask[current_offset:current_offset + current_mesh.num_vertices] = torch.tensor(pivot_mask)
        data.fine_mask[current_offset:current_offset + current_mesh.num_vertices] = torch.tensor(~pivot_mask)
        # Communication ownership is geometric, not hop-based.  This keeps a
        # fine residual local to a nearby pivot, so its target remains local.
        pivot_pos = current_mesh.vertex_positions[skeleton.pivots]
        fine_pos = current_mesh.vertex_positions
        owners = skeleton.pivots[cKDTree(pivot_pos).query(fine_pos, k=1)[1]]
        data.owner_pivot[current_offset:current_offset + current_mesh.num_vertices] = torch.tensor(owners)
        # Store a continuous pivot interpolation stencil for residual targets.
        section_start = perf_counter()
        # Delaunay barycentric interpolation is used inside the pivot hull;
        # inverse-distance weights provide a stable fallback outside it.
        # A d-dimensional simplex interpolation stencil has d+1 vertices.
        # The previous fixed width of three was correct for triangles but
        # under-parameterized the 3D case, where a tetrahedral stencil needs
        # four pivots.
        interp_width = current_mesh.mesh.dim() + 1
        interp_ids = np.full(
            (current_mesh.num_vertices, interp_width), -1, dtype=np.int64
        )
        interp_w = np.zeros(
            (current_mesh.num_vertices, interp_width), dtype=np.float32
        )
        interp_ids[:, 0] = owners
        interp_w[:, 0] = 1.0
        if current_mesh.mesh.dim() in (2, 3) and len(skeleton.pivots) >= interp_width:
            try:
                tri = Delaunay(pivot_pos)
                simplex = tri.find_simplex(fine_pos)
                good = simplex >= 0
                if good.any():
                    dim = current_mesh.mesh.dim()
                    transform = tri.transform[simplex[good], :dim]
                    delta = fine_pos[good] - tri.transform[simplex[good], dim]
                    bary = np.einsum('nij,nj->ni', transform, delta)
                    bary = np.c_[bary, 1.0 - bary.sum(axis=1)]
                    verts = tri.simplices[simplex[good]]
                    interp_ids[good] = skeleton.pivots[verts]
                    # Numerical roundoff can produce tiny negative values on
                    # simplex boundaries.  Keep the stencil nonnegative and
                    # normalized so the base remains a convex interpolation.
                    bary = np.maximum(bary, 0.0)
                    bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
                    interp_w[good] = bary.astype(np.float32)
            except Exception:
                pass
        # Use IDW for any non-Delaunay point, including hull exterior points.
        missing = interp_ids[:, 1] < 0
        if missing.any():
            dist, nn = cKDTree(pivot_pos).query(
                fine_pos[missing], k=min(interp_width, len(skeleton.pivots))
            )
            if np.ndim(dist) == 1:
                dist = dist[:, None]; nn = nn[:, None]
            ww = 1.0 / np.maximum(dist, 1e-8); ww /= ww.sum(axis=1, keepdims=True)
            interp_ids[missing, :nn.shape[1]] = skeleton.pivots[nn]
            interp_w[missing, :nn.shape[1]] = ww.astype(np.float32)
        data.pivot_interp_index = torch.full(
            (data.num_nodes, interp_width), -1, dtype=torch.long
        )
        data.pivot_interp_weight = torch.zeros(
            (data.num_nodes, interp_width), dtype=torch.float32
        )
        data.pivot_interp_index[current_offset:current_offset + current_mesh.num_vertices] = torch.tensor(interp_ids + current_offset)
        data.pivot_interp_weight[current_offset:current_offset + current_mesh.num_vertices] = torch.tensor(interp_w)
        profile_interpolation_seconds = perf_counter() - section_start
        role = torch.zeros(data.num_nodes, 2)
        role[current_offset:current_offset + current_mesh.num_vertices, 0] = torch.tensor(pivot_mask, dtype=torch.float32)
        role[current_offset:current_offset + current_mesh.num_vertices, 1] = torch.tensor(~pivot_mask, dtype=torch.float32)
        data.x = torch.cat([data.x, role], dim=1)
        # Make the residual reference an explicit local communication path.
        section_start = perf_counter()
        # These edges are added only on the current graph; quotient-M0 edges
        # remain the separate coarse communication structure.
        if use_pivot_communication:
            fine_vertices = np.flatnonzero(~pivot_mask)
            fine_owners = owners[fine_vertices]
            owner_edges = np.vstack([
                np.concatenate([fine_vertices, fine_owners]),
                np.concatenate([fine_owners, fine_vertices]),
            ]) + current_offset
            owner_attr = []
            owner_pos = current_mesh.vertex_positions
            if "euclidean_distance" in edge_feature_names:
                owner_attr.append(np.linalg.norm(
                    owner_pos[owner_edges[1] - current_offset]
                    - owner_pos[owner_edges[0] - current_offset], axis=1
                ))
            if "edge_curvature" in edge_feature_names:
                owner_attr.append(np.zeros(owner_edges.shape[1], dtype=np.float64))
            edge_indices.append(torch.tensor(owner_edges, dtype=torch.long))
            edge_attrs.append(torch.tensor(np.asarray(owner_attr).T, dtype=torch.float32))
            intra_current_masks.append(torch.ones(owner_edges.shape[1], dtype=torch.bool))
        # The quotient skeleton is an explicit pivot-pivot communication graph.
        # Build a geometric pivot topology: Delaunay in 2D, symmetric kNN in
        # 3D.  The quotient graph is deliberately not used here because its
        # hop partition can create long, crossing physical edges.
        if use_pivot_communication and current_mesh.mesh.dim() == 2 and len(skeleton.pivots) >= 3 and pivot_communication_topology == "delaunay":
            simplices = Delaunay(pivot_pos).simplices
            skeleton_edges = np.unique(np.sort(np.concatenate([
                simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]
            ], axis=0), axis=1), axis=0)
            skeleton_edges = skeleton.pivots[skeleton_edges]
        elif use_pivot_communication:
            k = min(6, max(1, len(skeleton.pivots) - 1))
            neigh = cKDTree(pivot_pos).query(pivot_pos, k=k + 1)[1][:, 1:]
            skeleton_edges = np.unique(np.sort(np.stack([
                np.repeat(np.arange(len(skeleton.pivots)), k), neigh.reshape(-1)
            ], axis=1), axis=1), axis=0)
            skeleton_edges = skeleton.pivots[skeleton_edges]
        edges_before_filter = skeleton_edges.copy() if use_pivot_communication else np.empty((0, 2), dtype=np.int64)
        if use_pivot_communication and current_mesh.mesh.dim() == 2:
            skeleton_edges = skeleton_edges[_segments_inside_2d_mesh(current_mesh, skeleton_edges)]
        if use_pivot_communication and skeleton_edges.size:
            pivot_edges = skeleton_edges + current_offset
            pivot_edges = np.concatenate([pivot_edges.T, pivot_edges[:, ::-1].T], axis=1)
            pivot_attr = []
            if "euclidean_distance" in edge_feature_names:
                pivot_attr.append(np.linalg.norm(
                    current_mesh.vertex_positions[pivot_edges[1] - current_offset]
                    - current_mesh.vertex_positions[pivot_edges[0] - current_offset], axis=1
                ))
            if "edge_curvature" in edge_feature_names:
                pivot_attr.append(np.zeros(pivot_edges.shape[1], dtype=np.float64))
            edge_indices.append(torch.tensor(pivot_edges, dtype=torch.long))
            edge_attrs.append(torch.tensor(np.asarray(pivot_attr).T, dtype=torch.float32))
            intra_current_masks.append(torch.ones(pivot_edges.shape[1], dtype=torch.bool))
        # The role edges are appended after the initial graph assembly above.
        # Rebuild edge tensors so the added communication paths are real graph
        # edges rather than only entries in local construction lists.
        data.edge_index = torch.cat(edge_indices, dim=1)
        data.edge_attr = torch.cat(edge_attrs, dim=0)
        data.edge_length = torch.norm(
            data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]], dim=1
        )
        data.intra_current_edge_mask = torch.cat(intra_current_masks, dim=0)
        profile_communication_seconds = perf_counter() - section_start
    data.current_interior_facet_elements = current_interior_facet_elements
    data.current_interior_facet_vertices = current_interior_facet_vertices
    if graph_profile_hook is not None:
        graph_profile_hook(dict(
            total_seconds=perf_counter() - profile_start,
            pivot_seconds=profile_pivot_seconds,
            interpolation_seconds=profile_interpolation_seconds,
            communication_seconds=profile_communication_seconds,
            graph=data, mesh=current_mesh, stage=current_level,
            pivots=skeleton.pivots.copy() if pivot_enabled else np.empty(0, dtype=np.int64),
            fallback=missing.copy() if pivot_enabled else np.zeros(current_mesh.num_vertices, dtype=bool),
            edges_before=edges_before_filter if pivot_enabled else np.empty((0, 2), dtype=np.int64),
            edges_after=skeleton_edges.copy() if pivot_enabled and use_pivot_communication else np.empty((0, 2), dtype=np.int64),
        ))
    return data
