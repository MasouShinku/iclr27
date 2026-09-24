"""
Mesh utilities: MeshWrapper, Gmsh mesh generation, sizing field tools.

Consolidates logic from:
- src/tasks/domains/mesh_wrapper.py
- src/tasks/domains/update_mesh.py
- src/mesh_util/sizing_field_util.py
- src/tasks/domains/geometry_util.py
"""

import os
import tempfile
import warnings
from functools import cached_property
from typing import Callable, Dict, Optional, Union

import gmsh
import numpy as np
import torch
from skfem import Basis, ElementTriP1, ElementTetP1, Mesh
from skfem.io import from_meshio


# ============================================================
# Geometry utilities
# ============================================================

def volume_to_edge_length(element_volumes: Union[float, np.ndarray], dim: int) -> np.ndarray:
    """Convert simplex volume to average edge length."""
    if dim == 2:
        return np.sqrt(4 / np.sqrt(3) * element_volumes)
    elif dim == 3:
        return np.power(12 / np.sqrt(2) * element_volumes, 1 / 3)
    raise ValueError(f"Mesh dimension {dim} not supported")


def edge_length_to_volume(sizing_field: np.ndarray, dim: int) -> np.ndarray:
    """Convert edge length to estimated element volume."""
    if dim == 2:
        return sizing_field**2 * np.sqrt(3) / 4
    elif dim == 3:
        return sizing_field**3 * np.sqrt(2) / 12
    raise ValueError(f"Dimension {dim} not supported")


def get_simplex_volumes(positions: np.ndarray, simplex_indices: np.ndarray) -> np.ndarray:
    """Compute volumes for an array of simplices (triangles or tetrahedra)."""
    if positions.shape[-1] == 2:
        # 2D: triangle areas
        a = positions[simplex_indices[:, 0]]
        b = positions[simplex_indices[:, 1]]
        c = positions[simplex_indices[:, 2]]
        area = np.abs(0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                              - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])))
        return area
    elif positions.shape[-1] == 3:
        # 3D: tetrahedron volumes
        v0 = positions[simplex_indices[:, 0]]
        v1 = positions[simplex_indices[:, 1]]
        v2 = positions[simplex_indices[:, 2]]
        v3 = positions[simplex_indices[:, 3]]
        return np.abs(np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0)
    raise ValueError(f"Cannot compute simplex volumes for {positions.shape[-1]} dimensions")


# ============================================================
# Boundary normals and curvature
# ============================================================

def _compute_boundary_vertex_normals(mesh) -> np.ndarray:
    """
    Compute averaged outward unit normals at boundary vertices.

    Returns:
        np.ndarray of shape (dim, n_boundary_nodes)
    """
    boundary_facets = mesh.boundary_facets()
    mapping = mesh.mapping()

    # Compute normals at the center of each boundary facet
    xi = np.zeros((mesh.dim() - 1, 1))
    tind = mesh.f2t[0, boundary_facets]
    normals = np.nan_to_num(
        mapping.normals(xi, tind, boundary_facets, mesh.t2f).squeeze(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Facet-to-node mapping
    facet_nodes = mesh.facets[:, boundary_facets]

    # Repeat normals for each node in the facet
    repeated_normals = np.repeat(normals, facet_nodes.shape[0], axis=1)
    flat_nodes = facet_nodes.T.reshape(-1)

    # Accumulate normals per node
    node_normals = np.zeros((mesh.dim(), mesh.p.shape[1]))
    np.add.at(node_normals, (slice(None), flat_nodes), repeated_normals)

    # Count contributions per node
    node_counts = np.bincount(flat_nodes, minlength=mesh.p.shape[1])

    # Normalize
    nonzero = node_counts > 0
    node_normals[:, nonzero] /= node_counts[nonzero]
    norms = np.linalg.norm(node_normals[:, nonzero], axis=0)
    valid_norms = np.isfinite(norms) & (norms > np.finfo(float).eps)
    boundary_columns = np.flatnonzero(nonzero)
    node_normals[:, boundary_columns[valid_norms]] /= norms[valid_norms]
    node_normals[:, boundary_columns[~valid_norms]] = 0.0

    # Restrict to boundary nodes
    boundary_nodes = mesh.boundary_nodes()
    return node_normals[:, boundary_nodes]


def _compute_curvature(mesh, boundary_normals: np.ndarray) -> np.ndarray:
    """
    Compute signed curvature at boundary edges.

    Positive = convex (outward bending), Negative = concave (inward bending).

    Returns:
        np.ndarray of shape (num_boundary_edges,)
    """
    boundary_nodes = mesh.boundary_nodes()
    boundary_vertices = mesh.p.T[boundary_nodes].T

    mesh_scale = np.max(mesh.p.max(axis=1) - mesh.p.min(axis=1))

    if mesh.dim() == 2:
        edges = mesh.facets[:, mesh.boundary_facets()]
    else:
        edges = mesh.edges[:, mesh.boundary_edges()]

    # Map global vertex indices to local boundary vertex indices
    edge_indices = np.searchsorted(boundary_nodes, edges)

    # Normals at edge endpoints
    n0 = boundary_normals[:, edge_indices[0]]
    n1 = boundary_normals[:, edge_indices[1]]

    # Angle between normals
    dot = np.clip(np.sum(n0 * n1, axis=0), -1.0, 1.0)
    angle = np.arccos(dot)

    # Edge vectors
    v0 = boundary_vertices[:, edge_indices[0]]
    v1 = boundary_vertices[:, edge_indices[1]]
    edge_vector = v1 - v0

    # Move vertices along normals to determine sign
    moved_v0 = v0 + n0 * 0.001 * mesh_scale
    moved_v1 = v1 + n1 * 0.001 * mesh_scale
    moved_edge_vector = moved_v1 - moved_v0

    original_length = np.linalg.norm(edge_vector, axis=0)
    moved_length = np.linalg.norm(moved_edge_vector, axis=0)
    sign = -np.sign(original_length - moved_length)

    return np.nan_to_num(sign * angle, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# MeshWrapper
# ============================================================

class MeshWrapper:
    """Lightweight wrapper around skfem mesh with cached properties."""

    def __init__(self, mesh):
        self._wrapped_mesh = mesh

    @property
    def mesh(self):
        return self._wrapped_mesh

    def __getattr__(self, name):
        return getattr(self._wrapped_mesh, name)

    @property
    def num_elements(self) -> int:
        return self.mesh.t.shape[1]

    @property
    def num_vertices(self) -> int:
        return self.mesh.nvertices

    @property
    def vertex_positions(self) -> np.ndarray:
        """Shape (num_vertices, dim)"""
        return self.mesh.p.T

    @property
    def element_indices(self) -> np.ndarray:
        """Shape (num_elements, vertices_per_element)"""
        return self.mesh.t.T

    @cached_property
    def element_midpoints(self) -> np.ndarray:
        """Shape (num_elements, dim)"""
        return self.vertex_positions[self.element_indices].mean(axis=1)

    @cached_property
    def simplex_volumes(self) -> np.ndarray:
        return get_simplex_volumes(
            positions=self.vertex_positions,
            simplex_indices=self.element_indices,
        )

    @property
    def mesh_edges(self) -> np.ndarray:
        """Shape (2, num_edges)"""
        if self.mesh.dim() == 2:
            return self.mesh.facets
        elif self.mesh.dim() == 3:
            return self.mesh.edges
        raise ValueError("Mesh dimension must be 2 or 3")

    @property
    def boundary_edges(self) -> np.ndarray:
        """Shape (2, num_boundary_edges) — vertex pairs for boundary edges."""
        if self.mesh.dim() == 2:
            return self.mesh.facets[:, self.mesh.boundary_facets()]
        elif self.mesh.dim() == 3:
            # boundary_edges() returns edge indices, not vertex pairs
            return self.mesh.edges[:, self.mesh.boundary_edges()]
        raise ValueError("Mesh dimension must be 2 or 3")

    @cached_property
    def element_neighbors(self) -> np.ndarray:
        """Shape (2, num_neighbor_pairs). Undirected element adjacency."""
        return self.mesh.f2t[:, self.mesh.f2t[1] != -1]

    @cached_property
    def midpoint_tree(self):
        from pykdtree.kdtree import KDTree
        return KDTree(self.element_midpoints)

    @cached_property
    def vertex_tree(self):
        from pykdtree.kdtree import KDTree
        return KDTree(self.vertex_positions)

    @cached_property
    def boundary_vertex_normals(self) -> np.ndarray:
        return _compute_boundary_vertex_normals(self.mesh)

    @cached_property
    def boundary_edge_curvatures(self) -> np.ndarray:
        return _compute_curvature(self.mesh, self.boundary_vertex_normals)

    def find_closest_elements(self, query_points: np.ndarray) -> np.ndarray:
        """
        Map each query point to one element in this mesh.
        Uses element_finder() first; falls back to nearest element midpoint
        for points outside the mesh (e.g., geometry mismatch after remeshing).

        Aligned with reference implementation: src/tasks/domains/mesh_wrapper.py
        """
        corresponding = self.find_containing_elements(query_points)

        missing = corresponding == -1
        if missing.any():
            _, candidate_indices = self.midpoint_tree.query(
                query_points[missing], k=1
            )
            corresponding[missing] = candidate_indices.astype(np.int64)
        return corresponding

    def find_containing_elements(self, query_points: np.ndarray) -> np.ndarray:
        """Find containing simplices using local KD candidates in one batch."""
        query_points = np.asarray(query_points, dtype=np.float64)
        corresponding = np.full(len(query_points), -1, dtype=np.int64)
        unresolved = np.arange(len(query_points))
        tolerance = 1e-10
        for requested in (16, 64, 256):
            if not len(unresolved):
                break
            k = min(requested, self.num_elements)
            _, candidates = self.midpoint_tree.query(query_points[unresolved], k=k)
            candidates = np.asarray(candidates, dtype=np.int64).reshape(len(unresolved), k)
            vertices = self.vertex_positions[self.element_indices[candidates]]
            transforms = (vertices[:, :, 1:] - vertices[:, :, :1]).transpose(0, 1, 3, 2)
            right = query_points[unresolved, None, :] - vertices[:, :, 0]
            coordinates = np.linalg.solve(transforms, right[..., None])[..., 0]
            barycentric = np.concatenate(
                [1.0 - coordinates.sum(axis=2, keepdims=True), coordinates], axis=2
            )
            inside = np.all(barycentric >= -tolerance, axis=2)
            found = inside.any(axis=1)
            if found.any():
                first = inside[found].argmax(axis=1)
                corresponding[unresolved[found]] = candidates[found, first]
            unresolved = unresolved[~found]
        return corresponding

    def __repr__(self):
        return f"MeshWrapper(nv={self.num_vertices}, ne={self.num_elements})"


# ============================================================
# Sizing field utilities
# ============================================================

def get_sizing_field_at_vertices(mesh: MeshWrapper) -> np.ndarray:
    """Compute sizing field (edge length) at mesh vertices."""
    element_volumes = mesh.simplex_volumes
    element_edge_lengths = volume_to_edge_length(element_volumes, mesh.dim())
    return project_elements_to_vertices(mesh, element_edge_lengths)


def get_sizing_field_at_elements(mesh: MeshWrapper) -> np.ndarray:
    """Compute sizing field (edge length) at mesh elements."""
    element_volumes = mesh.simplex_volumes
    return volume_to_edge_length(element_volumes, mesh.dim())


def get_sizing_field(mesh: MeshWrapper, mesh_node_type: str = "vertex") -> np.ndarray:
    """Compute sizing field at vertices or elements."""
    if mesh_node_type == "vertex":
        return get_sizing_field_at_vertices(mesh)
    elif mesh_node_type == "element":
        return get_sizing_field_at_elements(mesh)
    raise ValueError(f"Unknown node type: {mesh_node_type}")


def project_elements_to_vertices(mesh: MeshWrapper, element_values: np.ndarray) -> np.ndarray:
    """Project element-wise values to vertices via volume-weighted averaging."""
    from torch_scatter import scatter_add

    volumes = mesh.simplex_volumes
    # Repeat for each vertex in each element
    n_verts_per_elem = mesh.mesh.t.shape[0]
    volumes_rep = torch.tensor(np.repeat(volumes, n_verts_per_elem), dtype=torch.float64)
    values_rep = torch.tensor(np.repeat(element_values, n_verts_per_elem), dtype=torch.float64)
    index = torch.tensor(mesh.mesh.t.T.flatten(), dtype=torch.int64)

    vertex_sums = scatter_add(src=values_rep * volumes_rep, index=index, dim=0, dim_size=mesh.num_vertices)
    vertex_weights = scatter_add(src=volumes_rep, index=index, dim=0, dim_size=mesh.num_vertices)
    result = (vertex_sums / vertex_weights).numpy()
    return result


def project_sizing_field(from_mesh: MeshWrapper, to_mesh: MeshWrapper) -> np.ndarray:
    """Project a volume-weighted vertex sizing field with P1 interpolation."""
    source_vertex_sizing = get_sizing_field_at_vertices(from_mesh)
    return project_vertex_field(from_mesh, source_vertex_sizing, to_mesh)


def sample_element_sizing_at_vertices(
    from_mesh: MeshWrapper, to_mesh: MeshWrapper
) -> np.ndarray:
    """AMBER sampled-vertex labels from containing expert elements."""
    source_element_sizing = get_sizing_field_at_elements(from_mesh)
    source_elements = from_mesh.find_closest_elements(to_mesh.vertex_positions)
    return source_element_sizing[source_elements]


def project_vertex_field(from_mesh: MeshWrapper, from_values: np.ndarray,
                         to_mesh: MeshWrapper) -> np.ndarray:
    """Interpolate a P1 finite-element scalar field onto another mesh's vertices."""
    values = np.asarray(from_values).reshape(-1)
    if values.shape[0] != from_mesh.num_vertices:
        raise ValueError(
            "Vertex field length does not match source mesh: "
            f"{values.shape[0]} != {from_mesh.num_vertices}"
        )
    dim = from_mesh.dim()
    simplex_size = from_mesh.mesh.t.shape[0]
    if not ((dim == 2 and simplex_size == 3) or (dim == 3 and simplex_size == 4)):
        raise ValueError("P1 vertex-field interpolation requires triangles or tetrahedra")
    points = np.asarray(to_mesh.mesh.p).T
    cells = from_mesh.find_containing_elements(points)
    projected = np.empty(len(points), dtype=np.float64)
    valid = cells >= 0
    if valid.any():
        simplices = from_mesh.mesh.t[:, cells[valid]].T
        vertices = from_mesh.vertex_positions[simplices]
        transforms = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
        coordinates = np.linalg.solve(
            transforms, (points[valid] - vertices[:, 0])[..., None]
        )[..., 0]
        barycentric = np.concatenate(
            [1.0 - coordinates.sum(axis=1, keepdims=True), coordinates], axis=1
        )
        projected[valid] = np.sum(values[simplices] * barycentric, axis=1)
    if (~valid).any():
        _, nearest = from_mesh.vertex_tree.query(points[~valid], k=1)
        projected[~valid] = values[np.asarray(nearest, dtype=np.int64)]
    if not np.all(np.isfinite(projected)):
        raise ValueError("P1 vertex-field interpolation produced non-finite values")
    return projected


def compute_dcd_midpoint(generated_mesh: MeshWrapper, expert_mesh: MeshWrapper) -> float:
    """Density-aware Chamfer distance between element midpoint sets."""
    from scipy.spatial import KDTree

    generated_midpoints = generated_mesh.element_midpoints
    expert_midpoints = expert_mesh.element_midpoints
    generated_tree = KDTree(generated_midpoints)
    expert_tree = KDTree(expert_midpoints)

    generated_distances, generated_to_expert = expert_tree.query(
        generated_midpoints, k=1
    )
    expert_distances, expert_to_generated = generated_tree.query(
        expert_midpoints, k=1
    )

    expert_match_counts = np.bincount(
        generated_to_expert, minlength=len(expert_midpoints)
    ).astype(np.float64)
    generated_match_counts = np.bincount(
        expert_to_generated, minlength=len(generated_midpoints)
    ).astype(np.float64)
    generated_weights = (
        np.exp(-generated_distances)
        / expert_match_counts[generated_to_expert]
    )
    expert_weights = (
        np.exp(-expert_distances)
        / generated_match_counts[expert_to_generated]
    )
    return float(
        0.5
        * (
            np.mean(1.0 - generated_weights)
            + np.mean(1.0 - expert_weights)
        )
    )


def estimate_num_elements(mesh: MeshWrapper, sizing_field: np.ndarray,
                          node_type: str = "vertex") -> float:
    """Estimate number of elements from sizing field."""
    dim = mesh.dim()
    simplex_volumes = mesh.simplex_volumes

    if node_type == "vertex":
        # Smooth using gradation factor
        edges = mesh.mesh_edges
        pos0 = mesh.vertex_positions[edges[0]]
        pos1 = mesh.vertex_positions[edges[1]]
        edge_distances = np.linalg.norm(pos0 - pos1, axis=1)
        sf = sizing_field.copy()
        gradation = 1.3

        for _ in range(2):
            sizing0 = sf[edges[0]]
            sizing1 = sf[edges[1]]
            n_forward = edge_distances / sizing0
            min_sizing1 = sizing0 * gradation ** (-n_forward)
            n_backward = edge_distances / sizing1
            min_sizing0 = sizing1 * gradation ** (-n_backward)
            np.maximum.at(sf, edges[0], min_sizing0)
            np.maximum.at(sf, edges[1], min_sizing1)

        element_sf = sf[mesh.mesh.t].mean(axis=0)
        element_predicted_volumes = edge_length_to_volume(element_sf, dim)
        density = simplex_volumes / element_predicted_volumes
        # Use the same simplex-volume estimate in 2D and 3D.  A separate
        # empirical 3D multiplier changes the element budget without changing
        # the predicted sizing field and biases Beam3D toward under-resolution.
        return density.sum()
    elif node_type == "element":
        predicted_volumes = edge_length_to_volume(sizing_field, dim)
        return (simplex_volumes / predicted_volumes).sum()
    raise ValueError(f"Unsupported node_type: {node_type}")


def scale_sizing_field_to_budget(
    sizing_field: np.ndarray,
    mesh: MeshWrapper,
    max_elements: float,
    node_type: str = "vertex",
    tolerance: float = 0.99,
    max_iter: int = 20,
) -> np.ndarray:
    """Scale sizing field via binary search to fit element budget."""
    lower, upper = 0.0, 1.0
    inverse_scaling = 1.0

    for _ in range(max_iter):
        inverse_scaling = (lower + upper) / 2
        current = estimate_num_elements(mesh, sizing_field / inverse_scaling, node_type)
        if current > max_elements:
            upper = inverse_scaling
        elif current < max_elements * tolerance:
            lower = inverse_scaling
        else:
            break

    return sizing_field / inverse_scaling


# ============================================================
# Gmsh mesh generation
# ============================================================

class _GmshSession:
    """Context manager for Gmsh session."""

    def __init__(self, gmsh_kwargs: Optional[Dict] = None):
        self.gmsh_kwargs = gmsh_kwargs or {}

    def __enter__(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_kwargs.get("min_sizing_field", 1e-6))
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.gmsh_kwargs.get("max_sizing_field", 10))
        gmsh.model.add("model")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gmsh.model.remove()
        gmsh.finalize()
        return False


def _write_sizing_field_pos(positions: np.ndarray, sizing_field: np.ndarray, tmpfile):
    """Write sizing field to .pos file (Gmsh PostView format)."""
    if positions.ndim == 3:
        # Simplex vertices: shape (num_elements, vertices_per_simplex, dim)
        if positions.shape[-1] == 2:
            positions = np.concatenate((positions, np.zeros(positions.shape[:-1] + (1,))), axis=-1)

        if sizing_field.ndim == 1:
            sizing_field = np.repeat(sizing_field[:, None], positions.shape[1], axis=1)

        if positions.shape[1] == 3:  # triangles
            lines = [
                f"ST({p[0][0]},{p[0][1]},{p[0][2]},"
                f"{p[1][0]},{p[1][1]},{p[1][2]},"
                f"{p[2][0]},{p[2][1]},{p[2][2]})"
                f"{{{s[0]},{s[1]},{s[2]}}};"
                for p, s in zip(positions, sizing_field)
            ]
        else:  # tetrahedra
            lines = [
                f"SS({p[0][0]},{p[0][1]},{p[0][2]},"
                f"{p[1][0]},{p[1][1]},{p[1][2]},"
                f"{p[2][0]},{p[2][1]},{p[2][2]},"
                f"{p[3][0]},{p[3][1]},{p[3][2]})"
                f"{{{s[0]},{s[1]},{s[2]},{s[3]}}};"
                for p, s in zip(positions, sizing_field)
            ]
    elif positions.ndim == 2:
        # Point positions: shape (num_points, dim)
        if positions.shape[-1] == 2:
            positions = np.concatenate((positions, np.zeros((len(positions), 1))), axis=-1)
        lines = [
            f"SP({p[0]},{p[1]},{p[2]}){{{s}}};"
            for p, s in zip(positions, sizing_field)
        ]
    else:
        raise ValueError(f"Unsupported positions shape: {positions.shape}")

    tmpfile.write(b'View "sizing_field" {\n')
    chunk_size = 10000
    for i in range(0, len(lines), chunk_size):
        content = "\n".join(lines[i : i + chunk_size]) + "\n"
        tmpfile.write(content.encode("utf-8"))
    tmpfile.write(b"};\n")


def update_mesh(
    mesh: MeshWrapper,
    sizing_field: np.ndarray,
    geom_fn: Callable,
    gmsh_kwargs: Optional[Dict] = None,
    algorithm_idx: int = 6,
    sizing_field_positions: Optional[np.ndarray] = None,
) -> MeshWrapper:
    """
    Refine mesh using Gmsh with a sizing field.

    Args:
        mesh: Current mesh
        sizing_field: Target edge lengths at vertices or elements
        geom_fn: Function that creates the Gmsh geometry (returns pygmsh Geometry)
        gmsh_kwargs: Min/max sizing field bounds
        algorithm_idx: Gmsh meshing algorithm (6=Delaunay)

    Returns:
        New MeshWrapper with refined mesh
    """
    raw_mesh = mesh.mesh
    dimension = raw_mesh.dim()

    if sizing_field_positions is None:
        # Existing vertex/element field behavior.
        sizing_field_positions = raw_mesh.p[:, raw_mesh.t].T
        if len(sizing_field) == raw_mesh.nvertices:
            sizing_field = sizing_field[raw_mesh.t].T
        assert len(sizing_field) == len(sizing_field_positions), \
            f"Sizing field shape mismatch: {sizing_field.shape} vs {sizing_field_positions.shape}"
    else:
        # ImageAMBER predicts at active pixel centers rather than mesh nodes.
        sizing_field_positions = np.asarray(sizing_field_positions)
        sizing_field = np.asarray(sizing_field).reshape(-1)
        if sizing_field_positions.ndim != 2 or sizing_field_positions.shape[1] != dimension:
            raise ValueError(
                "sizing_field_positions must have shape (N, mesh_dimension)"
            )
        if len(sizing_field) != len(sizing_field_positions):
            raise ValueError(
                "Sizing field and sizing-field positions differ in length: "
                f"{len(sizing_field)} != {len(sizing_field_positions)}"
            )

    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    _write_sizing_field_pos(sizing_field_positions, sizing_field, tmpfile)
    tmpfile_path = tmpfile.name
    tmpfile.close()

    try:
        with _GmshSession(gmsh_kwargs):
            geom = geom_fn()
            gmsh.model.occ.synchronize()
            gmsh.merge(tmpfile_path)
            gmsh.model.mesh.field.add("PostView", 1)
            gmsh.model.mesh.field.setNumber(1, "ViewIndex", 0)
            gmsh.model.mesh.field.setAsBackgroundMesh(1)
            m = geom.generate_mesh(dim=dimension, verbose=False, algorithm=algorithm_idx)
    finally:
        os.remove(tmpfile_path)

    new_skfem_mesh = from_meshio(m)
    if hasattr(raw_mesh, "convert_new_mesh"):
        new_skfem_mesh = raw_mesh.convert_new_mesh(new_mesh=new_skfem_mesh)
    return MeshWrapper(new_skfem_mesh)
