"""Topology-only quotient graph from hop-FPS pivots."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


class HopCoarseGraph:
    def __init__(self, mesh, sizing, ratio):
        if not 0 < ratio <= 1:
            raise ValueError('pivot ratio must lie in (0, 1]')
        n = mesh.num_vertices
        edges = mesh.mesh_edges
        adjacency = coo_matrix((np.ones(2 * edges.shape[1]),
            (np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]])), shape=(n, n)).tocsr()
        nearest = np.full(n, np.inf)
        owner = np.zeros(n, dtype=np.int64)
        selected = []
        for index in range(int(np.ceil(ratio * n))):
            score = nearest.copy()
            score[selected] = -1
            pivot = int(np.argmin(sizing)) if not selected else int(np.argmax(score))
            selected.append(pivot)
            distance = dijkstra(adjacency, indices=pivot, directed=False, unweighted=True)
            improved = distance < nearest
            owner[improved] = index
            nearest[improved] = distance[improved]
        if not np.isfinite(nearest).all():
            raise ValueError('Pivot budget does not cover all connected components')
        self.pivots = np.asarray(selected)
        self.fine_owner = owner
        self.cover_hops = nearest
        self.vertex_positions = mesh.vertex_positions[self.pivots]
        self.num_vertices = len(selected)
        pairs = np.sort(owner[edges].T, axis=1)
        self.mesh_edges = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0).T
        self.boundary_edges = np.empty((2, 0), dtype=np.int64)
        self.boundary_edge_curvatures = np.empty(0)
