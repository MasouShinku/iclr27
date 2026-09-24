"""Dual-task mesh generation."""

import os
if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, MultivariateNormal
import gmsh
import pygmsh
from skfem import MeshTri1, Basis, LinearForm, asm, condense, solve, adaptive_theta, Functional, InteriorFacetBasis
from skfem.models import laplace
from skfem import ElementTriP1
from skfem.helpers import grad
from skfem.io import from_meshio


# Configuration
CONFIG = {
    "domain_type": "lshape",  # "lshape"  "lattice"
    "maximum_position_distortion": 0.3,

    # GMM 
    "num_components": 2,  # 2
    "lower_covariance_bound": 0.0001,
    "upper_covariance_bound": 0.0005,
    "mean_position_range": 0.5,

    "max_initial_element_volume": 0.001,
    "step0_element_volume": 0.01,

    "refinement_steps": 50,
    "error_threshold": 0.85,

    # Lattice 
    "lattice_n_holes": 3,
    "lattice_hole_size": 0.1,
}


# Section
class GMMDensity:
    """"""

    def __init__(self, bbox: np.ndarray, rng: np.random.RandomState, config: dict):
        self.bbox = bbox
        num_comp = config["num_components"]
        cov_range = np.array([config["lower_covariance_bound"], config["upper_covariance_bound"]])
        mean_range = np.array([0.5 - config["mean_position_range"], 0.5 + config["mean_position_range"]])

        weights = 1 + np.exp(rng.normal(size=num_comp))
        weights /= weights.sum()

        means = rng.uniform(mean_range[0], mean_range[1], (num_comp, 2))
        means = means * (bbox[2:] - bbox[:2]) + bbox[:2]  # 

        diag_cov = np.exp(rng.uniform(np.log(cov_range[0]), np.log(cov_range[1]), (num_comp, 2)))
        angles = rng.random(num_comp) * 2 * np.pi

        rot = np.zeros((num_comp, 2, 2))
        rot[:, 0, 0] = np.cos(angles)
        rot[:, 0, 1] = -np.sin(angles)
        rot[:, 1, 0] = np.sin(angles)
        rot[:, 1, 1] = np.cos(angles)
        cov = np.einsum('ijk,ikl,iml->ijm', rot, np.eye(2) * diag_cov[:, None, :], rot)

        # PyTorch GMM
        self._gmm = MixtureSameFamily(
            Categorical(torch.tensor(weights, dtype=torch.float64)),
            Independent(MultivariateNormal(torch.tensor(means, dtype=torch.float64),
                                           torch.tensor(cov, dtype=torch.float64)), 0)
        )

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        return torch.exp(self._gmm.log_prob(torch.tensor(samples, dtype=torch.float64))).numpy()


# Geometry
def lshape_geometry(hole_pos: np.ndarray = None) -> callable:
    """L"""
    if hole_pos is None:
        hole_pos = np.array([0.5, 0.5])
    nodes = np.array([[0, 0], [1, 0], [1, hole_pos[1]], [hole_pos[0], hole_pos[1]],
                      [hole_pos[0], 1], [0, 1]], dtype=np.float64)

    def geom_fn():
        geom = pygmsh.occ.Geometry()
        geom.add_polygon(nodes)
        return geom
    return geom_fn


def lattice_geometry(n_holes: int, hole_size: float) -> callable:
    """"""
    def geom_fn():
        geom = pygmsh.occ.Geometry()
        outer = geom.add_rectangle([0, 0, 0], a=1.0, b=1.0)
        spacing = (1.0 - n_holes * hole_size) / (n_holes + 1)
        holes = [geom.add_rectangle([spacing + i * (hole_size + spacing),
                                     spacing + j * (hole_size + spacing), 0],
                                    a=hole_size, b=hole_size)
                 for i in range(n_holes) for j in range(n_holes)]
        geom.boolean_difference([outer], holes)
        return geom
    return geom_fn


# ============== Mesh generation ==============
class gmsh_session:
    def __enter__(self):
        gmsh.initialize()
        for opt in ["General.Terminal", "General.Verbosity", "Mesh.MeshSizeExtendFromBoundary",
                    "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"]:
            gmsh.option.setNumber(opt, 0)
        gmsh.model.add("model")
        return self

    def __exit__(self, *args):
        gmsh.model.remove()
        gmsh.finalize()
        return False


def generate_mesh(geom_fn: callable, elem_size: float) -> MeshTri1:
    with gmsh_session():
        geom = geom_fn()
        gmsh.model.occ.synchronize()
        geom.characteristic_length_max = elem_size
        geom.characteristic_length_min = 0.7 * elem_size
        return from_meshio(geom.generate_mesh(dim=2, verbose=False))


def make_poisson_problem(seed: int, config: dict):
    """Recreate the geometry and load deterministically without AMR."""
    rng = np.random.RandomState(seed)
    if config["domain_type"] == "lshape":
        dist = config["maximum_position_distortion"]
        hole_pos = np.clip(np.array([0.5, 0.5]) + rng.uniform(-dist, dist, 2), 0.3, 0.7)
        geom_fn = lshape_geometry(hole_pos)
    else:
        geom_fn = lattice_geometry(config["lattice_n_holes"], config["lattice_hole_size"])
    # Both supported geometries are subsets of the unit square with its full
    # outer boundary.  The legacy diagnostic mesh therefore always produced
    # this same bounding box, but required an unnecessary Gmsh invocation.
    bbox = np.array([0.0, 0.0, 1.0, 1.0])
    return GMMDensity(bbox, rng, config), geom_fn


def volume_to_edge_length(vol: float) -> float:
    return np.sqrt(4 / np.sqrt(3) * vol)


# Solver
def solve_poisson(mesh: MeshTri1, load_fn: GMMDensity) -> np.ndarray:
    """ -Δu = f"""
    basis = Basis(mesh, ElementTriP1())

    def load(v, w):
        return load_fn.evaluate(np.stack(w.x, axis=-1)) * v

    K = asm(laplace, basis)
    f = asm(LinearForm(load), basis)
    return solve(*condense(K, f, I=mesh.interior_nodes()))


def get_error_indicator(mesh: MeshTri1, solution: np.ndarray, load_fn: GMMDensity) -> np.ndarray:
    """ =  + """
    basis = Basis(mesh, ElementTriP1())

    @Functional
    def interior(w):
        return w.h**2 * load_fn.evaluate(np.stack(w.x, axis=-1))**2

    eta = interior.elemental(basis)

    fb = [InteriorFacetBasis(mesh, ElementTriP1(), side=i) for i in [0, 1]]
    w = {f"u{i+1}": fb[i].interpolate(solution) for i in [0, 1]}

    @Functional
    def jump(w):
        n = w.n
        du1, du2 = grad(w["u1"]), grad(w["u2"])
        return w.h * ((du1[0] - du2[0]) * n[0] + (du1[1] - du2[1]) * n[1])**2

    eta_e = jump.elemental(fb[0], **w)
    tmp = np.zeros(mesh.facets.shape[1])
    np.add.at(tmp, fb[0].find, eta_e)
    eta += np.sum(0.5 * tmp[mesh.t2f], axis=0)

    return eta


# Refinement
def generate_expert_mesh(mesh: MeshTri1, load_fn: GMMDensity, config: dict) -> MeshTri1:
    for _ in range(config["refinement_steps"]):
        sol = solve_poisson(mesh, load_fn)
        err = get_error_indicator(mesh, sol, load_fn)
        mesh = mesh.refined(adaptive_theta(err, config["error_threshold"])).smoothed()
    return mesh


def generate_amr_trajectory(mesh: MeshTri1, load_fn: GMMDensity, config: dict) -> list:
    """Return meshes [M0, M1, ..., M_refinement_steps] from skfem AMR."""
    meshes = [mesh]
    for _ in range(config["refinement_steps"]):
        sol = solve_poisson(mesh, load_fn)
        err = get_error_indicator(mesh, sol, load_fn)
        mesh = mesh.refined(adaptive_theta(err, config["error_threshold"])).smoothed()
        meshes.append(mesh)
    return meshes


# Utilities
def get_simplex_volumes(mesh: MeshTri1) -> np.ndarray:
    p, t = mesh.p.T, mesh.t.T
    return np.abs(0.5 * ((p[t[:, 1], 0] - p[t[:, 0], 0]) * (p[t[:, 2], 1] - p[t[:, 0], 1])
                       - (p[t[:, 2], 0] - p[t[:, 0], 0]) * (p[t[:, 1], 1] - p[t[:, 0], 1])))


def project_to_vertices(mesh: MeshTri1, elem_vals: np.ndarray) -> np.ndarray:
    vols = get_simplex_volumes(mesh)
    sums, weights = np.zeros(mesh.p.shape[1]), np.zeros(mesh.p.shape[1])
    for i, elem in enumerate(mesh.t.T):
        for v in elem:
            sums[v] += elem_vals[i] * vols[i]
            weights[v] += vols[i]
    return sums / weights


# Data Generation
def generate_data(seed: int, config: dict, verbose=True) -> dict:
    rng = np.random.RandomState(seed)

    if config["domain_type"] == "lshape":
        dist = config["maximum_position_distortion"]
        hole_pos = np.clip(np.array([0.5, 0.5]) + rng.uniform(-dist, dist, 2), 0.3, 0.7)
        geom_fn = lshape_geometry(hole_pos)
    else:
        geom_fn = lattice_geometry(config["lattice_n_holes"], config["lattice_hole_size"])

    init_mesh = generate_mesh(geom_fn, volume_to_edge_length(config["max_initial_element_volume"]))
    bbox = np.concatenate([init_mesh.p.min(1), init_mesh.p.max(1)])

    load_fn = GMMDensity(bbox, rng, config)

    if verbose:
        print(f"[Seed {seed}] domain={config['domain_type']}, init_elem={init_mesh.nelements}")

    coarse = generate_mesh(geom_fn, volume_to_edge_length(config["step0_element_volume"]))
    expert = generate_expert_mesh(coarse, load_fn, config)

    sol = solve_poisson(expert, load_fn)
    load_vals = load_fn.evaluate(expert.p.T)

    # Sizing field
    sizing = project_to_vertices(expert, volume_to_edge_length(get_simplex_volumes(expert)))

    return {
        "seed": seed, "initial_mesh": init_mesh, "expert_mesh": expert,
        "solution": sol, "load": load_vals, "sizing_field": sizing, "load_fn": load_fn,
        "geom_fn": geom_fn,
    }


def generate_trajectory_data(seed: int, config: dict, levels: list = None, verbose=True) -> dict:
    """Generate a Poisson AMR trajectory sampled at selected refinement levels.

    M0 is the diagnostic initial mesh used by the original mars learner.  AMR
    starts from that same mesh so every sampled transition is monotone.
    """
    load_fn, geom_fn = make_poisson_problem(seed, config)

    initial = generate_mesh(geom_fn, volume_to_edge_length(config["max_initial_element_volume"]))
    all_meshes = generate_amr_trajectory(initial, load_fn, config)

    if levels is None:
        levels = [0, config["refinement_steps"]]
    levels = sorted(set(int(l) for l in levels))
    if levels[0] != 0:
        raise ValueError("Trajectory levels must include 0.")
    if levels[-1] > config["refinement_steps"]:
        raise ValueError("Trajectory level exceeds refinement_steps.")
    if levels[-1] != config["refinement_steps"]:
        raise ValueError(
            "Trajectory levels must end at refinement_steps "
            f"({config['refinement_steps']}), got {levels[-1]}."
        )

    meshes = [all_meshes[l] for l in levels]
    final_mesh = meshes[-1]
    final_solution = solve_poisson(final_mesh, load_fn)

    if verbose:
        sizes = ", ".join(f"{l}:{m.nelements}" for l, m in zip(levels, meshes))
        print(f"[Seed {seed}] trajectory levels/elements: {sizes}")

    return {
        "seed": seed,
        "levels": levels,
        "meshes": meshes,
        "all_meshes": all_meshes,
        "initial_mesh": meshes[0],
        "expert_mesh": final_mesh,
        "u_final": final_solution,
        "load_fn": load_fn,
        "geom_fn": geom_fn,
    }


def main():
    print("=" * 50)
    print("Poisson  - 10")
    print("=" * 50)

    stats = {"nodes": [], "edges": [], "elements": []}

    for i, seed in enumerate(range(42, 52), 1):  # seeds: 42-51 (10)
        print(f"\n[{i}/10] Generating seed {seed}...")
        data = generate_data(seed, CONFIG, verbose=False)
        mesh = data['expert_mesh']

        stats["nodes"].append(mesh.p.shape[1])
        stats["edges"].append(mesh.facets.shape[1])
        stats["elements"].append(mesh.nelements)

        print(f"  Nodes: {stats['nodes'][-1]}, Edges: {stats['edges'][-1]}, Elements: {stats['elements'][-1]}")

    print("\n" + "=" * 50)
    print(" (Expert Mesh)")
    print("=" * 50)
    for key in ["nodes", "edges", "elements"]:
        vals = np.array(stats[key])
        print(f"{key.capitalize()}:")
        print(f"  Mean: {vals.mean():.2f}")
        print(f"  Min:  {vals.min()}")
        print(f"  Max:  {vals.max()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
