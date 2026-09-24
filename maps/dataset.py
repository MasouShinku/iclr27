"""Progressive AMR trajectory dataset and graph batcher."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Batch

from config import Config
from data_generator.poisson_data_generator import CONFIG as POISSON_CONFIG
from data_generator.poisson_data_generator import generate_trajectory_data, make_poisson_problem
from graph_util import build_multilevel_graph
from mesh_util import (
    MeshWrapper,
    get_sizing_field,
    project_sizing_field,
    project_vertex_field,
    sample_element_sizing_at_vertices,
    update_mesh,
)


@dataclass
class Trajectory:
    seed: int
    levels: List[int]
    meshes: List[MeshWrapper]
    u_final: np.ndarray
    load_fn: object
    geom_fn: object
    task_name: str = "poisson"


def generate_trajectories(config: Config, seeds: List[int], split_name: str) -> List[Trajectory]:
    if config.task_name != "poisson":
        raise ValueError("This repository includes only the Poisson data generator.")
    if config.trajectory_levels != [0, 1, 2, 3]:
        raise ValueError("MAPS uses K=3 with trajectory_levels=[0,1,2,3].")
    pconf = dict(POISSON_CONFIG)
    pconf["refinement_steps"] = config.refinement_steps
    trajectories = []
    for i, seed in enumerate(seeds):
        raw = generate_trajectory_data(seed, pconf, levels=[0, config.refinement_steps], verbose=False)
        meshes = sequential_target_meshes(
            MeshWrapper(raw["meshes"][0]), MeshWrapper(raw["meshes"][-1]), raw["geom_fn"]
        )
        traj = Trajectory(
            seed=seed,
            levels=[0, 1, 2, 3],
            meshes=meshes,
            u_final=raw["u_final"],
            load_fn=raw["load_fn"],
            geom_fn=raw["geom_fn"],
        )
        trajectories.append(traj)
        sizes = " -> ".join(f"L{l}:{m.num_vertices}v/{m.num_elements}e" for l, m in zip(traj.levels, traj.meshes))
        print(f"  [{split_name}] {i + 1}/{len(seeds)} seed={seed} {sizes}")
    return trajectories


def sequential_target_meshes(initial, expert, geom_fn):
    """Re-query expert sizes on each newly generated mesh; retain both endpoints."""
    meshes = [initial]
    for k in range(2):
        current = meshes[-1]
        h = get_sizing_field(current)
        h_expert = project_sizing_field(expert, current)
        h_generation = h + (h_expert - h) / (3 - k)
        meshes.append(update_mesh(current, h_generation, geom_fn))
    meshes.append(expert)
    return meshes


def trajectory_split_seeds(config: Config) -> Dict[str, List[int]]:
    """Create the deterministic train/validation/test seed split."""
    rng = np.random.RandomState(config.seed)
    if config.same_sample_splits:
        shared_seed = config.split_seed if config.split_seed >= 0 else int(rng.randint(0, 2**31))
        return {
            "train": [shared_seed] * config.num_train,
            "val": [shared_seed] * config.num_val,
            "test": [shared_seed] * config.num_test,
        }
    total = config.num_train + config.num_val + config.num_test
    seeds = rng.randint(0, 2**31, size=total).tolist()
    return {
        "train": seeds[:config.num_train],
        "val": seeds[config.num_train:config.num_train + config.num_val],
        "test": seeds[config.num_train + config.num_val:],
    }


def save_trajectory_splits(config: Config, directory: str) -> None:
    """Generate and persist raw AMR trajectories for later graph rebuilding."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    splits = trajectory_split_seeds(config)
    manifest = {
        "format_version": 2,
        "initial_mesh_mode": "diagnostic_amr",
        "target_mesh_mode": "sequential_current_expert_k3_v1",
        "task_name": config.task_name,
        "seed": config.seed,
        "same_sample_splits": config.same_sample_splits,
        "split_seed": config.split_seed,
        "refinement_steps": config.refinement_steps,
        "trajectory_levels": config.trajectory_levels,
        "split_seeds": splits,
    }
    shared = {}
    for split_name, seeds in splits.items():
        for seed in seeds:
            if seed not in shared:
                shared[seed] = generate_trajectories(config, [seed], split_name)[0]
        trajectories = [shared[seed] for seed in seeds]
        records = [
            {
                "seed": t.seed,
                "levels": t.levels,
                "meshes": [m.mesh for m in t.meshes],
                "u_final": t.u_final,
                "load_fn": t.load_fn,
                "task_name": t.task_name,
            }
            for t in trajectories
        ]
        with (root / f"{split_name}.pkl").open("wb") as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def load_trajectory_splits(
    config: Config,
    directory: str,
    split_names=("train", "val", "test"),
) -> Dict[str, List[Trajectory]]:
    """Load cached meshes and reconstruct deterministic load/geometry functions."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Trajectory manifest not found: {manifest_path}")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("target_mesh_mode") != "sequential_current_expert_k3_v1":
        raise ValueError("Expected sequential K=3 targets; old AMR snapshot caches are not compatible.")
    if manifest.get("format_version") not in {2, 3}:
        raise ValueError(
            "Trajectory cache uses the old coarse-mesh format; regenerate it "
            "with generate_poisson_dataset.py."
        )
    expected = {
        "task_name": config.task_name,
        "seed": config.seed,
        "same_sample_splits": config.same_sample_splits,
        "split_seed": config.split_seed,
        "refinement_steps": config.refinement_steps,
        "trajectory_levels": config.trajectory_levels,
    }
    if manifest.get("format_version") == 3:
        expected["task_name"] = config.task_name
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        details = ", ".join(f"{key}: cache={manifest.get(key)!r}, config={expected[key]!r}" for key in mismatches)
        raise ValueError(f"Trajectory cache is incompatible with the current config: {details}")

    pconf = dict(POISSON_CONFIG)
    pconf["refinement_steps"] = config.refinement_steps
    loaded = {}
    for split_name in split_names:
        if split_name not in {"train", "val", "test"}:
            raise ValueError(f"Unknown trajectory split: {split_name}")
        path = root / f"{split_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Trajectory split not found: {path}")
        with path.open("rb") as handle:
            records = pickle.load(handle)
        expected_seeds = manifest["split_seeds"][split_name]
        cached_seeds = [record["seed"] for record in records]
        if cached_seeds != expected_seeds:
            raise ValueError(f"Cached {split_name} seeds do not match its manifest.")
        trajectories = []
        for record in records:
            load_fn = record["load_fn"]
            if manifest.get("format_version") == 3:
                geom_fn = record["geom_fn"]
                task_name = record.get("task_name", config.task_name)
            else:
                _, geom_fn = make_poisson_problem(record["seed"], pconf)
                task_name = "poisson"
            trajectories.append(Trajectory(
                seed=record["seed"],
                levels=record["levels"],
                meshes=[MeshWrapper(mesh) for mesh in record["meshes"]],
                u_final=record["u_final"],
                load_fn=load_fn,
                geom_fn=geom_fn,
                task_name=task_name,
            ))
        loaded[split_name] = trajectories
        print(f"Loaded {split_name} trajectories: {len(trajectories)} from {path}")
    return loaded


class ProgressiveSample:
    def __init__(self, trajectory: Trajectory, k: int, config: Config,
                 mesh: MeshWrapper = None, next_mesh: MeshWrapper = None,
                 is_online: bool = False):
        self.trajectory = trajectory
        self.k = k
        self.config = config
        self.mesh = trajectory.meshes[k] if mesh is None else mesh
        if next_mesh is not None:
            self.next_mesh = next_mesh
        else:
            self.next_mesh = trajectory.meshes[k + 1] if k + 1 < len(trajectory.meshes) else None
        self.final_mesh = trajectory.meshes[-1]
        self.is_online = is_online
        self.graph = self._build_graph()

    @property
    def graph_size(self) -> int:
        return self.graph.num_nodes + self.graph.num_edges

    def _build_graph(self):
        graph = build_multilevel_graph(
            self.trajectory.meshes[0],
            self.mesh,
            self.trajectory.load_fn,
            current_level=self.k,
            max_levels=len(self.trajectory.meshes),
            edge_feature_names=self.config.edge_features,
            use_fem_solution_input=self.config.use_fem_solution_input,
            use_multilevel_graph=self.config.use_multilevel_graph,
            task_name=self.trajectory.task_name,
            use_pivot_shortcuts=self.config.use_pivot_shortcuts,
            pivot_ratio=self.config.pivot_ratio,
            pivot_alpha=self.config.pivot_alpha,
            pivot_neighbors=self.config.pivot_neighbors,
            use_hop_coarse_graph=self.config.use_hop_coarse_graph,
            use_pivot_residual_heads=self.config.use_pivot_residual_heads,
            use_pivot_graph=self.config.use_pivot_graph,
            use_pivot_communication=self.config.use_pivot_communication,
            pivot_communication_topology=getattr(self.config, "pivot_communication_topology", "delaunay"),
        )
        u_target = project_vertex_field(self.final_mesh, self.trajectory.u_final, self.mesh)

        h_current = get_sizing_field(self.mesh)
        sizing_target_mesh = self.next_mesh
        if self.config.sizing_target_mode == "final" and self.next_mesh is not None:
            sizing_target_mesh = self.final_mesh
        elif self.config.sizing_target_mode != "next":
            raise ValueError(
                f"Unknown sizing_target_mode: {self.config.sizing_target_mode}"
            )
        if sizing_target_mesh is not None:
            if self.config.sizing_projection_mode == "volume_weighted":
                h_target_on_current = project_sizing_field(
                    sizing_target_mesh, self.mesh
                )
            elif self.config.sizing_projection_mode == "sampled_vertex":
                h_target_on_current = sample_element_sizing_at_vertices(
                    sizing_target_mesh, self.mesh
                )
            else:
                raise ValueError(
                    "Unknown sizing_projection_mode: "
                    f"{self.config.sizing_projection_mode}"
                )
            s_target = inverse_softplus(h_target_on_current) - inverse_softplus(h_current)
        else:
            s_target = np.zeros_like(h_current)

        mask = graph.current_level_mask
        graph.y_solution = torch.zeros(graph.num_nodes, 1, dtype=torch.float32)
        graph.y_sizing_log_ratio = torch.zeros(graph.num_nodes, 1, dtype=torch.float32)
        graph.current_sizing_field = torch.zeros(graph.num_nodes, 1, dtype=torch.float32)
        if self.config.use_pivot_residual_heads:
            mask_np = graph.current_level_mask.numpy()
            piv = graph.pivot_mask.numpy() & mask_np
            owner = graph.owner_pivot.numpy()
            interp_idx = graph.pivot_interp_index.numpy()
            interp_w = graph.pivot_interp_weight.numpy()
            graph.y_solution_pivot = torch.zeros(graph.num_nodes, 1)
            graph.y_sizing_pivot = torch.zeros(graph.num_nodes, 1)
            graph.y_solution_delta = torch.zeros(graph.num_nodes, 1)
            graph.y_sizing_delta = torch.zeros(graph.num_nodes, 1)
        graph.sizing_target_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        graph.y_solution[mask] = torch.tensor(u_target, dtype=torch.float32).unsqueeze(-1)
        graph.y_sizing_log_ratio[mask] = torch.tensor(s_target, dtype=torch.float32).unsqueeze(-1)
        graph.current_sizing_field[mask] = torch.tensor(h_current, dtype=torch.float32).unsqueeze(-1)
        if sizing_target_mesh is not None:
            graph.sizing_target_mask[mask] = True
        if self.config.use_pivot_residual_heads:
            pivot_idx = np.flatnonzero(piv)
            pivot_values_u = u_target[pivot_idx]
            pivot_values_s = s_target[pivot_idx]
            lookup_u = {int(i): float(v) for i,v in zip(pivot_idx,pivot_values_u)}
            lookup_s = {int(i): float(v) for i,v in zip(pivot_idx,pivot_values_s)}
            base_u = np.sum(np.array([[lookup_u.get(int(o), 0.0) for o in row] for row in interp_idx[mask_np]]) * interp_w[mask_np], axis=1)
            base_s = np.sum(np.array([[lookup_s.get(int(o), 0.0) for o in row] for row in interp_idx[mask_np]]) * interp_w[mask_np], axis=1)
            graph.y_solution_pivot[mask] = torch.tensor(u_target, dtype=torch.float32).unsqueeze(-1)
            graph.y_sizing_pivot[mask] = torch.tensor(s_target, dtype=torch.float32).unsqueeze(-1)
            graph.y_solution_delta[mask] = torch.tensor(u_target-base_u, dtype=torch.float32).unsqueeze(-1)
            graph.y_sizing_delta[mask] = torch.tensor(s_target-base_s, dtype=torch.float32).unsqueeze(-1)
        return graph


class ProgressiveDataset:
    def __init__(self, trajectories: List[Trajectory], config: Config):
        self.trajectories = trajectories
        self.config = config
        if config.include_expert_intermediate_samples:
            persistent_levels = range(len(trajectories[0].meshes)) if trajectories else []
        else:
            persistent_levels = [0]
        self.samples = [
            ProgressiveSample(traj, k, config)
            for traj in trajectories
            for k in persistent_levels
        ]
        self.num_persistent_samples = len(self.samples)
        self.sampled_count = [0] * len(self.samples)

    @property
    def online_size(self) -> int:
        return len(self.samples) - self.num_persistent_samples

    def add_online_samples(self, samples: List[ProgressiveSample], max_size: int) -> int:
        """Append model-rollout samples to a FIFO without touching persistent data."""
        if max_size <= 0:
            return 0
        baseline_count = int(np.median(self.sampled_count)) if self.sampled_count else 0
        for sample in samples:
            self.samples.append(sample)
            # Match AMBER's inherited sampled count behavior so a refresh does
            # not monopolize the next training batches.
            self.sampled_count.append(baseline_count)
        while self.online_size > max_size:
            del self.samples[self.num_persistent_samples]
            del self.sampled_count[self.num_persistent_samples]
        return len(samples)

    def sample_rollout_parent(self, max_depth: int) -> ProgressiveSample:
        """AMBER-style stratified sampling over available refinement depths."""
        valid = [sample for sample in self.samples if sample.k < max_depth]
        if not valid:
            raise ValueError("No replay sample is shallow enough to refine.")
        depths = sorted({sample.k for sample in valid})
        depth = int(np.random.choice(depths))
        candidates = [sample for sample in valid if sample.k == depth]
        return candidates[int(np.random.randint(len(candidates)))]

    @property
    def first(self) -> ProgressiveSample:
        return self.samples[0]

    @property
    def final_meshes(self):
        return [t.meshes[-1] for t in self.trajectories]

    def __len__(self):
        return len(self.samples)


def inverse_softplus(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-12, None)
    return values + np.log(-np.expm1(-values))


class BudgetBatcher:
    def __init__(self, dataset: ProgressiveDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def get_next_batch(self):
        order = np.argsort(self.dataset.sampled_count)
        graphs = []
        total = 0
        for idx in order:
            sample = self.dataset.samples[idx]
            if total + sample.graph_size > self.batch_size:
                continue
            total += sample.graph_size
            self.dataset.sampled_count[idx] += 1
            graphs.append(sample.graph)
        assert graphs, "Batch is empty; increase batch_size."
        return Batch.from_data_list(graphs)


class InfiniteBudgetLoader:
    def __init__(self, batcher: BudgetBatcher, steps_per_epoch: int):
        self.batcher = batcher
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield self.batcher.get_next_batch()

    def __len__(self):
        return self.steps_per_epoch


class SimpleGraphLoader:
    def __init__(self, samples: List[ProgressiveSample]):
        self.samples = samples

    def __iter__(self):
        for sample in self.samples:
            yield Batch.from_data_list([sample.graph])

    def __len__(self):
        return len(self.samples)
