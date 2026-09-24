"""Configuration for progressive coupled mesh-solution training."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    semantics_version: int = 9
    task_name: str = "poisson"
    seed: int = 42
    num_train: int = 5
    num_val: int = 2
    num_test: int = 2
    same_sample_splits: bool = False
    split_seed: int = -1

    refinement_steps: int = 50
    error_threshold: float = 0.85
    trajectory_levels: list = field(default_factory=lambda: [0, 1, 2, 3])

    latent_dimension: int = 64
    num_mpn_layers: int = 8
    decoder_hidden_dim: int = 64
    solution_dimension: int = 1
    mlp_num_layers: int = 3
    mlp_activation: str = "gelu"
    edge_dropout: float = 0.1
    residual_connections: str = "inner"
    layer_norm: str = "inner"
    use_hybrid_aggregation: bool = True
    # When disabled, each prediction uses only the current mesh topology.
    # This is the controlled single-level-graph ablation.
    use_multilevel_graph: bool = True
    use_pivot_shortcuts: bool = False
    use_hop_coarse_graph: bool = False
    use_pivot_residual_heads: bool = False
    pivot_direct_values: bool = False
    # Orthogonal ablation controls.  Residual heads imply pivot metadata;
    # use_pivot_graph keeps that metadata/role encoding for direct heads.
    use_pivot_graph: bool = False
    use_pivot_communication: bool = True
    pivot_coupling_enabled: bool = False
    pivot_ratio: float = 0.10
    pivot_alpha: float = 1.0
    pivot_neighbors: int = 2
    pivot_communication_topology: str = "delaunay"
    use_fem_solution_input: bool = False
    use_sizing_level_embedding: bool = False
    sizing_level_embedding_dim: int = 16
    use_transition_sizing_heads: bool = True
    # "next" learns Mk -> M(k+1). "final" follows AMBER and projects the
    # final expert sizing field onto every current/model-generated mesh.
    sizing_target_mode: str = "next"
    # "density_weighted" is the progressive objective. "amber_mse" exactly
    # matches AMBER's unweighted MSE in inverse-softplus residual space.
    sizing_loss_mode: str = "density_weighted"
    sizing_projection_mode: str = "volume_weighted"
    include_expert_intermediate_samples: bool = True
    # AMBER scales the predicted absolute sizing field by
    # 1 / damping_factor**(remaining_steps - 1). A non-positive value disables it.
    sizing_damping_factor: float = 0.0

    max_epochs: int = 20
    lr: float = 1e-3
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    lr_schedule: str = "plateau"
    weight_decay: float = 0.0
    gradient_clip_val: float = 0.5
    batch_size: int = 250000
    steps_per_epoch: int = 128
    validation_every_n_epochs: int = 10
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-6
    # Periodically collect model-generated intermediate meshes in an in-memory
    # FIFO. Set the buffer size to 0 to keep purely teacher-forced training.
    rollout_replay_size: int = 240
    rollout_replay_every_n_epochs: int = 5
    rollout_replay_trajectories: int = 8
    rollout_replay_start_epoch: int = 5
    # -1 uses the full validation set, 0 disables rollout validation,
    # and a positive value limits the number of trajectories.
    rollout_val_samples: int = -1
    rollout_checkpoint_start_epoch: int = 200
    rollout_checkpoint_max_regression: float = 0.1
    rollout_checkpoint_min_relative_improvement: float = 1e-3
    # Hierarchical rollout checkpointing: mesh quality is primary, solution
    # accuracy is compared only among checkpoints with similar mesh quality.
    hierarchical_checkpointing: bool = True
    hierarchical_mesh_tolerance: float = 0.01
    hierarchical_mesh_min_improvement: float = 1e-3

    lambda_phys: float = 1.0
    lambda_sizing: float = 1.0
    # Keep the detached posterior-error regularizer below the supervised
    # sizing term at the typical loss scales of the Poisson trajectories.
    lambda_couple: float = 0.001
    # "posterior" is Poisson-specific; "gradient_equidistribution" is the
    # legacy symmetric objective; "one_sided_gradient" only penalizes
    # under-resolution in high-gradient elements. The bidirectional mode
    # additionally weights solution supervision by predicted mesh density.
    couple_mode: str = "posterior"
    couple_gradient_alpha: float = 1.0
    couple_gradient_epsilon: float = 1e-6
    couple_reference: str = "mean"
    couple_warmup_epochs: int = 50
    # Cap the weighted coupling contribution relative to the weighted
    # supervised loss. A negative value disables the cap.
    couple_max_supervised_ratio: float = 0.1
    cross_guidance_epsilon: float = 1e-6
    # "none" preserves the original solution MSE. The threshold-density
    # mode adds bounded supervision only for large errors in fine regions.
    # "oracle_target_density" uses the supervised sizing target as a detached
    # oracle for the resolution-weighted physical MSE ablation.
    solution_guidance_mode: str = "none"
    solution_guidance_weight_cap: float = 1.5
    # Oracle physics-only ablation: use the cached expert trajectory meshes
    # during replay/validation/inference instead of the untrained sizing head.
    use_oracle_sizing_rollout: bool = False

    normalize_inputs: bool = True
    normalize_targets: bool = True
    input_clip: float = 1000.0
    # Keep the expert-dataset statistics fixed while model-rollout samples are
    # refreshed. This avoids moving the input and output coordinate systems.
    freeze_normalizer_after_initialization: bool = False

    min_sizing_field_factor: float = 1.25
    max_sizing_field_factor: float = 1.25
    max_mesh_elements: str = "auto"

    output_dir: str = "outputs"
    experiment_name: str = "progressive_poisson"
    resume_checkpoint: str = ""
    trajectory_data_dir: str = ""

    edge_features: list = field(default_factory=lambda: ["euclidean_distance", "edge_curvature"])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_args(cls) -> "Config":
        import sys

        config = cls()
        for arg in sys.argv[1:]:
            if "=" not in arg:
                continue
            key, value = arg.split("=", 1)
            if not hasattr(config, key):
                if key == "inference_steps":
                    raise ValueError(
                        "inference_steps is no longer configurable; inference steps are "
                        "derived from trajectory_levels."
                    )
                raise ValueError(f"Unknown config key: {key}")
            current = getattr(config, key)
            if value.lower() in ("none", "null"):
                value = None
            elif isinstance(current, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(current, list):
                value = json.loads(value)
            else:
                value = type(current)(value)
            setattr(config, key, value)
        return config

    def get_stack_config(self) -> dict:
        return {
            "residual_connections": self.residual_connections,
            "layer_norm": self.layer_norm,
            "aggregation": "mean+max" if self.use_hybrid_aggregation else "mean",
            "mlp": {
                "num_layers": self.mlp_num_layers,
                "activation_function": self.mlp_activation,
            },
        }

    def resolved_inference_steps(self) -> int:
        return max(0, len(self.trajectory_levels) - 1)
