"""Train progressive coupled Poisson model."""

import logging
import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithm import ProgressiveCoupledModel
from config import Config
from dataset import (
    InfiniteBudgetLoader,
    ProgressiveDataset,
    SimpleGraphLoader,
    generate_trajectories,
    load_trajectory_splits,
    trajectory_split_seeds,
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HierarchicalRolloutCheckpoint(Callback):
    """Save mesh-first checkpoints, then use physics within a mesh band.

    The grid score is dimensionless and combines element-count calibration with
    midpoint DCD.  A candidate with a materially better grid always wins; only
    candidates within ``hierarchical_mesh_tolerance`` of the best grid score
    are allowed to win on the equal-weight multi-level solution RMSE.
    """

    def __init__(self, dirpath, tolerance=0.01, min_improvement=1e-3):
        self.dirpath = Path(dirpath)
        self.tolerance = float(tolerance)
        self.min_improvement = float(min_improvement)
        self.best_grid = None
        self.best_physics = None
        self.best_epoch = None

    @staticmethod
    def _metric(metrics, name):
        value = metrics.get(name)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        ratio = self._metric(metrics, "val/rollout_element_ratio")
        dcd = self._metric(metrics, "val/rollout_dcd")
        logged_grid = self._metric(metrics, "val/rollout_grid_score")
        physics = self._metric(metrics, "val/rollout_solution_rmse")
        if ratio is None or dcd is None or physics is None:
            return

        grid_score = logged_grid if logged_grid is not None else abs(float(np.log(max(ratio, 1e-12)))) + dcd
        should_save = False
        reason = ""
        if self.best_grid is None:
            should_save = True
            reason = "initial"
        elif grid_score < self.best_grid - self.min_improvement:
            should_save = True
            reason = "better_mesh"
        elif grid_score <= self.best_grid + self.tolerance:
            if self.best_physics is None or physics < self.best_physics - 1e-6:
                should_save = True
                reason = "better_physics_within_mesh_band"

        if not should_save:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        path = self.dirpath / "best-hierarchical.ckpt"
        trainer.save_checkpoint(str(path))
        if reason in {"initial", "better_mesh"}:
            self.best_grid = grid_score
        # Keep the best primary grid reference fixed while selecting a better
        # physical predictor inside its tolerance band.
        self.best_physics = physics
        self.best_epoch = int(trainer.current_epoch) + 1
        metadata = {
            "epoch": self.best_epoch,
            "grid_score": grid_score,
            "element_ratio": ratio,
            "dcd_midpoint": dcd,
            "solution_multilevel_rmse": physics,
            "reason": reason,
        }
        (self.dirpath / "best-hierarchical.json").write_text(
            json.dumps(metadata, indent=2)
        )
        print(
            f"[Hierarchical checkpoint] epoch={self.best_epoch} "
            f"grid_score={grid_score:.6f} ratio={ratio:.4f} "
            f"DCD={dcd:.4f} solution_multilevel_RMSE={physics:.6f} reason={reason}",
            flush=True,
        )


def main():
    logging.getLogger("skfem.mesh.mesh").setLevel(logging.ERROR)
    config = Config.from_args()
    if config.resume_checkpoint:
        checkpoint = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=False)
        saved = checkpoint.get("hyper_parameters", {}).get("config")
        saved_version = saved.get("semantics_version", 1) if isinstance(saved, dict) else getattr(saved, "semantics_version", 1)
        if saved_version != config.semantics_version:
            raise ValueError(
                f"Checkpoint semantics_version={saved_version} is incompatible with "
                f"the current version {config.semantics_version}; start a new experiment."
            )
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")

    print("\n" + "=" * 70)
    print(f"Progressive Coupled Training - {config.task_name}")
    print("=" * 70)
    print(f"split: {config.num_train}/{config.num_val}/{config.num_test}")
    print(f"AMR refinement_steps: {config.refinement_steps}")
    print(f"trajectory_levels: {config.trajectory_levels}")
    print(f"derived inference_steps: {config.resolved_inference_steps()}")
    print(f"epochs: {config.max_epochs}, steps/epoch: {config.steps_per_epoch}")
    print(f"latent: {config.latent_dimension}, mpn layers: {config.num_mpn_layers}")
    print(f"edge dropout: {config.edge_dropout}")
    print(f"FEM solution node input: {config.use_fem_solution_input}")
    print(
        "sizing level embedding: "
        f"{config.use_sizing_level_embedding}"
        + (
            f" (dim={config.sizing_level_embedding_dim})"
            if config.use_sizing_level_embedding
            else ""
        )
    )
    print(
        f"transition-specific sizing heads: {config.use_transition_sizing_heads}"
    )
    print(
        f"sizing target: {config.sizing_target_mode}, "
        f"projection: {config.sizing_projection_mode}, "
        f"expert intermediate samples: {config.include_expert_intermediate_samples}, "
        f"damping factor: {config.sizing_damping_factor}"
    )
    print(
        "normalizer frozen after expert initialization: "
        f"{config.freeze_normalizer_after_initialization}"
    )
    if config.rollout_replay_size > 0:
        print(
            "rollout replay: "
            f"size={config.rollout_replay_size}, "
            f"every={config.rollout_replay_every_n_epochs} epochs, "
            f"trajectories={config.rollout_replay_trajectories}, "
            f"start={config.rollout_replay_start_epoch}"
        )
    print(
        "loss weights: "
        f"phys={config.lambda_phys}, sizing={config.lambda_sizing}, "
        f"couple={config.lambda_couple}"
    )
    if config.lambda_couple != 0.0:
        print(f"coupling mode: {config.couple_mode}")
        print(f"coupling warmup: {config.couple_warmup_epochs} epochs")
        if config.couple_max_supervised_ratio >= 0.0:
            print(
                "coupling contribution cap: "
                f"{config.couple_max_supervised_ratio:.1%} of supervised loss"
            )
    if config.lr_schedule == "linear_warmup_decay":
        print("LR scheduler: 10% linear warmup followed by linear decay")
    elif config.lr_schedule == "plateau" and config.rollout_val_samples != 0:
        print(
            f"LR scheduler: factor={config.lr_scheduler_factor}, "
            f"patience={config.lr_scheduler_patience} validations, "
            f"min_lr={config.min_lr}"
        )
    print("Checkpoint selection: best-loss plus mesh-first hierarchical rollout checkpoint")
    if config.resume_checkpoint:
        print(f"Resuming from checkpoint: {config.resume_checkpoint}")
    print("=" * 70 + "\n")

    if config.trajectory_data_dir:
        print(f"Loading trajectory cache: {config.trajectory_data_dir}")
        splits = load_trajectory_splits(config, config.trajectory_data_dir)
        train_traj, val_traj, test_traj = splits["train"], splits["val"], splits["test"]
    else:
        splits = trajectory_split_seeds(config)
        if config.same_sample_splits:
            print(f"Using same trajectory seed for train/val/test: {splits['train'][0]}")
        print("Generating training trajectories...")
        train_traj = generate_trajectories(config, splits["train"], "train")
        print("Generating validation trajectories...")
        val_traj = generate_trajectories(config, splits["val"], "val")
        print("Generating test trajectories...")
        test_traj = generate_trajectories(config, splits["test"], "test")

    train_dataset = ProgressiveDataset(train_traj, config)
    val_dataset = ProgressiveDataset(val_traj, config)
    test_dataset = ProgressiveDataset(test_traj, config)

    algorithm = ProgressiveCoupledModel(config, train_dataset, val_dataset.samples, test_dataset.samples)
    train_loader = InfiniteBudgetLoader(algorithm.batcher, config.steps_per_epoch)
    val_loader = SimpleGraphLoader(val_dataset.samples)
    test_loader = SimpleGraphLoader(test_dataset.samples)

    root = f"{config.output_dir}/{config.experiment_name}/seed{config.seed}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root}/checkpoints",
        filename="best-loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=True,
    )
    latest_checkpoint_callback = ModelCheckpoint(
        dirpath=f"{root}/checkpoints",
        filename="last",
        monitor=None,
        save_top_k=1,
        every_n_epochs=config.validation_every_n_epochs,
        enable_version_counter=False,
        verbose=False,
    )
    callbacks = [checkpoint_callback, latest_checkpoint_callback]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                check_finite=True,
                verbose=True,
            )
        )
    if config.hierarchical_checkpointing and config.rollout_val_samples != 0:
        callbacks.append(
            HierarchicalRolloutCheckpoint(
                f"{root}/checkpoints",
                tolerance=config.hierarchical_mesh_tolerance,
                min_improvement=config.hierarchical_mesh_min_improvement,
            )
        )
    logger = CSVLogger(save_dir=root, name="logs")

    trainer = Trainer(
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.validation_every_n_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=config.gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(
        algorithm,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=config.resume_checkpoint or None,
        # Local training checkpoints contain Config and optimizer state.
        # PyTorch 2.6 otherwise defaults Lightning restores to weights-only.
        weights_only=False,
    )
    trainer.test(algorithm, dataloaders=test_loader)
    print(f"\nTraining complete. Checkpoints: {root}/checkpoints")


_DEFAULTS = [
    "task_name=poisson",
    "num_train=5",
    "num_val=2",
    "num_test=2",
    "max_epochs=10",
    "refinement_steps=50",
    "trajectory_levels=[0,1,2,3]",
    "experiment_name=progressive_poisson",
]

user_keys = {arg.split("=")[0] for arg in sys.argv[1:] if "=" in arg}
for default in reversed(_DEFAULTS):
    key = default.split("=")[0]
    if key not in user_keys:
        sys.argv.insert(1, default)


if __name__ == "__main__":
    main()
