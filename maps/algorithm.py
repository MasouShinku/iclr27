"""Lightning module for progressive coupled Poisson training."""

from typing import List

import numpy as np
import torch
from lightning import LightningModule
from torch_geometric.data import Batch

from config import Config
from dataset import BudgetBatcher, ProgressiveDataset, ProgressiveSample, Trajectory
from losses import compute_losses
from mesh_util import (
    MeshWrapper,
    compute_dcd_midpoint,
    get_sizing_field,
    scale_sizing_field_to_budget,
    update_mesh,
)
from model import CoupledMeshSolutionNet
from normalizer import Normalizer


class ProgressiveCoupledModel(LightningModule):
    def __init__(self, config: Config, train_dataset: ProgressiveDataset,
                 val_samples: List[ProgressiveSample], test_samples: List[ProgressiveSample]):
        super().__init__()
        self.config = config
        self.train_dataset = train_dataset
        self.val_samples = val_samples
        self.test_samples = test_samples

        example = train_dataset.first.graph
        self.model = CoupledMeshSolutionNet(config, example.x.shape[1], example.edge_attr.shape[1])
        self.normalizer = Normalizer(
            example,
            normalize_inputs=config.normalize_inputs,
            normalize_targets=config.normalize_targets,
            input_clip=config.input_clip,
        )
        for sample in train_dataset.samples:
            self.normalizer.update(sample.graph)

        self.batcher = BudgetBatcher(train_dataset, config.batch_size)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.gmsh_kwargs = self._compute_gmsh_kwargs()
        self.max_mesh_elements = self._compute_max_elements()
        self._last_reported_lr = None
        self.save_hyperparameters("config")

    def _effective_lambda_couple(self):
        target = float(self.config.lambda_couple)
        warmup_epochs = int(self.config.couple_warmup_epochs)
        if target == 0.0 or warmup_epochs <= 0:
            return target
        scale = min(1.0, (int(self.current_epoch) + 1) / warmup_epochs)
        return target * scale

    def _compute_gmsh_kwargs(self):
        all_sf = [get_sizing_field(m) for m in self.train_dataset.final_meshes]
        return {
            "min_sizing_field": min(float(sf.min()) for sf in all_sf) / self.config.min_sizing_field_factor,
            "max_sizing_field": max(float(sf.max()) for sf in all_sf) * self.config.max_sizing_field_factor,
        }

    def _compute_max_elements(self):
        max_final = max(m.num_elements for m in self.train_dataset.final_meshes)
        value = self.config.max_mesh_elements
        if value is None or value == "none":
            return None
        if isinstance(value, str):
            if value == "auto":
                return int(max_final * 1.5)
            if value.startswith("x"):
                return int(max_final * float(value[1:]))
        return float(value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        if self.config.lr_schedule == "linear_warmup_decay":
            warmup_epochs = max(1, int(self.config.max_epochs * 0.1))

            def lr_scale(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                decay_epochs = max(1, self.config.max_epochs - warmup_epochs)
                return max(0.0, (self.config.max_epochs - epoch) / decay_epochs)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        if self.config.lr_schedule in {"none", "off"}:
            return optimizer
        if self.config.lr_schedule != "plateau":
            raise ValueError(f"Unknown lr_schedule: {self.config.lr_schedule}")
        if self.config.rollout_val_samples == 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            threshold=1e-3,
            threshold_mode="rel",
            min_lr=self.config.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # Schedule against the equal-weight physical error on all
                # rollout graphs, not only against mesh DCD.
                "monitor": "val/rollout_solution_rmse",
                "interval": "epoch",
                "frequency": self.config.validation_every_n_epochs,
                "strict": True,
            },
        }

    def on_train_epoch_start(self):
        if not self.trainer.optimizers:
            return
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        if self._last_reported_lr is None or not np.isclose(lr, self._last_reported_lr):
            print(f"[Learning rate epoch {self.current_epoch + 1}] lr={lr:.8g}", flush=True)
            self._last_reported_lr = lr

    def _step(self, batch, store):
        batch = batch.to(self.device)
        batch = self.normalizer.normalize_inputs(batch)
        if self.config.use_pivot_residual_heads:
            u_pred_phys, s_pred, components = self.model(batch, return_components=True)
            batch._pivot_components = components
            base = self.normalizer.denormalize_sizing(components["s_base"])
            # Fine sizing head is deliberately in physical offset coordinates;
            # only the pivot head uses the absolute sizing normalizer.
            delta = components["s_delta"]
            idx = batch.pivot_interp_index.clamp_min(0)
            w = batch.pivot_interp_weight
            interp_base = (base[idx] * w.unsqueeze(-1)).sum(dim=1)
            s_phys = torch.where(batch.pivot_mask[:, None], base, interp_base + delta)
            s_pred = self.normalizer.normalize_sizing(s_phys)
        else:
            u_pred_phys, s_pred = self._model_forward(batch)
        lambda_couple = self._effective_lambda_couple()
        loss, metrics = compute_losses(
            u_pred_phys,
            s_pred,
            batch,
            self.normalizer,
            self.config,
            lambda_couple=lambda_couple,
        )
        store.append({k: float(v.detach().cpu()) for k, v in metrics.items()})
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.training_step_outputs)
        last = self.training_step_outputs[-1]
        self.log("L_phys", last["L_phys"], on_step=True, prog_bar=True, logger=False)
        self.log("L_sizing", last["L_sizing"], on_step=True, prog_bar=True, logger=False)
        self.log(
            "L_couple_w",
            last["L_couple_weighted"],
            on_step=True,
            prog_bar=True,
            logger=False,
        )
        return loss

    def on_train_epoch_end(self):
        self._log_epoch("train", self.training_step_outputs)
        if self._should_refresh_rollout_replay():
            self._refresh_rollout_replay()

    def _should_refresh_rollout_replay(self):
        if self.config.rollout_replay_size <= 0:
            return False
        every = int(self.config.rollout_replay_every_n_epochs)
        start = int(self.config.rollout_replay_start_epoch)
        epoch = int(self.current_epoch) + 1
        return every > 0 and epoch >= start and (epoch - start) % every == 0

    @torch.no_grad()
    def _refresh_rollout_replay(self):
        """Collect model-generated Mk states with expert M(k+1) supervision."""
        if (
            self.config.sizing_target_mode == "final"
            and not self.config.include_expert_intermediate_samples
        ):
            self._refresh_amber_rollout_replay()
            return
        requested = min(
            int(self.config.rollout_replay_trajectories),
            len(self.train_dataset.trajectories),
        )
        if requested <= 0 or self.config.resolved_inference_steps() <= 1:
            return

        indices = np.random.choice(
            len(self.train_dataset.trajectories), requested, replace=False
        )
        was_training = self.model.training
        self.model.eval()
        generated_samples = []
        failures = 0
        try:
            # The final generated mesh is not needed as a sizing-head input.
            # Stop at M(K-1), which supplies online inputs for heads 1..K-1.
            for index in indices:
                trajectory = self.train_dataset.trajectories[int(index)]
                accumulated = [trajectory.meshes[0]]
                try:
                    for step in range(self.config.resolved_inference_steps() - 1):
                        next_mesh, _ = self.predict_next_mesh(trajectory, accumulated)
                        accumulated.append(next_mesh)
                        level = step + 1
                        sample = ProgressiveSample(
                            trajectory,
                            level,
                            self.config,
                            mesh=next_mesh,
                            next_mesh=trajectory.meshes[level + 1],
                            is_online=True,
                        )
                        if sample.graph_size <= self.config.batch_size:
                            generated_samples.append(sample)
                except Exception:
                    failures += 1
        finally:
            if was_training:
                self.model.train()
        if not self.config.freeze_normalizer_after_initialization:
            for sample in generated_samples:
                self.normalizer.update(sample.graph)
        self.train_dataset.add_online_samples(
            generated_samples, int(self.config.rollout_replay_size)
        )
        epoch = int(self.current_epoch) + 1
        print(
            f"[Rollout replay epoch {epoch}] trajectories={requested} "
            f"generated={len(generated_samples)} failures={failures} "
            f"buffer={self.train_dataset.online_size}/"
            f"{self.config.rollout_replay_size}",
            flush=True,
        )

    @torch.no_grad()
    def _refresh_amber_rollout_replay(self):
        """Add one-step model rollouts using AMBER's depth-stratified FIFO."""
        requested = int(self.config.rollout_replay_trajectories)
        if requested <= 0:
            return
        was_training = self.model.training
        self.model.eval()
        generated = 0
        failures = 0
        max_depth = self.config.resolved_inference_steps()
        try:
            for _ in range(requested):
                parent = self.train_dataset.sample_rollout_parent(max_depth)
                # predict_next_mesh derives the active head/depth from list
                # length, while only reading the first and last meshes.
                accumulated = [parent.trajectory.meshes[0]]
                accumulated.extend([parent.mesh] * parent.k)
                try:
                    next_mesh, _ = self.predict_next_mesh(
                        parent.trajectory, accumulated
                    )
                    level = parent.k + 1
                    sample = ProgressiveSample(
                        parent.trajectory,
                        level,
                        self.config,
                        mesh=next_mesh,
                        # A non-null target also supervises generated M3 with
                        # its remaining residual to the final expert mesh.
                        next_mesh=parent.trajectory.meshes[-1],
                        is_online=True,
                    )
                    if sample.graph_size > self.config.batch_size:
                        continue
                    if not self.config.freeze_normalizer_after_initialization:
                        self.normalizer.update(sample.graph)
                    self.train_dataset.add_online_samples(
                        [sample], int(self.config.rollout_replay_size)
                    )
                    generated += 1
                except Exception:
                    failures += 1
        finally:
            if was_training:
                self.model.train()
        epoch = int(self.current_epoch) + 1
        print(
            f"[AMBER replay epoch {epoch}] requested={requested} "
            f"generated={generated} failures={failures} "
            f"buffer={self.train_dataset.online_size}/"
            f"{self.config.rollout_replay_size}",
            flush=True,
        )

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.validation_step_outputs)

    def on_validation_epoch_end(self):
        self._log_epoch("val", self.validation_step_outputs)
        if not self.trainer.sanity_checking and self.config.rollout_val_samples != 0:
            self._log_rollout_validation()

    def test_step(self, batch, batch_idx):
        self._step(batch, self.test_step_outputs)

    def on_test_epoch_end(self):
        self._log_epoch("test", self.test_step_outputs)

    def _log_epoch(self, prefix, outputs):
        if not outputs:
            return
        avg = {k: np.mean([x[k] for x in outputs]) for k in outputs[0].keys()}
        self.log_dict({f"{prefix}/{k}": v for k, v in avg.items()}, logger=True)
        if prefix == "val" and not self.trainer.sanity_checking:
            epoch = self.current_epoch + 1
            print(
                f"[Validation epoch {epoch}] "
                f"loss={avg['loss']:.6f} "
                f"L_phys={avg['L_phys']:.6f} "
                f"L_sizing={avg['L_sizing']:.6f} "
                f"L_couple={avg['L_couple']:.6f} "
                f"L_couple_weighted={avg['L_couple_weighted']:.6f} "
                f"lambda_couple={avg['lambda_couple']:.6g} "
                f"target={avg['lambda_couple_target']:.6g}",
                flush=True,
            )
        outputs.clear()

    @torch.no_grad()
    def predict_next_mesh(self, trajectory: Trajectory, accumulated: List[MeshWrapper]):
        from graph_util import build_multilevel_graph

        k = len(accumulated) - 1
        graph = build_multilevel_graph(
            accumulated[0],
            accumulated[-1],
            trajectory.load_fn,
            current_level=k,
            max_levels=len(trajectory.meshes),
            edge_feature_names=self.config.edge_features,
            use_fem_solution_input=self.config.use_fem_solution_input,
            use_multilevel_graph=self.config.use_multilevel_graph,
            task_name=trajectory.task_name,
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
        graph = Batch.from_data_list([graph]).to(self.device)
        graph = self.normalizer.normalize_inputs(graph)
        if self.config.use_pivot_residual_heads:
            u_pred, _, components = self.model(graph, return_components=True)
            base = self.normalizer.denormalize_sizing(components["s_base"])
            delta = components["s_delta"]
            idx = graph.pivot_interp_index.clamp_min(0)
            interp_base = (base[idx] * graph.pivot_interp_weight.unsqueeze(-1)).sum(dim=1)
            s_phys_all = torch.where(graph.pivot_mask[:, None], base, interp_base + delta)
            s_pred = self.normalizer.normalize_sizing(s_phys_all)
        else:
            u_pred, s_pred = self._model_forward(graph)
        s_phys = self.normalizer.denormalize_sizing(s_pred).detach().cpu().numpy().flatten()
        mask = graph.current_level_mask.detach().cpu().numpy().astype(bool)
        s_current = s_phys[mask]
        current_mesh = accumulated[-1]
        if self.config.use_oracle_sizing_rollout:
            next_level = min(k + 1, len(trajectory.meshes) - 1)
            next_mesh = trajectory.meshes[next_level]
            expert_sizing = get_sizing_field(next_mesh)
            return next_mesh, {
                "u_pred": u_pred[graph.current_level_mask].detach().cpu().numpy().flatten(),
                "s_pred": s_current,
                "predicted_sizing_field": expert_sizing,
                "predicted_sizing_field_raw": expert_sizing,
                "predicted_sizing_field_damped": expert_sizing,
            }
        h_current = get_sizing_field(current_mesh)
        h_next_raw = self._apply_sizing_residual(h_current, s_current)
        h_next_damped = self._dampen_sizing_field(h_next_raw, k)
        h_next = np.clip(h_next_damped, self.gmsh_kwargs["min_sizing_field"], self.gmsh_kwargs["max_sizing_field"])
        if self.max_mesh_elements is not None:
            approx = self._safe_estimate(current_mesh, h_next)
            if approx > self.max_mesh_elements:
                h_next = scale_sizing_field_to_budget(h_next, current_mesh, self.max_mesh_elements, "vertex")
        new_mesh = update_mesh(current_mesh, h_next, trajectory.geom_fn, self.gmsh_kwargs)
        return new_mesh, {
            "u_pred": u_pred[graph.current_level_mask].detach().cpu().numpy().flatten(),
            "s_pred": s_current,
            "predicted_sizing_field": h_next,
            "predicted_sizing_field_raw": h_next_raw,
            "predicted_sizing_field_damped": h_next_damped,
        }

    @staticmethod
    def _apply_sizing_residual(h_current: np.ndarray, residual: np.ndarray) -> np.ndarray:
        h_current = np.clip(h_current, 1e-12, None)
        inverse_current = h_current + np.log(-np.expm1(-h_current))
        return np.logaddexp(0.0, inverse_current + residual)

    def _dampen_sizing_field(self, sizing_field: np.ndarray,
                             refinement_depth: int) -> np.ndarray:
        factor = float(self.config.sizing_damping_factor)
        if factor <= 0.0:
            return sizing_field
        if factor > 1.0:
            raise ValueError("sizing_damping_factor must be in (0, 1].")
        exponent = self.config.resolved_inference_steps() - refinement_depth - 1
        return sizing_field / (factor ** max(0, exponent))

    @torch.no_grad()
    def _log_rollout_validation(self):
        trajectories = []
        seen = set()
        for sample in self.val_samples:
            if sample.trajectory.seed not in seen:
                trajectories.append(sample.trajectory)
                seen.add(sample.trajectory.seed)
            if (
                self.config.rollout_val_samples > 0
                and len(trajectories) >= self.config.rollout_val_samples
            ):
                break
        if not trajectories:
            return
        was_training = self.model.training
        self.model.eval()
        ratios, dcds, element_count_errors = [], [], []
        solution_rmses, solution_m0_rmses = [], []
        solution_rmses_by_level = [
            [] for _ in range(self.config.resolved_inference_steps() + 1)
        ]
        step_ratios = [[] for _ in range(self.config.resolved_inference_steps())]
        for trajectory in trajectories:
            accumulated = [trajectory.meshes[0]]
            trajectory_level_rmses = []
            for step in range(self.config.resolved_inference_steps()):
                try:
                    next_mesh, preds = self.predict_next_mesh(trajectory, accumulated)
                except Exception:
                    break
                # Evaluate on the graph where this solution was predicted.
                # Every level is included with equal weight in validation.
                from mesh_util import project_vertex_field
                current_mesh = accumulated[-1]
                current_target = project_vertex_field(
                    trajectory.meshes[-1], trajectory.u_final, current_mesh
                )
                current_error = np.asarray(preds["u_pred"]) - current_target
                level = len(accumulated) - 1
                level_rmse = float(np.sqrt(np.mean(current_error ** 2)))
                solution_rmses_by_level[level].append(level_rmse)
                trajectory_level_rmses.append(level_rmse)
                accumulated.append(next_mesh)
                expert_next = trajectory.meshes[step + 1].num_elements
                step_ratios[step].append(next_mesh.num_elements / expert_next)
            predicted = accumulated[-1].num_elements
            expert = trajectory.meshes[-1].num_elements
            ratios.append(predicted / expert)
            element_count_errors.append(abs(predicted - expert) / expert)
            dcds.append(compute_dcd_midpoint(accumulated[-1], trajectory.meshes[-1]))
            u_pred = self.predict_current_solution(trajectory, accumulated)
            from mesh_util import project_vertex_field
            final_target = project_vertex_field(
                trajectory.meshes[-1], trajectory.u_final, accumulated[-1]
            )
            final_error = np.asarray(u_pred) - final_target
            final_level = len(accumulated) - 1
            final_level_rmse = float(np.sqrt(np.mean(final_error ** 2)))
            solution_rmses_by_level[final_level].append(final_level_rmse)
            trajectory_level_rmses.append(final_level_rmse)
            solution_rmses.append(float(np.mean(trajectory_level_rmses)))
            u_pred_m0 = project_vertex_field(accumulated[-1], u_pred, accumulated[0])
            u_target_m0 = project_vertex_field(
                trajectory.meshes[-1], trajectory.u_final, accumulated[0]
            )
            solution_m0_rmses.append(
                float(np.sqrt(np.mean((u_pred_m0 - u_target_m0) ** 2)))
            )
        if was_training:
            self.model.train()
        ratio = float(np.mean(ratios))
        dcd = float(np.mean(dcds))
        element_count_error = float(np.mean(element_count_errors))
        # This is the primary physical validation metric used by the
        # scheduler and hierarchical checkpoint. It explicitly penalizes
        # degradation at M1/M2/M3 instead of hiding it in an M0 projection.
        solution_rmse = float(np.mean(solution_rmses))
        solution_m0_rmse = float(np.mean(solution_m0_rmses))
        # Dimensionless primary mesh score: calibrate total density and spatial
        # placement before comparing the physical solution error.
        grid_score = float(abs(np.log(max(ratio, 1e-12))) + dcd)
        self.log("val/rollout_element_ratio", ratio, logger=True)
        self.log("val/rollout_dcd", dcd, logger=True)
        self.log("val/rollout_grid_score", grid_score, logger=True)
        self.log(
            "val/rollout_element_count_error",
            element_count_error,
            logger=True,
        )
        self.log("val/rollout_solution_rmse", solution_rmse, logger=True)
        self.log("val/rollout_solution_m0_rmse", solution_m0_rmse, logger=True)
        level_means = []
        for level, values in enumerate(solution_rmses_by_level):
            if values:
                level_mean = float(np.mean(values))
                level_means.append(level_mean)
                self.log(
                    f"val/rollout_solution_rmse_level_{level}",
                    level_mean,
                    logger=True,
                )
        step_text = ", ".join(
            f"s{step}={np.mean(values):.3f}" for step, values in enumerate(step_ratios) if values
        )
        print(
            f"[Rollout validation epoch {self.current_epoch + 1}] "
            f"samples={len(ratios)} element_ratio={ratio:.4f} DCD={dcd:.4f} "
            f"grid_score={grid_score:.4f} "
            f"count_error={element_count_error:.4f} "
            f"solution_RMSE={solution_rmse:.6f} "
            f"solution_M0_projected_RMSE={solution_m0_rmse:.6f} "
            f"levels=[{', '.join(f's{idx}={value:.6f}' for idx, value in enumerate(level_means))}] "
            f"steps=[{step_text}]",
            flush=True,
        )

    @torch.no_grad()
    def predict_current_solution(self, trajectory: Trajectory, accumulated: List[MeshWrapper]):
        """Predict the solution on the last mesh in an accumulated rollout."""
        from graph_util import build_multilevel_graph

        k = len(accumulated) - 1
        graph = build_multilevel_graph(
            accumulated[0],
            accumulated[-1],
            trajectory.load_fn,
            current_level=k,
            max_levels=len(trajectory.meshes),
            edge_feature_names=self.config.edge_features,
            use_fem_solution_input=self.config.use_fem_solution_input,
            use_multilevel_graph=self.config.use_multilevel_graph,
            task_name=trajectory.task_name,
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
        graph = Batch.from_data_list([graph]).to(self.device)
        graph = self.normalizer.normalize_inputs(graph)
        u_pred, _ = self._model_forward(graph)
        return u_pred[graph.current_level_mask].detach().cpu().numpy().flatten()

    def _model_forward(self, graph):
        return self.model(graph)

    @staticmethod
    def _safe_estimate(mesh: MeshWrapper, sizing_field: np.ndarray):
        from mesh_util import estimate_num_elements
        try:
            return estimate_num_elements(mesh, sizing_field, "vertex")
        except Exception:
            return 0
