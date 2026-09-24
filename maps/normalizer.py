"""Running normalizers for progressive graphs."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


class TorchRunningMeanStd(nn.Module):
    def __init__(self, epsilon: float = 1e-6, shape: Tuple[int, ...] = ()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(*shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(*shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

    def update(self, arr: Tensor):
        # Replay graphs are intentionally stored on CPU while Lightning moves
        # this module's running statistics to the training device.
        arr = arr.detach().to(device=self.mean.device, dtype=torch.float64)
        batch_mean = arr.mean(dim=0)
        batch_var = torch.nan_to_num(arr.var(dim=0), nan=0.0)
        batch_count = arr.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.mean = mean
        self.var = m2 / total
        self.count = total


class Normalizer(nn.Module):
    def __init__(self, example_graph: Data, normalize_inputs=True, normalize_targets=True,
                 input_clip: float = 1000.0, epsilon: float = 1e-6):
        super().__init__()
        self.input_clip = input_clip
        self.epsilon = epsilon
        self.node_normalizer = TorchRunningMeanStd(epsilon, (example_graph.x.shape[1],)) if normalize_inputs else None
        self.edge_normalizer = TorchRunningMeanStd(epsilon, (example_graph.edge_attr.shape[1],)) if normalize_inputs else None
        self.solution_normalizer = TorchRunningMeanStd(epsilon, (1,)) if normalize_targets else None
        self.sizing_normalizer = TorchRunningMeanStd(epsilon, (1,)) if normalize_targets else None

    def update(self, graph: Data):
        if self.node_normalizer is not None:
            self.node_normalizer.update(graph.x)
        if self.edge_normalizer is not None:
            self.edge_normalizer.update(graph.edge_attr)
        mask = graph.current_level_mask
        if self.solution_normalizer is not None:
            self.solution_normalizer.update(graph.y_solution[mask])
        if self.sizing_normalizer is not None:
            sizing_mask = getattr(graph, "sizing_target_mask", mask)
            if sizing_mask.any():
                self.sizing_normalizer.update(graph.y_sizing_log_ratio[sizing_mask])

    def normalize_inputs(self, graph: Data):
        if self.node_normalizer is not None:
            graph.x = self._normalize(graph.x, self.node_normalizer)
        if self.edge_normalizer is not None:
            graph.edge_attr = self._normalize(graph.edge_attr, self.edge_normalizer)
        return graph

    def normalize_solution(self, y: Tensor) -> Tensor:
        return self._normalize(y, self.solution_normalizer) if self.solution_normalizer is not None else y

    def denormalize_solution(self, y: Tensor) -> Tensor:
        return self._denormalize(y, self.solution_normalizer) if self.solution_normalizer is not None else y

    def normalize_sizing(self, y: Tensor) -> Tensor:
        return self._normalize(y, self.sizing_normalizer) if self.sizing_normalizer is not None else y

    def denormalize_sizing(self, y: Tensor) -> Tensor:
        return self._denormalize(y, self.sizing_normalizer) if self.sizing_normalizer is not None else y

    def _normalize(self, x: Tensor, normalizer: TorchRunningMeanStd) -> Tensor:
        view_shape = [1] * x.ndim
        view_shape[-1] = -1
        out = (x - normalizer.mean.view(view_shape)) / torch.sqrt(normalizer.var.view(view_shape) + self.epsilon)
        return torch.clamp(out, -self.input_clip, self.input_clip).float()

    def _denormalize(self, x: Tensor, normalizer: TorchRunningMeanStd) -> Tensor:
        return (x * torch.sqrt(normalizer.var + self.epsilon) + normalizer.mean).to(x.dtype)


def inverse_softplus(values: Tensor) -> Tensor:
    values = values.clamp_min(1e-12)
    return values + torch.log(-torch.expm1(-values))
