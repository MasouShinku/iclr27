"""Coupled mesh-solution network for progressive AMR."""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

from mpn import MessagePassingStack


def make_mlp(in_dim: int, out_dim: int, hidden: int, layers: int, activation: str):
    acts = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
    }
    if activation not in acts:
        raise ValueError(f"Unknown activation: {activation}")
    modules = [nn.Linear(in_dim, hidden), acts[activation]()]
    for _ in range(max(0, layers - 2)):
        modules += [nn.Linear(hidden, hidden), acts[activation]()]
    modules.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*modules)


class CoupledMeshSolutionNet(nn.Module):
    def __init__(self, config, in_node_features: int, in_edge_features: int):
        super().__init__()
        d = config.latent_dimension
        self.proj_node = nn.Linear(in_node_features, d)
        self.proj_edge = nn.Linear(in_edge_features, d)
        self.shared_mpn = MessagePassingStack(d, config.num_mpn_layers, config.get_stack_config())
        self.edge_dropout = float(config.edge_dropout)
        if not 0.0 <= self.edge_dropout < 1.0:
            raise ValueError("edge_dropout must be in [0, 1).")

        self.physics_head = make_mlp(
            d, config.solution_dimension, config.decoder_hidden_dim,
            config.mlp_num_layers, config.mlp_activation
        )
        physics_last = self.physics_head[-1]
        if isinstance(physics_last, nn.Linear):
            # softplus(-2.25) is close to the mean Poisson solution magnitude.
            nn.init.constant_(physics_last.bias, -2.25)
        self.num_levels = len(config.trajectory_levels)
        self.use_sizing_level_embedding = config.use_sizing_level_embedding
        if self.use_sizing_level_embedding:
            self.sizing_level_embedding = nn.Linear(
                self.num_levels, config.sizing_level_embedding_dim
            )
            sizing_input_dim = d + config.sizing_level_embedding_dim
        else:
            self.sizing_level_embedding = None
            sizing_input_dim = d
        self.use_transition_sizing_heads = config.use_transition_sizing_heads
        self.use_pivot_residual_heads = config.use_pivot_residual_heads
        self.pivot_direct_values = getattr(config, "pivot_direct_values", False)
        num_sizing_heads = self.num_levels - 1 if self.use_transition_sizing_heads else 1
        if self.use_pivot_residual_heads:
            self.physics_pivot_head = make_mlp(d, config.solution_dimension, config.decoder_hidden_dim, config.mlp_num_layers, config.mlp_activation)
            self.physics_fine_head = make_mlp(d, config.solution_dimension, config.decoder_hidden_dim, config.mlp_num_layers, config.mlp_activation)
            self.mesh_pivot_heads = nn.ModuleList([make_mlp(sizing_input_dim, 1, config.decoder_hidden_dim, config.mlp_num_layers, config.mlp_activation) for _ in range(num_sizing_heads)])
            self.mesh_fine_heads = nn.ModuleList([make_mlp(sizing_input_dim, 1, config.decoder_hidden_dim, config.mlp_num_layers, config.mlp_activation) for _ in range(num_sizing_heads)])
        self.mesh_heads = nn.ModuleList([
            make_mlp(
                sizing_input_dim, 1, config.decoder_hidden_dim,
                config.mlp_num_layers, config.mlp_activation
            )
            for _ in range(num_sizing_heads)
        ])
        for head in self.mesh_heads:
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.bias)

    def forward(self, graph: Data, return_components: bool = False):
        g = graph.clone()
        g.x = self.proj_node(g.x)
        g.edge_attr = self.proj_edge(g.edge_attr)
        if self.training and self.edge_dropout > 0.0:
            g.edge_index, edge_mask = dropout_edge(
                g.edge_index, p=self.edge_dropout, training=True
            )
            g.edge_attr = g.edge_attr[edge_mask]
        self.shared_mpn(g)
        h = g.x
        # Predict in physical space; positivity is enforced by the output
        # activation, while normalization is applied only when computing loss.
        components = {}
        if self.use_pivot_residual_heads:
            pivot = graph.pivot_mask
            u_base = torch.nn.functional.softplus(self.physics_pivot_head(h))
            u_delta = self.physics_fine_head(h)
            if self.pivot_direct_values:
                u_delta = torch.nn.functional.softplus(u_delta)
            owner = graph.owner_pivot.clamp_min(0)
            idx = graph.pivot_interp_index.clamp_min(0)
            w = graph.pivot_interp_weight
            interp_base = (u_base[idx] * w.unsqueeze(-1)).sum(dim=1)
            u_pred_phys = torch.where(pivot[:,None], u_base, u_delta if self.pivot_direct_values else interp_base + u_delta)
            components.update(u_base=u_base, u_delta=u_delta)
        else:
            u_pred_phys = torch.nn.functional.softplus(self.physics_head(h))
        sizing_features = [h]
        if self.sizing_level_embedding is not None:
            level_onehot = torch.nn.functional.one_hot(
                graph.level_index, num_classes=self.num_levels
            ).to(dtype=h.dtype)
            sizing_features.append(self.sizing_level_embedding(level_onehot))
        sizing_input = torch.cat(sizing_features, dim=-1)
        if self.use_transition_sizing_heads:
            s_pred = h.new_zeros((h.shape[0], 1))
            for transition, head in enumerate(self.mesh_heads):
                transition_mask = (
                    graph.current_level_mask
                    & (graph.level_index == transition)
                )
                s_pred[transition_mask] = head(sizing_input[transition_mask])
        else:
            s_pred = self.mesh_heads[0](sizing_input)
        if self.use_pivot_residual_heads:
            s_pivot = h.new_zeros((h.shape[0], 1)); s_fine = h.new_zeros((h.shape[0],1))
            for transition, (hp, hf) in enumerate(zip(self.mesh_pivot_heads, self.mesh_fine_heads)):
                transition_mask = graph.current_level_mask & (graph.level_index == transition)
                s_pivot[transition_mask] = hp(sizing_input[transition_mask])
                s_fine[transition_mask] = hf(sizing_input[transition_mask])
            idx = graph.pivot_interp_index.clamp_min(0)
            w = graph.pivot_interp_weight
            interp_s = (s_pivot[idx] * w.unsqueeze(-1)).sum(dim=1)
            s_pred = torch.where(graph.pivot_mask[:, None], s_pivot, s_fine if self.pivot_direct_values else interp_s + s_fine)
            components.update(s_base=s_pivot, s_delta=s_fine)
        if return_components:
            return u_pred_phys, s_pred, components
        return u_pred_phys, s_pred
