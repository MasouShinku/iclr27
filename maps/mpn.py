"""
Message Passing Network — single-file consolidation.

Combines InputEmbedding, MessagePassingBlock, MessagePassingStack, LatentMLP,
EdgeDropout into one MPN module.

Supports inner residual connections and inner layer normalization.
No hierarchical/heterogeneous node handling.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_edge
from torch_scatter import scatter_mean, scatter_max


# ============================================================
# LatentMLP — feedforward network used inside message passing
# ============================================================

class LatentMLP(nn.Module):
    """MLP with configurable layers and activation."""

    def __init__(self, in_features: int, latent_dimension: int,
                 num_layers: int = 2, activation: str = "leakyrelu"):
        super().__init__()
        layers = nn.ModuleList()
        prev = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, latent_dimension))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leakyrelu":
                layers.append(nn.LeakyReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev = latent_dimension
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ============================================================
# Embedding — project raw features to latent dimension
# ============================================================

class Embedding(nn.Module):
    """Linear projection from input features to latent dimension."""

    def __init__(self, in_features: int, latent_dimension: int):
        super().__init__()
        self.linear = nn.Linear(in_features, latent_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class InputEmbedding(nn.Module):
    """Embed both node and edge features to latent dimension (in-place)."""

    def __init__(self, in_node_features: int, in_edge_features: int, latent_dimension: int):
        super().__init__()
        self.node_embedding = Embedding(in_node_features, latent_dimension)
        self.edge_embedding = Embedding(in_edge_features, latent_dimension)

    def forward(self, graph: Data) -> None:
        graph.x = self.node_embedding(graph.x)
        graph.edge_attr = self.edge_embedding(graph.edge_attr)


# ============================================================
# MessagePassingBlock — single layer: edge update + node update
# ============================================================

class MessagePassingBlock(nn.Module):
    """
    Single message passing step.
    Edge update: MLP(src_features, dst_features, edge_features) -> new edge_features
    Node update: MLP(node_features, aggregated_edge_features) -> new node_features

    Supports inner residual connections and inner layer norm.
    NOW SUPPORTS variable input dimensions (for Size MPN first layer 2D input).
    """

    def __init__(self, latent_dimension: int, stack_config: Dict,
                 node_in_dim: Optional[int] = None, edge_in_dim: Optional[int] = None):
        """
        Args:
            node_in_dim: Input node dimension (defaults to latent_dimension)
            edge_in_dim: Input edge dimension (defaults to latent_dimension)
        """
        super().__init__()
        self._latent_dim = latent_dimension
        node_in_dim = node_in_dim or latent_dimension
        edge_in_dim = edge_in_dim or latent_dimension

        mlp_config = stack_config.get("mlp", {})
        num_layers = mlp_config.get("num_layers", 2)
        activation = mlp_config.get("activation_function", "leakyrelu")

        # Aggregation mode: "mean" (default) or "mean+max" (hybrid)
        self.use_hybrid_agg = stack_config.get("aggregation", "mean") == "mean+max"

        # Edge module: input = [src_node, dst_node, edge_attr]
        edge_input_dim = 2 * node_in_dim + edge_in_dim
        self.edge_mlp = LatentMLP(edge_input_dim, latent_dimension, num_layers, activation)

        # Node module: input = [node, agg_mean(, agg_max)]
        # Aggregated edges are ALWAYS latent_dimension (from edge_mlp output)
        node_input_dim = node_in_dim + latent_dimension
        if self.use_hybrid_agg:
            node_input_dim += latent_dimension  # +1 for agg_max
        self.node_mlp = LatentMLP(node_input_dim, latent_dimension, num_layers, activation)

        # Residual and layer norm config
        residual = (stack_config.get("residual_connections") or "").lower()
        layer_norm = (stack_config.get("layer_norm") or "").lower()

        self.use_inner_residual = (residual == "inner") and (node_in_dim == latent_dimension)
        self.use_inner_layer_norm = (layer_norm == "inner")

        if self.use_inner_layer_norm:
            self.node_ln = nn.LayerNorm(latent_dimension)
            self.edge_ln = nn.LayerNorm(latent_dimension)

    def forward(self, graph: Data) -> None:
        """In-place update of graph.x and graph.edge_attr."""
        src, dst = graph.edge_index

        # --- Edge update ---
        old_edge = graph.edge_attr
        edge_input = torch.cat([graph.x[src], graph.x[dst], graph.edge_attr], dim=1)
        graph.edge_attr = self.edge_mlp(edge_input)
        if self.use_inner_residual:
            graph.edge_attr = graph.edge_attr + old_edge
        if self.use_inner_layer_norm:
            graph.edge_attr = self.edge_ln(graph.edge_attr)

        # --- Node update ---
        old_node = graph.x
        num_nodes = graph.x.shape[0]
        if self.use_hybrid_agg:
            agg_mean = scatter_mean(graph.edge_attr, dst, dim=0, dim_size=num_nodes)
            agg_max, _ = scatter_max(graph.edge_attr, dst, dim=0, dim_size=num_nodes)
            # scatter_max fills nodes with no incoming edges with -inf; clamp to 0
            agg_max = agg_max.clamp(min=0.0)
            node_input = torch.cat([graph.x, agg_mean, agg_max], dim=1)
        else:
            agg_edges = scatter_mean(graph.edge_attr, dst, dim=0, dim_size=num_nodes)
            node_input = torch.cat([graph.x, agg_edges], dim=1)
        graph.x = self.node_mlp(node_input)
        if self.use_inner_residual:
            graph.x = graph.x + old_node
        if self.use_inner_layer_norm:
            graph.x = self.node_ln(graph.x)


# ============================================================
# MessagePassingStack — N blocks
# ============================================================

class MessagePassingStack(nn.Module):
    """
    Stack of MessagePassingBlocks.

    NOW SUPPORTS first_layer_input_dim for cases where the first layer
    receives different input dimension (e.g., Size MPN receives 2D input).
    """

    def __init__(self, latent_dimension: int, num_steps: int, stack_config: Dict,
                 first_layer_input_dim: Optional[int] = None):
        """
        Args:
            first_layer_input_dim: If provided, the first block's node input will be this dimension.
                                  All subsequent blocks use latent_dimension.
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_steps):
            if i == 0 and first_layer_input_dim is not None:
                # First layer with custom input dimension
                block = MessagePassingBlock(
                    latent_dimension,
                    stack_config,
                    node_in_dim=first_layer_input_dim,
                    edge_in_dim=latent_dimension  # edges are always D
                )
            else:
                # Standard layer
                block = MessagePassingBlock(latent_dimension, stack_config)
            self.blocks.append(block)

    def forward(self, graph: Data) -> None:
        for block in self.blocks:
            block(graph)

    def forward_jknet(self, graph: Data, interval: int = 3) -> list:
        """
        Run all blocks, collecting node features every `interval` steps.
        Returns list of cloned node-feature tensors.
        e.g. 12 blocks, interval=3  →  features after blocks 3, 6, 9, 12
        """
        collected = []
        for i, block in enumerate(self.blocks):
            block(graph)
            if (i + 1) % interval == 0:
                collected.append(graph.x.clone())
        return collected


# ============================================================
# MPN — full module: Embedding + EdgeDropout + Stack
# ============================================================

class MPN(nn.Module):
    """
    Complete Message Passing Network.

    forward(graph) returns node features [num_nodes, latent_dim].
    Clones the graph internally to avoid in-place modification issues.
    """

    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        latent_dimension: int,
        num_steps: int,
        stack_config: Dict,
        edge_dropout: float = 0.0,
        create_graph_copy: bool = True,
    ):
        super().__init__()
        self.input_embedding = InputEmbedding(in_node_features, in_edge_features, latent_dimension)
        self.stack = MessagePassingStack(latent_dimension, num_steps, stack_config)
        self.edge_dropout = edge_dropout
        self.create_graph_copy = create_graph_copy
        self.latent_dimension = latent_dimension

    def forward(self, graph: Data) -> torch.Tensor:
        """
        Returns:
            Node features tensor of shape [num_nodes, latent_dim]
        """
        import copy
        if self.create_graph_copy:
            graph = copy.deepcopy(graph)

        # Edge dropout (training only)
        if self.edge_dropout > 0.0 and self.training:
            graph.edge_index, edge_mask = dropout_edge(
                edge_index=graph.edge_index, p=self.edge_dropout, training=True
            )
            graph.edge_attr = graph.edge_attr[edge_mask]

        self.input_embedding(graph)
        self.stack(graph)
        return graph.x

    def forward_with_intermediate(self, graph: Data) -> List[torch.Tensor]:
        """
        Like forward(), but returns intermediate node features after each block.

        Returns:
            List of node feature tensors [x_after_block_0, ..., x_after_block_N-1],
            each of shape [num_nodes, latent_dim].
        """
        import copy
        if self.create_graph_copy:
            graph = copy.deepcopy(graph)

        if self.edge_dropout > 0.0 and self.training:
            graph.edge_index, edge_mask = dropout_edge(
                edge_index=graph.edge_index, p=self.edge_dropout, training=True
            )
            graph.edge_attr = graph.edge_attr[edge_mask]

        self.input_embedding(graph)

        intermediate = []
        for block in self.stack.blocks:
            block(graph)
            intermediate.append(graph.x.clone())

        return intermediate
