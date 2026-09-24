"""Losses for progressive coupling."""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

def compute_posterior_estimator(u_pred, graph):
    """Current-element Poisson residual estimator from a detached solution."""
    elements = graph.current_element_index
    positions = graph.pos[elements]
    values = u_pred.detach()[elements, 0]

    edge_01 = positions[:, 1] - positions[:, 0]
    edge_02 = positions[:, 2] - positions[:, 0]
    element_matrix = torch.stack((edge_01, edge_02), dim=-1)
    value_differences = torch.stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]), dim=-1)
    gradients = torch.linalg.solve(
        element_matrix.transpose(-1, -2), value_differences.unsqueeze(-1)
    ).squeeze(-1)

    areas = graph.current_element_area[:, 0]
    # skfem CellBasis uses h = sqrt(abs(det J)) = sqrt(2 * triangle_area).
    element_h = torch.sqrt(2.0 * areas)
    estimator = areas * element_h.square() * graph.current_element_load[:, 0]

    facet_elements = graph.current_interior_facet_elements
    facet_vertices = graph.current_interior_facet_vertices
    facet_positions = graph.pos[facet_vertices]
    tangents = facet_positions[:, 1] - facet_positions[:, 0]
    edge_lengths = torch.linalg.vector_norm(tangents, dim=-1).clamp_min(1e-12)
    normals = torch.stack((-tangents[:, 1], tangents[:, 0]), dim=-1) / edge_lengths.unsqueeze(-1)
    gradient_jump = ((gradients[facet_elements[:, 0]] - gradients[facet_elements[:, 1]]) * normals).sum(-1)
    facet_contribution = 0.5 * edge_lengths.square() * gradient_jump.square()
    estimator = estimator + scatter_add(
        facet_contribution.repeat_interleave(2),
        facet_elements.reshape(-1),
        dim=0,
        dim_size=elements.shape[0],
    )
    return estimator.detach()


def compute_gradient_equidistribution_loss(
    u_pred_phys, s_phys, sizing_mask, graph, alpha=1.0, epsilon=1e-6
):
    """Equalize element monitor mass V_K * (|grad u| + eps)^alpha.

    The gradient is computed from the P1 solution predicted on the current
    mesh.  Statistics are normalized per graph so unrelated PDE samples in a
    batch do not compete for one monitor-mass constant.
    """
    elements = graph.current_element_index
    element_mask = sizing_mask[elements]
    valid = element_mask.reshape(element_mask.shape[0], -1).all(dim=1)
    if not valid.any():
        return s_phys.sum() * 0.0

    positions = graph.pos[elements]
    values = u_pred_phys[elements, 0]
    edge_01 = positions[:, 1] - positions[:, 0]
    edge_02 = positions[:, 2] - positions[:, 0]
    matrix = torch.stack((edge_01, edge_02), dim=-1)
    differences = torch.stack(
        (values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]), dim=-1
    )
    gradients = torch.linalg.solve(
        matrix.transpose(-1, -2), differences.unsqueeze(-1)
    ).squeeze(-1)
    gradient_norm = torch.linalg.vector_norm(gradients, dim=-1)

    h_current = graph.current_sizing_field.clamp_min(1e-12)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_next = F.softplus(inverse_current + s_phys)
    h_element = h_next[elements].mean(dim=(1, 2))
    # In 2D, the predicted target volume is proportional to h_next^2. The
    # omitted dimensional constant is irrelevant to the coefficient of
    # variation objective.
    monitor_mass = h_element.square() * (
        gradient_norm + float(epsilon)
    ).pow(float(alpha))
    monitor_mass = monitor_mass[valid]

    if hasattr(graph, "batch") and graph.batch is not None:
        graph_index = graph.batch[elements[valid, 0]]
        num_graphs = int(graph.batch.max().item()) + 1
    else:
        graph_index = torch.zeros(
            monitor_mass.shape[0], dtype=torch.long, device=monitor_mass.device
        )
        num_graphs = 1
    sums = scatter_add(monitor_mass, graph_index, dim=0, dim_size=num_graphs)
    counts = scatter_add(
        torch.ones_like(monitor_mass), graph_index, dim=0, dim_size=num_graphs
    )
    means = sums / counts.clamp_min(1.0)
    centered = monitor_mass - means[graph_index]
    variances = scatter_add(centered.square(), graph_index, dim=0, dim_size=num_graphs)
    variances = variances / counts.clamp_min(1.0)
    supervised = counts > 0
    return (variances[supervised] / (means[supervised].square() + float(epsilon))).mean()


def compute_one_sided_gradient_resolution_loss(
    u_pred_phys, s_phys, sizing_mask, graph, epsilon=1e-6, reference="mean"
):
    """Penalize under-resolution only in above-average-gradient elements.

    The solution gradient and the graph-level threshold are detached, so this
    regularizer updates only the sizing branch. Geometry-driven refinement in
    low-gradient regions is deliberately left unconstrained.
    """
    elements = graph.current_element_index
    valid = sizing_mask[elements].all(dim=1)
    if not valid.any():
        return s_phys.sum() * 0.0

    positions = graph.pos[elements]
    values = u_pred_phys[elements, 0]
    edge_vectors = positions[:, 1:] - positions[:, :1]
    value_differences = values[:, 1:] - values[:, :1]
    determinant = torch.linalg.det(edge_vectors).abs()
    geometry_scale = edge_vectors.abs().amax(dim=(-2, -1)).clamp_min(
        torch.finfo(edge_vectors.dtype).tiny
    )
    dimension = edge_vectors.shape[-1]
    nondegenerate = determinant > (
        16.0 * torch.finfo(edge_vectors.dtype).eps * geometry_scale.pow(dimension)
    )
    valid = valid & nondegenerate
    if not valid.any():
        return s_phys.sum() * 0.0
    gradients = torch.linalg.solve(
        edge_vectors[valid], value_differences[valid].unsqueeze(-1)
    ).squeeze(-1)
    gradient_norm = torch.linalg.vector_norm(gradients, dim=-1).detach()

    h_current = graph.current_sizing_field.clamp_min(1e-12)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_next = F.softplus(inverse_current + s_phys)
    # A d-simplex has d+1 vertices. The geometric mean avoids allowing one
    # large vertex sizing value to dominate an otherwise refined element.
    h_element = torch.exp(
        torch.log(h_next[elements].clamp_min(1e-12)).mean(dim=(1, 2))
    )

    area = graph.current_element_area[:, 0][valid]
    h_element = h_element[valid]
    if hasattr(graph, "batch") and graph.batch is not None:
        graph_index = graph.batch[elements[valid, 0]]
        num_graphs = int(graph.batch.max().item()) + 1
    else:
        graph_index = torch.zeros(
            h_element.shape[0], dtype=torch.long, device=h_element.device
        )
        num_graphs = 1

    area_sum = scatter_add(area, graph_index, dim=0, dim_size=num_graphs)
    g_mean = scatter_add(
        area * gradient_norm, graph_index, dim=0, dim_size=num_graphs
    ) / area_sum.clamp_min(float(epsilon))
    h_mean = scatter_add(
        area * h_element, graph_index, dim=0, dim_size=num_graphs
    ) / area_sum.clamp_min(float(epsilon))

    activation = F.relu(
        gradient_norm / (g_mean[graph_index] + float(epsilon)) - 1.0
    ).detach()
    threshold = (h_mean * g_mean).detach()[graph_index]
    if reference == "median":
        threshold = torch.empty_like(h_element)
        for index in range(num_graphs):
            selected = graph_index == index
            if not selected.any():
                continue
            def weighted_median(values):
                ordered = torch.argsort(values)
                cumulative = area[selected][ordered].cumsum(0)
                middle = torch.searchsorted(cumulative, cumulative[-1] * 0.5)
                return values[ordered[middle.clamp_max(len(ordered) - 1)]]
            threshold[selected] = weighted_median(h_element[selected].detach()) * weighted_median(gradient_norm[selected])
    elif reference == "local":
        # Map global element ids (PyG offsets these facets) into valid elements.
        remap = torch.full((len(elements),), -1, device=elements.device, dtype=torch.long)
        remap[valid] = torch.arange(int(valid.sum()), device=elements.device)
        pairs = remap[graph.current_interior_facet_elements]
        pairs = pairs[(pairs >= 0).all(dim=1)]
        own = torch.arange(len(area), device=area.device)
        source = torch.cat((own, pairs[:, 0], pairs[:, 1]))
        target = torch.cat((own, pairs[:, 1], pairs[:, 0]))
        local_area = scatter_add(area[source], target, dim=0, dim_size=len(area))
        local_h = scatter_add((area * h_element.detach())[source], target, dim=0, dim_size=len(area)) / local_area.clamp_min(float(epsilon))
        local_g = scatter_add((area * gradient_norm)[source], target, dim=0, dim_size=len(area)) / local_area.clamp_min(float(epsilon))
        threshold = (local_h * local_g).detach()
    elif reference != "mean":
        raise ValueError(f"Unknown coupling reference: {reference}")
    violation = F.relu(
        h_element * (gradient_norm + float(epsilon))
        / (threshold + float(epsilon))
        - 1.0
    )
    weights = area * activation
    numerator = scatter_add(
        weights * violation.square(), graph_index, dim=0, dim_size=num_graphs
    )
    denominator = scatter_add(
        weights, graph_index, dim=0, dim_size=num_graphs
    )
    valid_graphs = area_sum > 0
    return (
        numerator[valid_graphs]
        / (denominator[valid_graphs] + float(epsilon))
    ).mean()


def _current_vertex_control_volume(graph):
    """Return the P1 lumped control volume contributed by current elements."""
    elements = graph.current_element_index
    num_vertices = graph.pos.shape[0]
    simplex_size = elements.shape[1]
    area = graph.current_element_area[:, 0]
    contribution = area.repeat_interleave(simplex_size) / float(simplex_size)
    volume = torch.zeros(num_vertices, device=area.device, dtype=area.dtype)
    volume.index_add_(0, elements.reshape(-1), contribution)
    return volume


def compute_resolution_weighted_solution_loss(
    u_pred_norm, u_target_norm, s_phys, sizing_mask, graph, epsilon=1e-6
):
    """Solution MSE weighted by detached predicted local mesh density.

    The density is computed from the predicted next sizing field.  Both the
    density and its per-graph mean are detached, so this term updates only the
    solution readout (while the shared trunk still receives that gradient).
    """
    mask = graph.current_level_mask
    if not mask.any():
        return u_pred_norm.sum() * 0.0
    h_current = graph.current_sizing_field.clamp_min(epsilon)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_next = F.softplus(inverse_current + s_phys).detach().clamp_min(epsilon)
    dimension = int(graph.current_element_index.shape[1] - 1)
    rho = h_next.pow(-dimension).reshape(-1)
    volume = _current_vertex_control_volume(graph)
    batch = getattr(graph, "batch", None)
    if batch is None:
        graph_index = torch.zeros_like(volume, dtype=torch.long)
        num_graphs = 1
    else:
        graph_index = batch
        num_graphs = int(batch.max().item()) + 1
    rho_integral = scatter_add(volume * rho, graph_index, dim=0, dim_size=num_graphs)
    volume_integral = scatter_add(volume, graph_index, dim=0, dim_size=num_graphs)
    rho_mean = rho_integral / volume_integral.clamp_min(epsilon)
    weights = volume * (1.0 + rho / rho_mean[graph_index].clamp_min(epsilon))
    active = mask.reshape(-1) & (volume > 0)
    squared_error = (u_pred_norm - u_target_norm).square().squeeze(-1)
    numerator = scatter_add(
        (weights * squared_error * active), graph_index, dim=0, dim_size=num_graphs
    )
    denominator = scatter_add(
        weights * active, graph_index, dim=0, dim_size=num_graphs
    )
    valid = denominator > 0
    return (numerator[valid] / denominator[valid].clamp_min(epsilon)).mean()


def compute_oracle_resolution_weighted_solution_loss(
    u_pred_norm, u_target_norm, s_target, sizing_mask, graph,
    epsilon=1e-6, weight_cap=2.0
):
    """Resolution-weighted solution MSE using the supervised sizing target.

    This is an oracle upper-bound ablation: the sizing target is used only as
    a detached guidance field.  The normalization is the same as the
    predicted-density weighted objective, but no sizing prediction can affect
    this loss.
    """
    mask = graph.current_level_mask.reshape(-1)
    if not mask.any():
        return u_pred_norm.sum() * 0.0
    h_current = graph.current_sizing_field.clamp_min(epsilon)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_target = F.softplus(inverse_current + s_target).detach().reshape(-1).clamp_min(epsilon)
    dimension = int(graph.current_element_index.shape[1] - 1)
    rho = h_target.pow(-dimension)
    volume = _current_vertex_control_volume(graph)
    active = mask & (volume > 0)
    batch = getattr(graph, "batch", None)
    if batch is None:
        graph_index = torch.zeros_like(volume, dtype=torch.long)
        num_graphs = 1
    else:
        graph_index = batch
        num_graphs = int(batch.max().item()) + 1
    active_weight = volume * active
    volume_sum = scatter_add(active_weight, graph_index, dim=0, dim_size=num_graphs)
    p = active_weight / volume_sum[graph_index].clamp_min(epsilon)
    rho_bar = scatter_add(p * rho, graph_index, dim=0, dim_size=num_graphs).detach()
    weights = (1.0 + rho / rho_bar[graph_index].clamp_min(epsilon))
    if weight_cap > 0.0:
        weights = weights.clamp_max(float(weight_cap))
    weights = weights.detach()
    squared_error = (u_pred_norm - u_target_norm).square().reshape(-1)
    numerator = scatter_add(
        p * weights * squared_error, graph_index, dim=0, dim_size=num_graphs
    )
    valid = volume_sum > 0
    return numerator[valid].mean()


def compute_threshold_density_solution_loss(
    u_pred_norm, u_target_norm, s_phys, sizing_mask, graph, epsilon=1e-6
):
    """Original full-field MSE plus bounded excess-error guidance.

    The base term is intentionally the same node MSE used by the original
    implementation.  The additional term uses per-graph lumped control-volume
    probabilities, detached target sizing density, and a weighted
    mean-plus-standard-deviation error threshold.  Consequently this path
    updates only the solution readout; sizing statistics are guidance only.
    """
    mask = graph.current_level_mask.reshape(-1)
    if not mask.any():
        return u_pred_norm.sum() * 0.0
    squared_error = (u_pred_norm - u_target_norm).square().reshape(-1)
    base = squared_error[mask].mean()
    volume = _current_vertex_control_volume(graph)
    active = mask & (volume > 0)
    dimension = int(graph.current_element_index.shape[1] - 1)
    h_current = graph.current_sizing_field.clamp_min(epsilon)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_target = F.softplus(inverse_current + s_phys).detach().reshape(-1).clamp_min(epsilon)
    rho = h_target.pow(-dimension)

    batch = getattr(graph, "batch", None)
    if batch is None:
        graph_index = torch.zeros_like(volume, dtype=torch.long)
        num_graphs = 1
    else:
        graph_index = batch
        num_graphs = int(batch.max().item()) + 1
    active_weight = volume * active
    volume_sum = scatter_add(active_weight, graph_index, dim=0, dim_size=num_graphs)
    p = active_weight / volume_sum[graph_index].clamp_min(epsilon)
    rho_bar = scatter_add(p * rho, graph_index, dim=0, dim_size=num_graphs).detach()
    density_boost = (1.0 - rho_bar[graph_index] / rho).clamp_min(0.0).clamp_max(1.0).detach()
    error_mean = scatter_add(p * squared_error, graph_index, dim=0, dim_size=num_graphs).detach()
    error_var = scatter_add(
        p * (squared_error - error_mean[graph_index]).square(),
        graph_index, dim=0, dim_size=num_graphs,
    ).clamp_min(0.0).detach()
    threshold = (error_mean + torch.sqrt(error_var)).detach()
    excess = F.relu(squared_error - threshold[graph_index])
    extra = scatter_add(
        p * density_boost * excess, graph_index, dim=0, dim_size=num_graphs
    )
    valid = volume_sum > 0
    return base + extra[valid].mean()


def compute_one_sided_gradient_constraint_loss(
    u_pred_phys, s_phys, sizing_mask, graph, epsilon=1e-6
):
    """One-sided h-g constraint from predicted solution to sizing only."""
    elements = graph.current_element_index
    element_mask = sizing_mask[elements]
    valid = element_mask.reshape(element_mask.shape[0], -1).all(dim=1)
    if not valid.any():
        return s_phys.sum() * 0.0
    positions = graph.pos[elements]
    values = u_pred_phys[elements, 0]
    edge_vectors = positions[:, 1:] - positions[:, :1]
    value_differences = values[:, 1:] - values[:, :1]
    determinant = torch.linalg.det(edge_vectors).abs()
    scale = edge_vectors.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(edge_vectors.dtype).tiny)
    dimension = edge_vectors.shape[-1]
    valid = valid & (determinant > 16.0 * torch.finfo(edge_vectors.dtype).eps * scale.pow(dimension))
    if not valid.any():
        return s_phys.sum() * 0.0
    gradients = torch.linalg.solve(
        edge_vectors[valid], value_differences[valid].unsqueeze(-1)
    ).squeeze(-1)
    gradient_norm = torch.linalg.vector_norm(gradients, dim=-1).detach()
    h_current = graph.current_sizing_field.clamp_min(epsilon)
    inverse_current = h_current + torch.log(-torch.expm1(-h_current))
    h_next = F.softplus(inverse_current + s_phys)
    h_element = torch.exp(
        torch.log(h_next[elements].clamp_min(epsilon)).mean(dim=(1, 2))
    )
    h_element = h_element[valid]
    area = graph.current_element_area[:, 0][valid]
    batch = getattr(graph, "batch", None)
    if batch is None:
        graph_index = torch.zeros_like(area, dtype=torch.long)
        num_graphs = 1
    else:
        graph_index = batch[elements[valid, 0]]
        num_graphs = int(batch.max().item()) + 1
    area_sum = scatter_add(area, graph_index, dim=0, dim_size=num_graphs)
    g_bar = (scatter_add(area * gradient_norm, graph_index, dim=0, dim_size=num_graphs)
             / area_sum.clamp_min(epsilon)).detach()
    h_bar = (scatter_add(area * h_element, graph_index, dim=0, dim_size=num_graphs)
             / area_sum.clamp_min(epsilon)).detach()
    high = (gradient_norm > g_bar[graph_index]).detach()
    if not high.any():
        return s_phys.sum() * 0.0
    h_max = (h_bar[graph_index] * g_bar[graph_index]
             / (gradient_norm + epsilon)).detach()
    violation = F.relu(h_element / (h_max + epsilon) - 1.0).square()
    numerator = scatter_add(area * high * violation, graph_index, dim=0, dim_size=num_graphs)
    denominator = scatter_add(area * high.to(area.dtype), graph_index, dim=0, dim_size=num_graphs)
    active = denominator > 0
    return (numerator[active] / denominator[active].clamp_min(epsilon)).mean()


def compute_density_weighted_sizing_loss(s_pred_phys, s_target, sizing_mask, graph):
    """Apply physical mesh-density weighting in inverse-softplus space."""
    elements = graph.current_element_index
    valid_elements = sizing_mask[elements].all(dim=1)
    if not valid_elements.any():
        return s_pred_phys.sum() * 0.0

    current_sizing = graph.current_sizing_field.clamp_min(1e-12)
    inverse_current = current_sizing + torch.log(-torch.expm1(-current_sizing))
    target_sizing = F.softplus(inverse_current + s_target)

    squared_error = (s_pred_phys - s_target).square()
    element_error = squared_error[elements].mean(dim=(1, 2))[valid_elements]
    element_target_sizing = target_sizing[elements].mean(dim=(1, 2))[valid_elements]
    element_area = graph.current_element_area[:, 0][valid_elements]
    # Previous physical-density weight. Applied directly to inverse-softplus
    # residual MSE, it overweights small target sizes by roughly 1 / h^2:
    # weights = element_area / element_target_sizing.square().clamp_min(1e-12)
    sizing_jacobian = 1.0 - torch.exp(-element_target_sizing)
    weights = element_area * (
        sizing_jacobian / element_target_sizing.clamp_min(1e-12)
    ).square()

    if hasattr(graph, "batch") and graph.batch is not None:
        graph_index = graph.batch[elements[valid_elements, 0]]
        num_graphs = int(graph.batch.max().item()) + 1
    else:
        graph_index = torch.zeros_like(element_error, dtype=torch.long)
        num_graphs = 1
    weighted_error = scatter_add(weights * element_error, graph_index, dim=0, dim_size=num_graphs)
    weight_sum = scatter_add(weights, graph_index, dim=0, dim_size=num_graphs)
    supervised_graphs = weight_sum > 0
    return (weighted_error[supervised_graphs] / weight_sum[supervised_graphs]).mean()


def compute_losses(
    u_pred_phys,
    s_pred,
    graph,
    normalizer,
    config,
    lambda_couple=None,
):
    if lambda_couple is None:
        lambda_couple = float(config.lambda_couple)
    mesh_dim_value = getattr(graph, "mesh_dim", 2)
    if torch.is_tensor(mesh_dim_value):
        mesh_dims = torch.unique(mesh_dim_value.detach()).cpu().tolist()
    elif isinstance(mesh_dim_value, (list, tuple)):
        mesh_dims = list(set(int(value) for value in mesh_dim_value))
    else:
        mesh_dims = [int(mesh_dim_value)]
    if len(mesh_dims) != 1:
        raise ValueError(f"A batch cannot mix mesh dimensions: {mesh_dims}")
    mesh_dim = int(mesh_dims[0])
    mask = graph.current_level_mask
    sizing_mask = getattr(graph, "sizing_target_mask", mask)
    u_target = graph.y_solution
    s_target = graph.y_sizing_log_ratio

    u_target_norm = normalizer.normalize_solution(u_target)
    u_pred_norm = normalizer.normalize_solution(u_pred_phys)
    s_phys = normalizer.denormalize_sizing(s_pred)
    if config.use_pivot_residual_heads:
        # Separate supervision prevents the numerous fine nodes from defining
        # the pivot objective, while the reconstructed field remains the
        # quantity used by rollout and evaluation.
        pivot = graph.pivot_mask & mask
        fine = graph.fine_mask & mask
        components = getattr(graph, "_pivot_components", None)
        if components is None:
            raise RuntimeError("pivot residual components were not attached to graph")
        u_base = components["u_base"]
        u_delta = components["u_delta"]
        s_base = normalizer.denormalize_sizing(components["s_base"])
        s_delta = components["s_delta"]
        idx = graph.pivot_interp_index.clamp_min(0)
        w = graph.pivot_interp_weight
        u_base_target = (u_target[idx] * w.unsqueeze(-1)).sum(dim=1)
        s_base_target = (s_target[idx] * w.unsqueeze(-1)).sum(dim=1)
        if getattr(config, "pivot_direct_values", False):
            u_base_target = torch.zeros_like(u_base_target)
            s_base_target = torch.zeros_like(s_base_target)
        l_phys = 0.5 * (
            F.mse_loss(normalizer.normalize_solution(u_base[pivot]), u_target_norm[pivot])
            + F.mse_loss(normalizer.normalize_solution(u_delta[fine]),
                         normalizer.normalize_solution((u_target-u_base_target)[fine]))
        )
        if bool(sizing_mask.any()):
            l_sizing = 0.5 * (
                F.mse_loss(s_base[pivot], s_target[pivot])
                + F.mse_loss(s_delta[fine], (s_target-s_base_target)[fine])
            )
        else:
            l_sizing = l_phys * 0.0
        # Keep legacy checkpoints reproducible; enable repaired coupling only
        # in explicitly versioned runs after the single-sample gate.
        if not getattr(config, "pivot_coupling_enabled", False):
            zero = l_phys * 0.0
            total = l_phys + config.lambda_sizing * l_sizing
            return total, {
                "loss": total, "L_phys": l_phys, "L_sizing": l_sizing,
                "L_couple": zero, "L_couple_weighted": zero,
                "lambda_couple": zero, "lambda_couple_target": zero,
                "s_mean": s_pred[mask].mean().detach(),
            }
        supervised_loss = config.lambda_phys * l_phys + config.lambda_sizing * l_sizing
        s_mean = s_pred[mask].mean().detach()
        if bool(sizing_mask.any()) and lambda_couple != 0.0:
            if config.couple_mode == "one_sided_gradient":
                l_couple = compute_one_sided_gradient_resolution_loss(
                    u_pred_phys, s_phys, sizing_mask, graph,
                    epsilon=config.couple_gradient_epsilon,
                    reference=getattr(config, "couple_reference", "mean"))
            elif config.couple_mode == "gradient_equidistribution":
                l_couple = compute_gradient_equidistribution_loss(
                    u_pred_phys, s_phys, sizing_mask, graph,
                    alpha=config.couple_gradient_alpha,
                    epsilon=config.couple_gradient_epsilon)
            else:
                raise ValueError(f"Unsupported four-head coupling: {config.couple_mode}")
        else:
            l_couple = l_phys * 0.0
        target_lambda = torch.as_tensor(lambda_couple, device=l_couple.device, dtype=l_couple.dtype)
        cap = float(config.couple_max_supervised_ratio)
        effective_lambda = torch.minimum(
            target_lambda,
            cap * supervised_loss.detach() / l_couple.detach().clamp_min(1e-12)
        ) if cap >= 0 else target_lambda
        weighted = effective_lambda * l_couple
        total = supervised_loss + weighted
        return total, {
            "loss": total.detach(), "L_phys": l_phys.detach(),
            "L_sizing": l_sizing.detach(), "L_couple": l_couple.detach(),
            "L_couple_weighted": weighted.detach(),
            "lambda_couple": effective_lambda.detach(),
            "lambda_couple_target": target_lambda.detach(), "s_mean": s_mean,
        }
    if config.solution_guidance_mode == "oracle_target_density":
        l_phys = compute_oracle_resolution_weighted_solution_loss(
            u_pred_norm, u_target_norm, s_target, sizing_mask, graph,
            epsilon=config.cross_guidance_epsilon,
            weight_cap=config.solution_guidance_weight_cap,
        )
    elif config.solution_guidance_mode == "threshold_density_excess":
        l_phys = compute_threshold_density_solution_loss(
            u_pred_norm, u_target_norm, s_phys, sizing_mask, graph,
            epsilon=config.cross_guidance_epsilon,
        )
    elif config.couple_mode == "bidirectional_cross_guidance":
        l_phys = compute_resolution_weighted_solution_loss(
            u_pred_norm, u_target_norm, s_phys, sizing_mask, graph,
            epsilon=config.cross_guidance_epsilon,
        )
    else:
        l_phys = F.mse_loss(u_pred_norm[mask], u_target_norm[mask])

    if sizing_mask.any():
        # Both values are residuals in softplus-inverse sizing space.
        if config.sizing_loss_mode == "density_weighted":
            l_sizing = compute_density_weighted_sizing_loss(
                s_phys, s_target, sizing_mask, graph
            )
        elif config.sizing_loss_mode == "amber_mse":
            l_sizing = F.mse_loss(s_phys[sizing_mask], s_target[sizing_mask])
        else:
            raise ValueError(f"Unknown sizing_loss_mode: {config.sizing_loss_mode}")
        s_mean = s_phys[sizing_mask].mean().detach()

        # Only the residual-based posterior mode is Poisson-specific and
        # triangle-only. One-sided gradient coupling is simplex-dimensional
        # and therefore also applies to tetrahedral Beam3D batches.
        coupling_supported = config.couple_mode != "posterior" or mesh_dim == 2
        if lambda_couple != 0.0 and coupling_supported:
            element_sizing_mask = sizing_mask[graph.current_element_index].all(dim=1)
            h_current = graph.current_sizing_field.clamp_min(1e-12)
            inverse_current = h_current + torch.log(-torch.expm1(-h_current))
            h_next = F.softplus(inverse_current + s_phys)
            log_ratio = torch.log(h_next / h_current)
            element_s = log_ratio[graph.current_element_index].mean(dim=(1, 2))
            if config.couple_mode == "gradient_equidistribution":
                l_couple = compute_gradient_equidistribution_loss(
                    u_pred_phys,
                    s_phys,
                    sizing_mask,
                    graph,
                    alpha=config.couple_gradient_alpha,
                    epsilon=config.couple_gradient_epsilon,
                )
            elif config.couple_mode == "one_sided_gradient":
                l_couple = compute_one_sided_gradient_resolution_loss(
                    u_pred_phys,
                    s_phys,
                    sizing_mask,
                    graph,
                    epsilon=config.couple_gradient_epsilon,
                    reference=getattr(config, "couple_reference", "mean"),
                )
            elif config.couple_mode == "bidirectional_cross_guidance":
                l_couple = compute_one_sided_gradient_constraint_loss(
                    u_pred_phys, s_phys, sizing_mask, graph,
                    epsilon=config.cross_guidance_epsilon,
                )
            elif config.couple_mode == "posterior":
                posterior_estimator = compute_posterior_estimator(u_pred_phys, graph)
                log_info = 4.0 * element_s[element_sizing_mask] + torch.log(
                    posterior_estimator[element_sizing_mask].clamp_min(1e-12)
                )
                if hasattr(graph, "batch") and graph.batch is not None:
                    graph_index = graph.batch[graph.current_element_index[element_sizing_mask, 0]]
                else:
                    graph_index = torch.zeros(
                        log_info.shape[0], dtype=torch.long, device=log_info.device
                    )
                mean_by_graph = scatter_mean(log_info, graph_index, dim=0).detach()
                squared_error = (log_info - mean_by_graph[graph_index]).square()
                l_couple = scatter_mean(squared_error, graph_index, dim=0).mean()
            else:
                raise ValueError(f"Unknown couple_mode: {config.couple_mode}")
        else:
            l_couple = s_pred.sum() * 0.0
    else:
        zero = s_pred.sum() * 0.0
        l_sizing = zero
        l_couple = zero
        s_mean = zero.detach()

    supervised_loss = (
        config.lambda_phys * l_phys + config.lambda_sizing * l_sizing
    )
    lambda_couple_target = torch.as_tensor(
        lambda_couple, device=l_couple.device, dtype=l_couple.dtype
    )
    max_ratio = float(config.couple_max_supervised_ratio)
    if max_ratio >= 0.0:
        lambda_couple_cap = (
            max_ratio
            * supervised_loss.detach()
            / l_couple.detach().clamp_min(1e-12)
        )
        lambda_couple_effective = torch.minimum(
            lambda_couple_target, lambda_couple_cap
        )
    else:
        lambda_couple_effective = lambda_couple_target
    l_couple_weighted = lambda_couple_effective * l_couple
    total = supervised_loss + l_couple_weighted
    return total, {
        "loss": total.detach(),
        "L_phys": l_phys.detach(),
        "L_sizing": l_sizing.detach(),
        "L_couple": l_couple.detach(),
        "L_couple_weighted": l_couple_weighted.detach(),
        "lambda_couple": lambda_couple_effective.detach(),
        "lambda_couple_target": lambda_couple_target.detach(),
        "s_mean": s_mean,
    }
