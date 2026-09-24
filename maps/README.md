# MAPS

Joint prediction of physical solutions and sizing fields on iteratively
updated meshes using a pivot-augmented graph and interpolated pivot bases
with non-pivot offsets.

This repository contains Poisson data generation and training. Standalone
evaluation, visualization, baseline runners and pretrained weights are not included.

## Installation

Python 3.12 was used for validation. Install PyTorch for your CUDA version,
then install a matching `torch-scatter` wheel following the
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
pip install -r requirements.txt
```

On headless Debian/Ubuntu, Gmsh may require the system package `libglu1-mesa`.
Training is intended for an NVIDIA GPU; memory requirements depend on graph size.

## Training

Run from the repository root:

```bash
# Same instance in train, validation and test.
python run_poisson.py --single --epochs 100

# Full training: 60/20/20 instances.
python run_poisson.py --epochs 300
```

The launcher generates and caches data on first use. Training retains its
validation, checkpoint selection and final test pass. No standalone evaluation
or plotting is launched. Early stopping can end training before the requested
maximum epoch count.

| Setting | Full | Single instance |
| --- | --- | --- |
| Seed | 42 | 42 |
| Latent dimension | 64 | 64 |
| Message-passing layers | 20 | 20 |
| Pivot fraction | 10% | 10% |
| Remeshing transitions | 3 | 3 |
| Steps per epoch | 128 | 16 |
| Replay capacity (graphs) | 240 | 32 |
| Trajectories per refresh | 8 | 1 |

Replay starts at epoch 5 and refreshes every five epochs.
Data is stored in `datasets/{single,full}/`; training output is stored in
`outputs/{single,full}/maps/seed42/`, with checkpoints in `checkpoints/`.
Generated artifacts are excluded from Git.

## Target Meshes

The initial mesh and 50-step AMR expert mesh are retained as endpoints.
For `k = 0, 1`, extract actual current nodal sizes `h`, P1-project expert
nodal sizes onto current vertices as `h_expert`, and generate the next mesh
with the Gmsh field:

```text
h_generation = h + (h_expert - h) / (3 - k)
```

Sizes and expert projections are recomputed after each remeshing step.
The final target is the expert mesh, yielding levels `[0,1,2,3]`.
Old intermediate-AMR-snapshot caches are rejected.

## Implementation

Pivots use hop-based farthest-point sampling. Base fields use barycentric
interpolation with inverse-distance fallback. Pivot values and non-pivot
offsets have separate readout families for solution and sizing.

The training configuration retains sampled-vertex sizing labels,
inverse-softplus sizing residuals and transition-specific sizing heads.
Solution weighting is disabled. `pivot_coupling_enabled=False` in the launcher
also disables coupling for this configuration. These describe the released
implementation, rather than an idealized formulation.

## Layout

- `run_poisson.py`: single-instance and full-training launcher.
- `generate_poisson_dataset.py`, `data_generator/`: Poisson FEM and data generation.
- `dataset.py`: target meshes, supervision and replay storage.
- `graph_util.py`, `hop_coarse.py`: graph construction and pivots.
- `model.py`, `mpn.py`: neural model and message passing.
- `algorithm.py`, `losses.py`, `normalizer.py`: training and rollout logic.
- `mesh_util.py`: interpolation, sizing and Gmsh utilities.
- `config.py`, `train_poisson.py`: configuration and training entry point.

## Checks

```bash
python -m unittest test_targets -v
```

Tests cover the three-transition default and sequential recomputation on newly
generated meshes. Single-instance training for 100 epochs and a complete learned
rollout were exercised before removing evaluation artifacts.
