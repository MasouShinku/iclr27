# Single: python run_poisson.py --single --epochs 100
# Full: python run_poisson.py --epochs 300
"""Run the copied final Poisson configuration with sequential K=3 targets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    mode = 'single' if args.single else 'full'
    cache = ROOT / 'datasets' / mode
    output = ROOT / 'outputs' / mode
    cfg = dict(task_name='poisson', seed=42,
        num_train=1 if args.single else 60, num_val=1 if args.single else 20,
        num_test=1 if args.single else 20, same_sample_splits=args.single,
        trajectory_levels=[0, 1, 2, 3], refinement_steps=50,
        trajectory_data_dir=str(cache), output_dir=str(output), experiment_name='maps',
        latent_dimension=64, num_mpn_layers=20, decoder_hidden_dim=64,
        use_hop_coarse_graph=True, use_pivot_residual_heads=True,
        sizing_projection_mode='sampled_vertex', sizing_damping_factor=1.0,
        couple_mode='one_sided_gradient', pivot_coupling_enabled=False,
        freeze_normalizer_after_initialization=True, max_epochs=args.epochs,
        steps_per_epoch=16 if args.single else 128,
        rollout_replay_size=32 if args.single else 240,
        rollout_replay_trajectories=1 if args.single else 8,
        early_stopping_patience=30)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    overrides = [f'{k}={v if isinstance(v, str) else json.dumps(v)}' for k, v in cfg.items()]
    if not (cache / 'manifest.json').exists():
        subprocess.run([sys.executable, 'generate_poisson_dataset.py', *overrides],
                       cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, 'train_poisson.py', *overrides], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
