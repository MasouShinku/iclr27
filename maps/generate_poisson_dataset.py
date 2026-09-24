"""Generate and cache deterministic Poisson AMR trajectory splits."""

import logging

from config import Config
from dataset import save_trajectory_splits


def main():
    config = Config.from_args()
    if not config.trajectory_data_dir:
        raise ValueError("trajectory_data_dir is required.")
    logging.getLogger("skfem.mesh.mesh").setLevel(logging.ERROR)
    print(f"Writing trajectory cache to: {config.trajectory_data_dir}")
    print(f"split: {config.num_train}/{config.num_val}/{config.num_test}")
    print(f"refinement_steps: {config.refinement_steps}")
    print(f"trajectory_levels: {config.trajectory_levels}")
    save_trajectory_splits(config, config.trajectory_data_dir)
    print("Trajectory cache complete.")


if __name__ == "__main__":
    main()
