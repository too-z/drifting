"""Unified entrypoint — torch port of main.py (identical CLI).

    python -m pt.main --gen --config configs/gen/latent_ablation.yaml --workdir runs/x
    torchrun --standalone --nproc_per_node=8 -m pt.main --gen --config ... --workdir ...
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--gen", action="store_true", help="Train generator (default: train MAE)")
parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs")


def main() -> None:
    args = parser.parse_args()
    args.output_dir = args.workdir
    # Import lazily so distributed init only runs for the active path.
    if args.gen:
        from pt.train import main as train_gen_main

        train_gen_main(args)
    else:
        from pt.train_mae import main as train_mae_main

        train_mae_main(args)


if __name__ == "__main__":
    main()
