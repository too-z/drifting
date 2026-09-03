"""Unified entrypoint for tabular geneartor training

    python -m run.main --gen --config configs/tabular_cleveland.yaml --workdir runs/x
    torchrun --standalone --nproc_per_node=8 -m run.main --gen --config ... --workdir ...
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--gen", action="store_true", help="Accepted for CLI compatibility generator training is the only mode")
parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs")


def main() -> None:
    args = parser.parse_args()
    args.output_dir = args.workdir
    # Import lazily so distributed init only runs once argv is parsed.
    from run.train import main as train_gen_main
    train_gen_main(args)


if __name__ == "__main__":
    main()
