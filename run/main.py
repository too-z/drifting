"""Unified entrypoint for tabular geneartor training

    python -m run.main --gen --config configs/tabular_cleveland.yaml
    torchrun --standalone --nproc_per_node=8 -m run.main --gen --config ... 
    without --workdir the run lands in results/<config-stem>-<date>-<time>/.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument("--gen", action="store_true", help="Accepted for CLI compatibility generator training is the only mode")
parser.add_argument("--workdir", type=str, default=None, help="Local workdir root")


def main() -> None:
    args = parser.parse_args()
    # Import lazily so distributed init only runs once argv is parsed.
    from run.train import main as train_gen_main
    train_gen_main(args)


if __name__ == "__main__":
    main()
