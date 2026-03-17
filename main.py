"""Top-level entry-point for the Nanopore DNA Sequencing Basecaller.

Run training:
    python main.py train

Run inference (requires a pre-trained adapter):
    python main.py inference [--adapter_path PATH]
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nanopore DNA Sequencing Basecaller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train ---
    subparsers.add_parser("train", help="Train the basecaller on dummy data.")

    # --- inference ---
    inf_parser = subparsers.add_parser(
        "inference", help="Run basecalling inference on a dummy signal."
    )
    inf_parser.add_argument(
        "--adapter_path",
        type=str,
        default="./nanopore_lora_adapter",
        help="Path to saved PEFT LoRA adapter (default: ./nanopore_lora_adapter).",
    )

    args = parser.parse_args()

    if args.command == "train":
        from src.train import main as train_main
        train_main()
    elif args.command == "inference":
        from src.inference import main as inference_main
        inference_main(adapter_path=args.adapter_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
