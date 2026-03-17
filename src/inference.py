"""Inference script for the Nanopore DNA Sequencing Basecaller.

Usage
-----
    python -m src.inference [--adapter_path ./nanopore_lora_adapter]

The script:
1. Re-creates the base Wav2Vec2ForCTC architecture.
2. Loads the saved LoRA adapter weights via ``PeftModel.from_pretrained``.
3. Merges adapter weights into the base model (``merge_and_unload``) for
   fast, dependency-free inference.
4. Normalises a dummy 1-D Nanopore signal, runs greedy CTC decoding, and
   prints the predicted DNA sequence.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

from .dataset import normalize_signal
from .model import BLANK_TOKEN_ID, VOCAB_SIZE
from .utils import decode_ctc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER_PATH: str = "./nanopore_lora_adapter"
SAMPLE_RATE: int = 4_000        # Hz
DUMMY_SIGNAL_SECONDS: float = 0.5  # seconds of synthetic signal


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_merged_model(adapter_path: str) -> Wav2Vec2ForCTC:
    """Load and merge a LoRA-adapted Wav2Vec2ForCTC for efficient inference.

    Args:
        adapter_path: Directory produced by ``model.save_pretrained()`` that
            contains the LoRA adapter config and weights.

    Returns:
        Plain ``Wav2Vec2ForCTC`` with adapter weights merged into base
        parameters (no PEFT dependency at runtime).
    """
    print(f"Loading LoRA adapter from '{adapter_path}' …")

    # Reconstruct the same base config used during training
    base_config = Wav2Vec2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 1, 1, 1, 1, 1),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        feat_proj_dropout=0.0,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        final_dropout=0.1,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=0,
    )

    base_model = Wav2Vec2ForCTC(base_config)

    # Load LoRA adapter and merge
    peft_model: PeftModel = PeftModel.from_pretrained(base_model, adapter_path)
    print("Merging LoRA weights into base model …")
    merged_model: Wav2Vec2ForCTC = peft_model.merge_and_unload()
    merged_model.eval()

    print("Model ready for inference.")
    return merged_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def basecall(model: Wav2Vec2ForCTC, signal: np.ndarray) -> str:
    """Run greedy CTC basecalling on a single raw Nanopore signal.

    Args:
        model: Merged (or any) ``Wav2Vec2ForCTC`` in eval mode.
        signal: 1-D NumPy array of raw Nanopore current values.

    Returns:
        Predicted DNA sequence string (characters from {A, C, G, T}).
    """
    normalised = normalize_signal(signal)
    input_tensor = torch.tensor(normalised, dtype=torch.float32).unsqueeze(0)  # (1, T)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_values=input_tensor).logits  # (1, T', V)

    # Greedy argmax per frame
    pred_ids: list[int] = logits.squeeze(0).argmax(dim=-1).cpu().tolist()
    sequence: str = decode_ctc(pred_ids)
    return sequence


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main(adapter_path: str = DEFAULT_ADAPTER_PATH) -> None:
    """Load the trained model and basecall a dummy signal.

    Args:
        adapter_path: Path to the saved PEFT adapter directory.
    """
    print("=" * 60)
    print("Nanopore DNA Sequencing Basecaller — Inference")
    print("=" * 60)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: '{adapter_path}'. "
            "Run src/train.py first to produce the saved adapter."
        )

    model = load_merged_model(adapter_path)

    # Create a dummy 0.5-second signal (2 000 samples at 4 000 Hz)
    rng = np.random.default_rng(0)
    dummy_signal = rng.normal(loc=100.0, scale=10.0, size=int(SAMPLE_RATE * DUMMY_SIGNAL_SECONDS)).astype(np.float32)

    print(f"\nInput signal: {len(dummy_signal)} samples ({DUMMY_SIGNAL_SECONDS}s @ {SAMPLE_RATE} Hz)")

    predicted_sequence = basecall(model, dummy_signal)

    print(f"\nBasecalled sequence ({len(predicted_sequence)} nt):")
    print(f"  {predicted_sequence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nanopore basecaller inference")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to the saved PEFT LoRA adapter directory.",
    )
    args = parser.parse_args()
    main(adapter_path=args.adapter_path)
