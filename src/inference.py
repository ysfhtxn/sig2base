"""Inference script for the Nanopore DNA Sequencing Basecaller.

Usage
-----
    python -m src.inference [--adapter_path ./nanopore_lora_adapter]
                            [--pretrained_model facebook/wav2vec2-base]

The script:
1. Re-creates the Nanopore-adapted ``Wav2Vec2ForCTC`` base architecture via
   :func:`~model.build_nanopore_base_model` (the same function used during
   training) so the architecture is guaranteed to match.
2. Loads the saved LoRA adapter weights via ``PeftModel.from_pretrained``;
   the CNN and ``lm_head`` weights that were listed in ``modules_to_save``
   during training are automatically restored from the adapter checkpoint.
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
from transformers import Wav2Vec2ForCTC

from .dataset import normalize_signal
from .model import build_nanopore_base_model
from .utils import decode_ctc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER_PATH: str = "./nanopore_lora_adapter"
DEFAULT_PRETRAINED_MODEL: str = "facebook/wav2vec2-base"
SAMPLE_RATE: int = 4_000        # Hz
DUMMY_SIGNAL_SECONDS: float = 0.5  # seconds of synthetic signal


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_merged_model(
    adapter_path: str,
    pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
) -> Wav2Vec2ForCTC:
    """Load and merge a LoRA-adapted Wav2Vec2ForCTC for efficient inference.

    Uses :func:`~model.build_nanopore_base_model` to reconstruct the exact
    same base architecture that was used during training, then loads the
    saved LoRA adapter (which also contains the trained CNN and ``lm_head``
    weights stored via ``modules_to_save``).

    Args:
        adapter_path: Directory produced by ``peft_model.save_pretrained()``
            that contains the LoRA adapter config and weights.
        pretrained_model_name: HuggingFace model id or local path of the
            pre-trained Wav2Vec2 checkpoint.  Must match the one used during
            training (default: ``"facebook/wav2vec2-base"``).

    Returns:
        Plain ``Wav2Vec2ForCTC`` with all adapter and CNN weights merged into
        the base parameters (no PEFT dependency at runtime).
    """
    print(f"Loading LoRA adapter from '{adapter_path}' …")

    # Reconstruct the exact same base model used during training.
    base_model = build_nanopore_base_model(pretrained_model_name)

    # Load LoRA adapter weights.  The saved checkpoint also contains the
    # trained CNN (feature_extractor / feature_projection) and lm_head
    # weights because those modules were listed in modules_to_save.
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


def main(
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
) -> None:
    """Load the trained model and basecall a dummy signal.

    Args:
        adapter_path: Path to the saved PEFT adapter directory.
        pretrained_model_name: HuggingFace model id or local path of the
            pre-trained Wav2Vec2 checkpoint used during training.
    """
    print("=" * 60)
    print("Nanopore DNA Sequencing Basecaller — Inference")
    print("=" * 60)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter directory not found: '{adapter_path}'. "
            "Run src/train.py first to produce the saved adapter."
        )

    model = load_merged_model(adapter_path, pretrained_model_name)

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
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=DEFAULT_PRETRAINED_MODEL,
        help="HuggingFace model id or local path used during training (default: facebook/wav2vec2-base).",
    )
    args = parser.parse_args()
    main(adapter_path=args.adapter_path, pretrained_model_name=args.pretrained_model)
