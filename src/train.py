"""Training script for the Nanopore DNA Sequencing Basecaller.

Usage
-----
    python -m src.train

The script generates a small synthetic dataset (random 1-D signals and random
ACGT sequences), builds the LoRA-augmented Wav2Vec2ForCTC model, and runs a
short training loop via the Hugging Face ``Trainer`` API.  The final PEFT
adapter weights are saved to ``./nanopore_lora_adapter/``.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import EvalPrediction, Trainer, TrainingArguments

from .dataset import DataCollatorCTCWithPadding, NanoporeDataset
from .model import get_nanopore_lora_model
from .utils import VOCAB, compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCLEOTIDES: str = "ACGT"
SAMPLE_RATE: int = 4_000        # Hz
BASES_PER_SEC: int = 400        # nominal Nanopore translocation speed
SAMPLES_PER_BASE: int = SAMPLE_RATE // BASES_PER_SEC  # = 10

OUTPUT_DIR: str = "./nanopore_lora_adapter"


# ---------------------------------------------------------------------------
# Dummy data generation
# ---------------------------------------------------------------------------


def generate_dummy_data(
    num_samples: int = 64,
    min_bases: int = 20,
    max_bases: int = 80,
    rng_seed: int = 42,
) -> Tuple[List[np.ndarray], List[str]]:
    """Generate synthetic Nanopore signals and random DNA sequences.

    Each signal is simulated as Gaussian noise (mimicking pore current
    fluctuations) with a length proportional to the sequence length:
    ``len(sequence) * SAMPLES_PER_BASE`` samples.

    Args:
        num_samples: Number of (signal, sequence) pairs to generate.
        min_bases: Minimum sequence length in nucleotides.
        max_bases: Maximum sequence length in nucleotides.
        rng_seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(signals, sequences)`` where ``signals`` is a list of 1-D
        float32 NumPy arrays and ``sequences`` is a list of ACGT strings.
    """
    rng = np.random.default_rng(rng_seed)
    random.seed(rng_seed)

    signals: List[np.ndarray] = []
    sequences: List[str] = []

    for _ in range(num_samples):
        num_bases = rng.integers(min_bases, max_bases + 1)
        seq = "".join(random.choices(NUCLEOTIDES, k=int(num_bases)))
        # Raw current is modelled as Gaussian noise; amplitude varies per base
        signal_len = int(num_bases) * SAMPLES_PER_BASE
        signal = rng.normal(loc=0.0, scale=1.0, size=signal_len).astype(np.float32)
        signals.append(signal)
        sequences.append(seq)

    print(
        f"Generated {num_samples} synthetic samples "
        f"(sequence length {min_bases}–{max_bases} nt)."
    )
    return signals, sequences


# ---------------------------------------------------------------------------
# Metrics wrapper for Trainer
# ---------------------------------------------------------------------------


def _compute_metrics_for_trainer(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Adapter between ``EvalPrediction`` and ``compute_metrics``.

    Args:
        eval_pred: Named tuple with ``predictions`` (logits, shape
            ``[B, T, V]``) and ``label_ids`` (shape ``[B, L]``).

    Returns:
        Dictionary with ``"cer"`` and ``"identity_rate"``.
    """
    logits: np.ndarray = eval_pred.predictions
    label_ids: np.ndarray = eval_pred.label_ids

    # Greedy argmax over vocabulary dimension
    pred_ids: List[List[int]] = np.argmax(logits, axis=-1).tolist()
    label_ids_list: List[List[int]] = label_ids.tolist()

    return compute_metrics(pred_ids, label_ids_list)


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full training pipeline on synthetic dummy data."""
    print("=" * 60)
    print("Nanopore DNA Sequencing Basecaller — Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    signals, sequences = generate_dummy_data(num_samples=64)

    split = int(0.8 * len(signals))
    train_signals, eval_signals = signals[:split], signals[split:]
    train_seqs, eval_seqs = sequences[:split], sequences[split:]

    train_dataset = NanoporeDataset(train_signals, train_seqs)
    eval_dataset = NanoporeDataset(eval_signals, eval_seqs)

    print(f"Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model = get_nanopore_lora_model()

    # ------------------------------------------------------------------
    # 3. Data collator
    # ------------------------------------------------------------------
    collator = DataCollatorCTCWithPadding()

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=use_fp16,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    print(f"fp16 training: {use_fp16}")

    # ------------------------------------------------------------------
    # 5. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_metrics_for_trainer,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save PEFT adapter
    # ------------------------------------------------------------------
    print(f"Saving PEFT adapter to '{OUTPUT_DIR}' …")
    model.save_pretrained(OUTPUT_DIR)
    print("Done. Training complete.")


if __name__ == "__main__":
    main()
