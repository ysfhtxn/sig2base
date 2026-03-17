"""Training script for the Nanopore DNA Sequencing Basecaller.

Usage
-----
    python -m src.train --chunks chunks.npy --refs references.npy --ref_lens reference_lengths.npy

The script loads real Nanopore data pre-processed by Bonito, builds the
LoRA-augmented Wav2Vec2ForCTC model, and runs a training loop via the
Hugging Face ``Trainer`` API.  The final PEFT adapter weights are saved to
``./nanopore_lora_adapter/``.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import EvalPrediction, Trainer, TrainingArguments

from .data_loader import load_bonito_npy_data
from .dataset import DataCollatorCTCWithPadding, NanoporeDataset
from .model import get_nanopore_lora_model
from .utils import VOCAB, compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = "./nanopore_lora_adapter"
TRAIN_SPLIT_RATIO: float = 0.9


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
    """Run the full training pipeline on real Bonito-preprocessed data."""
    parser = argparse.ArgumentParser(
        description="Train the Nanopore DNA Sequencing Basecaller."
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks.npy (Bonito signal chunks).",
    )
    parser.add_argument(
        "--refs",
        required=True,
        help="Path to references.npy (Bonito integer-encoded references).",
    )
    parser.add_argument(
        "--ref_lens",
        required=True,
        help="Path to reference_lengths.npy (valid length per reference).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Nanopore DNA Sequencing Basecaller — Training")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    signals, labels = load_bonito_npy_data(
        chunks_path=args.chunks,
        refs_path=args.refs,
        ref_lens_path=args.ref_lens,
    )

    split = int(TRAIN_SPLIT_RATIO * len(signals))
    train_signals, eval_signals = signals[:split], signals[split:]
    train_labels, eval_labels = labels[:split], labels[split:]

    train_dataset = NanoporeDataset(train_signals, train_labels)
    eval_dataset = NanoporeDataset(eval_signals, eval_labels)

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
