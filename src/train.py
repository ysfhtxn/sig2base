"""Training script for the Nanopore DNA Sequencing Basecaller.

Usage
-----
    python -m src.train --chunks chunks.npy --refs references.npy --ref_lens reference_lengths.npy \
        --train_batch_size 4 --eval_batch_size 4 --num_train_epochs 3 \
        --learning_rate 3e-4 --warmup_ratio 0.1 --weight_decay 0.01 \
        --logging_steps 5 --eval_strategy epoch --save_strategy epoch

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
        help="Path to chunks.npy (Bonito signal chunks).",
    )
    parser.add_argument(
        "--refs",
        help="Path to references.npy (Bonito integer-encoded references).",
    )
    parser.add_argument(
        "--ref_lens",
        help="Path to reference_lengths.npy (valid length per reference).",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output directory for checkpoints and adapter (default: ./nanopore_lora_adapter).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Per-device train batch size (default: 4).",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Per-device eval batch size (default: 4).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4).",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Logging steps (default: 5).",
    )
    parser.add_argument(
        "--eval_strategy",
        default="epoch",
        help="Evaluation strategy (default: epoch).",
    )
    parser.add_argument(
        "--save_strategy",
        default="epoch",
        help="Save strategy (default: epoch).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0).",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        help="Disable fp16 even if CUDA is available.",
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="Print model architecture and exit without training.",
    )
    args = parser.parse_args()

    if not args.print_model:
        missing = [
            name
            for name, value in (
                ("--chunks", args.chunks),
                ("--refs", args.refs),
                ("--ref_lens", args.ref_lens),
            )
            if value is None
        ]
        if missing:
            parser.error(
                "the following arguments are required: " + ", ".join(missing)
            )

    print("=" * 60)
    print("Nanopore DNA Sequencing Basecaller — Training")
    print("=" * 60)

    if args.print_model:
        model = get_nanopore_lora_model()
        
        try:
            from torchinfo import summary
            print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)
            
            dummy_input = torch.randn(4, 10000, dtype=torch.float32, device=model.device)
            
            summary(
                model,
                input_data=dummy_input,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                depth=7,
                row_settings=["var_names"]
            )
            print("="*75 + "\n")
            
        except ImportError:
            print("[Warning] 'torchinfo' 未安装 (pip install torchinfo)。回退到原生打印：")
            print(model)
        except Exception as e:
            print(f"[Warning] 结合 input_data 打印失败: {e}")
            print("正在尝试无维度输入的纯静态打印...")
            try:
                summary(model, col_names=["num_params", "trainable"], depth=7)
            except Exception as e2:
                print(f"[Error] 静态打印也失败了: {e2}")
                print(model)
                
        return

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

    model = get_nanopore_lora_model()

    collator = DataCollatorCTCWithPadding()

    use_fp16 = torch.cuda.is_available() and not args.no_fp16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=use_fp16,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
    )

    print(f"fp16 training: {use_fp16}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=_compute_metrics_for_trainer,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving PEFT adapter to '{args.output_dir}' …")
    model.save_pretrained(args.output_dir)
    print("Done. Training complete.")


if __name__ == "__main__":
    main()
