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
from typing import Dict, List, Tuple

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


def _print_ctc_length_diagnostics(
    model: torch.nn.Module,
    signals: List[np.ndarray],
    labels: List[List[int]],
    split: int,
    max_samples: int = 512,
) -> Tuple[int, int]:
    """Print CTC alignment diagnostics before training starts.

    A CTC sample is invalid when the encoder output length is shorter than the
    target label length, i.e. ``T_out < U``. When ``ctc_zero_infinity=True``,
    those invalid samples can silently contribute zero loss.

    Returns:
        ``(n_invalid_train, n_invalid_eval)``.
    """
    # PEFT wrapper -> recover underlying Wav2Vec2ForCTC model.
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model

    if not hasattr(base_model, "_get_feat_extract_output_lengths"):
        print("[CTC-DIAG] Skipped: model has no _get_feat_extract_output_lengths().")
        return 0, 0

    def _diag_one_split(name: str, split_signals: List[np.ndarray], split_labels: List[List[int]]) -> int:
        n = min(max_samples, len(split_signals))
        if n == 0:
            print(f"[CTC-DIAG] {name}: empty split.")
            return 0

        input_lens = torch.tensor(
            [int(split_signals[i].shape[0]) for i in range(n)],
            dtype=torch.long,
        )
        target_lens = torch.tensor(
            [len(split_labels[i]) for i in range(n)],
            dtype=torch.long,
        )

        output_lens = base_model._get_feat_extract_output_lengths(input_lens).cpu()
        invalid_mask = output_lens < target_lens
        n_invalid = int(invalid_mask.sum().item())

        print(f"[CTC-DIAG] {name}: checked={n}, invalid(T_out<U)={n_invalid} ({n_invalid / n:.2%})")
        print(
            "[CTC-DIAG] "
            f"{name} lens stats: "
            f"input[min/med/max]={int(input_lens.min())}/{int(input_lens.median())}/{int(input_lens.max())}, "
            f"output[min/med/max]={int(output_lens.min())}/{int(output_lens.median())}/{int(output_lens.max())}, "
            f"target[min/med/max]={int(target_lens.min())}/{int(target_lens.median())}/{int(target_lens.max())}"
        )

        if n_invalid > 0:
            bad_idx = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)[:5].tolist()
            examples = [
                (
                    int(input_lens[i].item()),
                    int(output_lens[i].item()),
                    int(target_lens[i].item()),
                )
                for i in bad_idx
            ]
            print(f"[CTC-DIAG] {name} bad examples (input, output, target): {examples}")

        return n_invalid

    train_invalid = _diag_one_split("train", signals[:split], labels[:split])
    eval_invalid = _diag_one_split("eval", signals[split:], labels[split:])
    return train_invalid, eval_invalid


def _run_forward_sanity_check(
    model: torch.nn.Module,
    dataset: NanoporeDataset,
    collator: DataCollatorCTCWithPadding,
    max_items: int = 8,
) -> None:
    """Run one small forward pass and report numerical health.

    This catches issues that are often masked during Trainer logging
    (e.g. NaN/Inf loss filtered into 0.0 by ``logging_nan_inf_filter``).
    """
    n = min(max_items, len(dataset))
    if n == 0:
        print("[NUM-DIAG] skipped: empty training dataset.")
        return

    features = [dataset[i] for i in range(n)]
    batch = collator(features)

    input_values = batch["input_values"]
    labels = batch["labels"]

    input_is_finite = torch.isfinite(input_values).all().item()
    valid_labels = labels[labels != -100]
    if valid_labels.numel() > 0:
        label_min = int(valid_labels.min().item())
        label_max = int(valid_labels.max().item())
    else:
        label_min = -1
        label_max = -1

    print(
        "[NUM-DIAG] batch stats: "
        f"shape(input)={tuple(input_values.shape)}, "
        f"shape(labels)={tuple(labels.shape)}, "
        f"input_finite={bool(input_is_finite)}, "
        f"valid_label_range=[{label_min}, {label_max}]"
    )

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_values=input_values.to(device),
            labels=labels.to(device),
        )
    if model_was_training:
        model.train()

    loss = outputs.loss
    logits = outputs.logits
    loss_finite = torch.isfinite(loss).item()
    logits_finite_ratio = float(torch.isfinite(logits).float().mean().item())

    print(
        "[NUM-DIAG] forward stats: "
        f"loss={float(loss.detach().cpu()):.6f}, "
        f"loss_finite={bool(loss_finite)}, "
        f"logits_finite_ratio={logits_finite_ratio:.6f}"
    )

    if not loss_finite or logits_finite_ratio < 1.0 or not input_is_finite:
        raise ValueError(
            "Numerical sanity check failed before training: "
            "non-finite input/loss/logits detected. "
            "Try --no_fp16, lower --learning_rate (e.g. 3e-5), "
            "and check raw chunks for NaN/Inf."
        )


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
    parser.add_argument(
        "--strict_ctc_lengths",
        action="store_true",
        help=(
            "Abort training if any sampled item violates CTC length condition "
            "(encoder output length < target length)."
        ),
    )
    parser.add_argument(
        "--skip_forward_sanity_check",
        action="store_true",
        help="Skip one-batch numerical forward sanity check before Trainer.train().",
    )
    parser.add_argument(
        "--logging_nan_inf_filter",
        action="store_true",
        help=(
            "Enable Trainer NaN/Inf logging filter. Disabled by default to avoid "
            "masking numerical issues as 0.0 loss."
        ),
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
        model.eval() # 设置为评估模式，确保 Dropout 等行为一致

               
        try:
            from torchinfo import summary
            print("\n" + "="*30 + " MODEL SUMMARY " + "="*30)
            
            dummy_input = torch.randn(1, 10000, dtype=torch.float32, device=model.device)
            
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

    train_invalid, eval_invalid = _print_ctc_length_diagnostics(
        model=model,
        signals=signals,
        labels=labels,
        split=split,
    )
    if args.strict_ctc_lengths and (train_invalid > 0 or eval_invalid > 0):
        raise ValueError(
            "CTC length diagnostics found invalid samples (T_out < target_len). "
            "Please reduce downsampling or increase chunk length before training."
        )

    collator = DataCollatorCTCWithPadding()

    if not args.skip_forward_sanity_check:
        _run_forward_sanity_check(
            model=model,
            dataset=train_dataset,
            collator=collator,
        )

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    if use_fp16 and args.learning_rate >= 1e-4:
        print(
            "[Warning] fp16 + relatively high learning rate may cause overflow. "
            "If loss/grad becomes NaN, retry with --no_fp16 and/or lower --learning_rate."
        )

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
        logging_nan_inf_filter=args.logging_nan_inf_filter,
    )

    print(f"fp16 training: {use_fp16}")
    print(f"logging_nan_inf_filter: {args.logging_nan_inf_filter}")

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
