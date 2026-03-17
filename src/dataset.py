"""Dataset and data-collation utilities for the Nanopore DNA Sequencing Basecaller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import VOCAB


# ---------------------------------------------------------------------------
# Signal Normalisation
# ---------------------------------------------------------------------------


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Apply Z-score normalisation to a 1-D Nanopore electrical signal.

    Each signal chunk is independently normalised so that the model receives
    a zero-mean, unit-variance input regardless of absolute pore current
    levels.

    Args:
        signal: 1-D array of raw Nanopore current values (picoamperes).

    Returns:
        Normalised signal with mean ≈ 0 and standard deviation ≈ 1.
        If the standard deviation is zero (constant signal) the original
        signal is returned unchanged to avoid division by zero.
    """
    mean: float = float(np.mean(signal))
    std: float = float(np.std(signal))
    if std < 1e-8:
        return signal.astype(np.float32)
    return ((signal - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NanoporeDataset(Dataset):
    """PyTorch Dataset wrapping paired (signal, sequence) Nanopore examples.

    Each item consists of:
    - ``input_values``: Z-score normalised 1-D float tensor of raw current
      values.
    - ``labels``: 1-D integer tensor of token IDs for the reference DNA
      sequence (A/C/G/T only; no blank or special tokens).

    Args:
        signals: List of 1-D NumPy arrays, one per read.
        sequences: List of DNA strings (e.g. ``"ACGT"``), one per read.
        vocab: Mapping from token string to integer ID (defaults to VOCAB).
    """

    def __init__(
        self,
        signals: List[np.ndarray],
        sequences: List[str],
        vocab: Optional[Dict[str, int]] = None,
    ) -> None:
        if len(signals) != len(sequences):
            raise ValueError(
                f"Number of signals ({len(signals)}) must match number of "
                f"sequences ({len(sequences)})."
            )
        self.signals = signals
        self.sequences = sequences
        self.vocab: Dict[str, int] = vocab if vocab is not None else VOCAB
        unk_id: int = self.vocab.get("<unk>", 3)
        self._unk_id = unk_id

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single (input_values, labels) pair.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys ``"input_values"`` (float32 tensor) and
            ``"labels"`` (int64 tensor).
        """
        signal: np.ndarray = normalize_signal(self.signals[idx])
        input_values: torch.Tensor = torch.tensor(signal, dtype=torch.float32)

        label_ids: List[int] = [
            self.vocab.get(ch, self._unk_id) for ch in self.sequences[idx]
        ]
        labels: torch.Tensor = torch.tensor(label_ids, dtype=torch.long)

        return {"input_values": input_values, "labels": labels}


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------


@dataclass
class DataCollatorCTCWithPadding:
    """Collate variable-length signals and label sequences into padded batches.

    - Signals are padded with ``0.0`` (electrically neutral / normalised zero).
    - Labels are padded with ``-100`` so that the CTC loss ignores padded
      positions.

    Args:
        pad_signal_value: Value used to right-pad shorter signals (default 0.0).
        label_pad_token_id: Value used to right-pad shorter label sequences
            (default -100).
    """

    pad_signal_value: float = 0.0
    label_pad_token_id: int = -100

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Pad and stack a list of dataset items into a single batch dict.

        Args:
            features: List of dicts, each with ``"input_values"`` and
                ``"labels"`` tensors of variable length.

        Returns:
            Batch dictionary with ``"input_values"`` (B, T_max) float32
            tensor and ``"labels"`` (B, L_max) int64 tensor.
        """
        input_values_list: List[torch.Tensor] = [f["input_values"] for f in features]
        labels_list: List[torch.Tensor] = [f["labels"] for f in features]

        # Pad signals
        max_signal_len: int = max(iv.shape[0] for iv in input_values_list)
        padded_signals: List[torch.Tensor] = []
        for iv in input_values_list:
            pad_len = max_signal_len - iv.shape[0]
            padded = torch.nn.functional.pad(
                iv, (0, pad_len), value=self.pad_signal_value
            )
            padded_signals.append(padded)
        batched_input: torch.Tensor = torch.stack(padded_signals, dim=0)

        # Pad labels
        max_label_len: int = max(lb.shape[0] for lb in labels_list)
        padded_labels: List[torch.Tensor] = []
        for lb in labels_list:
            pad_len = max_label_len - lb.shape[0]
            padded = torch.nn.functional.pad(
                lb, (0, pad_len), value=self.label_pad_token_id
            )
            padded_labels.append(padded)
        batched_labels: torch.Tensor = torch.stack(padded_labels, dim=0)

        return {
            "input_values": batched_input,
            "labels": batched_labels,
        }
