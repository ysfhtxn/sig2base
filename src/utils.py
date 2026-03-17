"""Utility functions and constants for the Nanopore DNA Sequencing Basecaller."""

from __future__ import annotations

import itertools
from typing import Dict, List

import jiwer


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB: Dict[str, int] = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "A": 4,
    "C": 5,
    "G": 6,
    "T": 7,
    "<blank>": 8,
}

ID_TO_TOKEN: Dict[int, str] = {v: k for k, v in VOCAB.items()}

BLANK_TOKEN_ID: int = VOCAB["<blank>"]
PAD_TOKEN_ID: int = VOCAB["<pad>"]


# ---------------------------------------------------------------------------
# CTC Decoding
# ---------------------------------------------------------------------------


def decode_ctc(token_ids: List[int]) -> str:
    """Decode a sequence of token IDs produced by CTC greedy search.

    The decoding process:
    1. Remove consecutive duplicate token IDs (CTC collapse).
    2. Remove the ``<blank>`` token (ID 8).
    3. Map remaining IDs to nucleotide characters.

    Args:
        token_ids: Raw per-frame token predictions (list of integer IDs).

    Returns:
        Decoded DNA string consisting of A/C/G/T characters.
    """
    # Step 1: collapse consecutive duplicates
    collapsed: List[int] = [k for k, _ in itertools.groupby(token_ids)]

    # Step 2: remove blank tokens and map to characters
    decoded_chars: List[str] = [
        ID_TO_TOKEN.get(tid, "<unk>")
        for tid in collapsed
        if tid != BLANK_TOKEN_ID
    ]

    return "".join(decoded_chars)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(pred_ids: List[List[int]], label_ids: List[List[int]]) -> Dict[str, float]:
    """Compute Character Error Rate (CER) and Identity Rate for CTC predictions.

    Blank tokens and padding tokens (-100) are handled before scoring.
    CER is computed using ``jiwer.cer`` (edit distance at character level).

    Args:
        pred_ids: List of per-sample prediction token ID sequences.
        label_ids: List of per-sample label token ID sequences.  Positions
            padded with ``-100`` are skipped.

    Returns:
        Dictionary with ``"cer"`` (float in [0, 1]) and
        ``"identity_rate"`` (1 - CER, float in [0, 1]).
    """
    predictions: List[str] = [decode_ctc(ids) for ids in pred_ids]

    references: List[str] = []
    for ids in label_ids:
        filtered = [tid for tid in ids if tid != -100]
        ref_chars = [ID_TO_TOKEN.get(tid, "") for tid in filtered]
        references.append("".join(ref_chars))

    # Ensure neither list is empty (jiwer requires non-empty strings)
    safe_references = [r if r else " " for r in references]
    safe_predictions = [p if p else " " for p in predictions]

    cer_score: float = jiwer.cer(safe_references, safe_predictions)

    return {
        "cer": cer_score,
        "identity_rate": 1.0 - cer_score,
    }
