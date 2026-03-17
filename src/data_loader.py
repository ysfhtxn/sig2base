"""Data loading utilities for the Nanopore DNA Sequencing Basecaller.

Supports loading pre-processed Bonito numpy arrays and raw Pod5 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np

from .dataset import normalize_signal

# ---------------------------------------------------------------------------
# Bonito label mapping
# ---------------------------------------------------------------------------

# Bonito encodes bases as: 1=A, 2=C, 3=G, 4=T, 0=pad
# Our VOCAB encodes:       4=A, 5=C, 6=G, 7=T
_BONITO_TO_VOCAB: dict[int, int] = {
    1: 4,  # A
    2: 5,  # C
    3: 6,  # G
    4: 7,  # T
}


# ---------------------------------------------------------------------------
# Bonito .npy loader
# ---------------------------------------------------------------------------


def load_bonito_npy_data(
    chunks_path: str | Path,
    refs_path: str | Path,
    ref_lens_path: str | Path,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Load Bonito-preprocessed Nanopore data from numpy arrays.

    Bonito outputs three numpy files:
    - ``chunks.npy``: float array of shape ``(N, chunk_length)`` or
      ``(N, chunk_length, 1)`` — normalised signal chunks.
    - ``references.npy``: int array of shape ``(N, max_ref_length)`` —
      ground-truth sequences encoded as 1=A, 2=C, 3=G, 4=T, 0=pad.
    - ``reference_lengths.npy``: int array of shape ``(N,)`` — valid length
      of each reference sequence (used to strip padding).

    The integer labels are remapped from Bonito's encoding to the project's
    VOCAB (A→4, C→5, G→6, T→7).  Samples whose valid reference length is 0
    are skipped.

    Args:
        chunks_path: Path to ``chunks.npy``.
        refs_path: Path to ``references.npy``.
        ref_lens_path: Path to ``reference_lengths.npy``.

    Returns:
        Tuple ``(signals, labels)`` where ``signals`` is a list of 1-D
        float32 NumPy arrays and ``labels`` is a list of lists of integer
        token IDs in our VOCAB encoding.
    """
    chunks = np.load(str(chunks_path), mmap_mode="r")
    references = np.load(str(refs_path), mmap_mode="r")
    reference_lengths = np.load(str(ref_lens_path), mmap_mode="r")

    # Squeeze to 2-D (N, chunk_length) and ensure float32
    if chunks.ndim == 3:
        chunks = chunks.squeeze(-1)
    chunks = chunks.astype(np.float32)

    signals: List[np.ndarray] = []
    labels: List[List[int]] = []

    n_total = len(chunks)
    n_skipped = 0

    for i in range(n_total):
        ref_len = int(reference_lengths[i])
        if ref_len == 0:
            n_skipped += 1
            continue

        # Slice off padding using the valid reference length
        raw_ref = references[i, :ref_len]
        mapped = [_BONITO_TO_VOCAB[b_int] for b in raw_ref if (b_int := int(b)) in _BONITO_TO_VOCAB]

        if len(mapped) == 0:
            n_skipped += 1
            continue

        signals.append(chunks[i].copy())
        labels.append(mapped)

    print(
        f"Loaded {len(signals)} samples from Bonito data "
        f"({n_skipped} skipped due to empty references)."
    )
    return signals, labels


# ---------------------------------------------------------------------------
# Pod5 reader (bonus utility)
# ---------------------------------------------------------------------------


def read_pod5_signals(pod5_path: str | Path) -> Generator[np.ndarray, None, None]:
    """Yield normalised signal arrays from a Pod5 file.

    Requires the ``pod5`` package (``pip install pod5``).

    The raw integer signal stored inside each Pod5 read is extracted,
    converted to float32, and passed through :func:`~dataset.normalize_signal`
    before being yielded so that downstream code can consume it directly.

    Args:
        pod5_path: Path to the ``.pod5`` file.

    Yields:
        Normalised 1-D float32 NumPy arrays, one per read.
    """
    try:
        import pod5  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'pod5' package is required to read .pod5 files. "
            "Install it with: pip install pod5"
        ) from exc

    with pod5.Reader(str(pod5_path)) as reader:
        for read in reader.reads():
            signal = read.signal.astype(np.float32)
            yield normalize_signal(signal)
