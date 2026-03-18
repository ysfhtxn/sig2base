# sig2base — Nanopore DNA Sequencing Basecaller

A parameter-efficient Nanopore basecaller built on top of Facebook's
[wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) using
[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) via the
🤗 PEFT library.

---

## Architecture

| Component | Details |
|---|---|
| **Base model** | `facebook/wav2vec2-base` (pre-trained Transformer, weights frozen) |
| **LoRA adapters** | Rank-8 adapters on `q_proj` / `v_proj` in all Transformer layers |
| **CNN feature extractor** | 3-layer stack with strides `(5, 2, 1)` → ~10× downsample, trained from scratch |
| **Classification head** | Linear `lm_head` over a 9-token DNA vocabulary, trained from scratch |
| **Vocabulary** | `<pad>`, `<s>`, `</s>`, `<unk>`, `A`, `C`, `G`, `T`, `<blank>` |
| **Loss** | CTC (Connectionist Temporal Classification) |

The CNN strides are tuned for Nanopore signals sampled at ~4 kHz, where the
standard wav2vec2 7-layer CNN (320× downsample) would be too aggressive.

Because the CNN and `lm_head` are randomly re-initialized, they are listed in
`modules_to_save` inside the LoRA config so their trained weights are stored
alongside the LoRA deltas in the adapter checkpoint.

---

## Installation

```bash
# Python ≥ 3.10 required
pip install -e .
# or with uv:
uv sync
```

---

## Data preparation (Bonito format)

Pre-process raw Nanopore reads with
[Bonito](https://github.com/nanoporetech/bonito) to obtain three numpy files:

| File | Shape | Description |
|---|---|---|
| `chunks.npy` | `(N, chunk_len)` | Normalised signal chunks (float32) |
| `references.npy` | `(N, max_ref_len)` | Ground-truth bases encoded as 1=A, 2=C, 3=G, 4=T, 0=pad |
| `reference_lengths.npy` | `(N,)` | Valid length of each reference |

---

## Training

```bash
python -m src.train \
    --chunks      data/chunks.npy \
    --refs        data/references.npy \
    --ref_lens    data/reference_lengths.npy \
    --output_dir  ./nanopore_lora_adapter \
    --num_train_epochs 10 \
    --train_batch_size 8 \
    --eval_batch_size  8 \
    --learning_rate 3e-4 \
    --warmup_ratio 0.1
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--chunks` | — | Path to `chunks.npy` (required) |
| `--refs` | — | Path to `references.npy` (required) |
| `--ref_lens` | — | Path to `reference_lengths.npy` (required) |
| `--output_dir` | `./nanopore_lora_adapter` | Where to save the adapter |
| `--num_train_epochs` | 3 | Training epochs |
| `--learning_rate` | 3e-4 | Learning rate |
| `--no_fp16` | — | Disable fp16 even on CUDA |
| `--print_model` | — | Print model summary and exit |

The final LoRA adapter (including trained CNN and `lm_head` weights) is saved
to `--output_dir`.

---

## Inference

```bash
python -m src.inference \
    --adapter_path  ./nanopore_lora_adapter \
    --pretrained_model facebook/wav2vec2-base
```

Or via the top-level entry-point:

```bash
python main.py inference \
    --adapter_path ./nanopore_lora_adapter \
    --pretrained_model facebook/wav2vec2-base
```

`--pretrained_model` must match the checkpoint used during training.

---

## Programmatic use

```python
from src.model import build_nanopore_base_model, get_nanopore_lora_model
from src.inference import load_merged_model, basecall
import numpy as np

# --- Training ---
peft_model = get_nanopore_lora_model("facebook/wav2vec2-base")
# ... train with HuggingFace Trainer ...
peft_model.save_pretrained("./nanopore_lora_adapter")

# --- Inference ---
model = load_merged_model(
    "./nanopore_lora_adapter",
    pretrained_model_name="facebook/wav2vec2-base",  # must match training
)
signal = np.random.randn(4000).astype("float32")   # 1 s @ 4 kHz
sequence = basecall(model, signal)
print(sequence)   # e.g. "ACGTACGT..."
```

---

## Project structure

```
sig2base/
├── main.py              # CLI entry-point (train / inference)
├── src/
│   ├── model.py         # build_nanopore_base_model, get_nanopore_lora_model
│   ├── train.py         # HuggingFace Trainer-based training loop
│   ├── inference.py     # load_merged_model, basecall
│   ├── dataset.py       # NanoporeDataset, DataCollatorCTCWithPadding
│   ├── data_loader.py   # Bonito .npy loader, Pod5 reader
│   └── utils.py         # VOCAB, decode_ctc, compute_metrics
└── pyproject.toml
```
