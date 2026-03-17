"""Model construction for the Nanopore DNA Sequencing Basecaller.

Architecture summary
--------------------
- Base model: ``Wav2Vec2ForCTC`` initialised **from scratch** (random weights)
  with a compact config tuned to Nanopore signal characteristics.
- Parameter-efficient fine-tuning: LoRA is injected into the Transformer
  attention layers (``q_proj`` / ``v_proj``).
- The CNN feature extractor and feature-projection layers are **unfrozen** so
  the model can learn to encode raw Nanopore current from scratch.
- The ``lm_head`` classification head is always fully trainable because the
  vocabulary (size 9) differs from any pre-trained checkpoint.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel  # re-exported for convenience
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 9          # <pad>, <s>, </s>, <unk>, A, C, G, T, <blank>
BLANK_TOKEN_ID: int = 8      # index of <blank> in VOCAB


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def get_nanopore_lora_model() -> "PeftModel":
    """Build a LoRA-wrapped Wav2Vec2ForCTC model for Nanopore basecalling.

    Configuration highlights
    ~~~~~~~~~~~~~~~~~~~~~~~~
    - ``vocab_size=9``: matches the Nanopore nucleotide vocabulary.
    - ``hidden_size=512``, ``num_hidden_layers=6``: compact transformer that
      is practical to train from scratch on CPU/single-GPU.
    - ``conv_stride=(5, 2, 1)``: the three CNN stages downsample the 4 000 Hz
      signal by 5 × 2 × 1 = 10 ×, producing roughly one feature frame per
      base (10 samples / base at 400 bases / sec).
    - LoRA targets ``q_proj`` and ``v_proj``; ``modules_to_save=["lm_head"]``
      keeps the classification head fully trainable.
    - CNN layers are explicitly unfrozen after LoRA injection.

    Returns:
        PEFT-wrapped model with trainable LoRA adapters and CNN layers.
    """
    # ------------------------------------------------------------------
    # 1. Build the base Wav2Vec2ForCTC configuration
    # ------------------------------------------------------------------
    config = Wav2Vec2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,        # 512 / 8 = 64 head-dim
        intermediate_size=2048,
        # CNN feature extractor
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 1, 1, 1, 1, 1),  # effective downsample ≈ 10×
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        # Projection
        feat_proj_dropout=0.0,
        # Transformer
        hidden_dropout=0.1,
        attention_dropout=0.1,
        final_dropout=0.1,
        # CTC head
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=0,
    )

    print("Initialising Wav2Vec2ForCTC from scratch …")
    model = Wav2Vec2ForCTC(config)
    model.config.ctc_loss_reduction = "mean"

    # ------------------------------------------------------------------
    # 2. Define and inject LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        # lm_head must be fully trainable: our vocab differs from any
        # pre-trained checkpoint.
        modules_to_save=["lm_head"],
    )

    print("Injecting LoRA adapters …")
    peft_model = get_peft_model(model, lora_config)

    # ------------------------------------------------------------------
    # 3. Unfreeze CNN layers
    # ------------------------------------------------------------------
    unfrozen: int = 0
    for name, param in peft_model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            param.requires_grad = True
            unfrozen += 1

    print(f"Unfrozen {unfrozen} CNN / feature-projection parameter tensors.")
    peft_model.print_trainable_parameters()

    return peft_model
