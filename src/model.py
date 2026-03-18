"""Model construction for the Nanopore DNA Sequencing Basecaller.

Architecture summary
--------------------
- Base model: ``Wav2Vec2ForCTC`` initialised from a PRE-TRAINED checkpoint.
- The Transformer layers retain pre-trained weights.
- The CNN feature extractor is overridden with custom strides for Nanopore
  and is randomly initialized (and trained end-to-end).
- The classification head is re-initialized for our 9-token vocabulary.
- Parameter-efficient fine-tuning: LoRA is injected into the Transformer
  attention layers (``q_proj`` / ``v_proj``).
- The CNN (``feature_extractor``, ``feature_projection``) and ``lm_head``
  are listed in ``modules_to_save`` so their trained weights are included
  in the saved LoRA adapter checkpoint.
"""
from __future__ import annotations

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel  # re-exported for convenience
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 9          # <pad>, <s>, </s>, <unk>, A, C, G, T, <blank>
BLANK_TOKEN_ID: int = 8      # index of <blank> in VOCAB

# ---------------------------------------------------------------------------
# Base model builder (shared between training and inference)
# ---------------------------------------------------------------------------


def build_nanopore_base_model(
    pretrained_model_name: str = "facebook/wav2vec2-base",
) -> Wav2Vec2ForCTC:
    """Build the Nanopore-adapted base ``Wav2Vec2ForCTC`` from a pre-trained checkpoint.

    This function is the single source of truth for the model architecture.
    It is called both during training (before LoRA injection) and during
    inference (to reconstruct the base model before loading the saved adapter).

    Changes applied on top of the pre-trained checkpoint:

    * Vocabulary size is set to ``VOCAB_SIZE`` (9 tokens: pad, bos, eos, unk,
      A, C, G, T, blank) and the ``lm_head`` is re-initialized.
    * The CNN feature extractor is replaced with a 3-layer stack
      (``conv_dim=(512,512,512)``, ``conv_stride=(5,2,1)``,
      ``conv_kernel=(10,3,3)``) giving ~10× downsampling, appropriate for
      Nanopore signals sampled at ~4 kHz.
    * ``ignore_mismatched_sizes=True`` ensures mismatched layers (CNN and
      ``lm_head``) are randomly re-initialized while pre-trained Transformer
      weights are loaded unchanged.

    Args:
        pretrained_model_name: HuggingFace model id or local path of the
            pre-trained Wav2Vec2 checkpoint (default: ``"facebook/wav2vec2-base"``).

    Returns:
        A ``Wav2Vec2ForCTC`` instance with pre-trained Transformer weights and
        randomly-initialized CNN / ``lm_head`` layers.
    """
    print(f"Loading configuration from {pretrained_model_name}...")
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)

    # Overwrite vocabulary
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = BLANK_TOKEN_ID
    config.ctc_loss_reduction = "mean"
    config.ctc_zero_infinity = True

    # Overwrite CNN architecture for Nanopore physics (~10x downsample).
    # Speech models use 7 CNN layers (320x downsample), which is too coarse
    # for DNA signals.  At a 4 kHz Nanopore sample rate and a median base
    # dwell of ~4–10 samples per base, a ~10x downsample yields roughly one
    # output frame per base, matching the CTC alignment requirements.
    
    config.conv_dim = (512, 512, 512)
    config.conv_stride = (5, 2, 1)
    config.conv_kernel = (10, 3, 3)
    config.num_feat_extract_layers = len(config.conv_dim)
    # config.conv_dim = (64, 64, 128, 128, 256, 256, 512)
    # config.conv_stride = (1, 1, 3, 3, 2, 2, 2)
    # config.conv_kernel = (5, 5, 5, 5, 5, 5, 5)
    # config.num_feat_extract_layers = len(config.conv_dim)

    print(f"Loading pre-trained weights from {pretrained_model_name}...")
    # ignore_mismatched_sizes=True loads matching Transformer weights and
    # randomly initializes the CNN and lm_head (size mismatch → new init).
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model


# ---------------------------------------------------------------------------
# LoRA model factory
# ---------------------------------------------------------------------------


def get_nanopore_lora_model(
    pretrained_model_name: str = "facebook/wav2vec2-base-960h",
) -> "PeftModel":
    """Build a LoRA-wrapped ``Wav2Vec2ForCTC`` for Nanopore basecalling.

    Calls :func:`build_nanopore_base_model` and then:

    1. Injects LoRA adapters into the Transformer attention projections
       (``q_proj`` / ``v_proj``).
    2. Lists the CNN (``feature_extractor``, ``feature_projection``) and
       ``lm_head`` in ``modules_to_save`` so that their trained weights are
       persisted in the adapter checkpoint alongside the LoRA deltas.

    Args:
        pretrained_model_name: Passed through to :func:`build_nanopore_base_model`.

    Returns:
        A PEFT-wrapped model ready for fine-tuning.
    """
    model = build_nanopore_base_model(pretrained_model_name)
    target_modules = []
    for i in range(12):
        target_modules.extend([
            f"wav2vec2.encoder.layers.{i}.attention.k_proj",
            f"wav2vec2.encoder.layers.{i}.attention.v_proj",
            f"wav2vec2.encoder.layers.{i}.attention.q_proj",
            f"wav2vec2.encoder.layers.{i}.attention.out_proj",
            f"wav2vec2.encoder.layers.{i}.feed_forward.intermediate_dense",
            f"wav2vec2.encoder.layers.{i}.feed_forward.output_dense",
        ])
    # ------------------------------------------------------------------
    # Define and inject LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        # lm_head is newly initialized → must be fully trained and saved.
        # feature_extractor and feature_projection are newly initialized
        # (custom CNN strides) → must also be fully trained and saved.
        modules_to_save=["lm_head", "feature_extractor", "feature_projection"],
    )

    print("Injecting LoRA adapters ...")
    peft_model = get_peft_model(model, lora_config)

    peft_model.print_trainable_parameters()
    return peft_model