"""Model construction for the Nanopore DNA Sequencing Basecaller.

Architecture summary
--------------------
- Base model: ``Wav2Vec2ForCTC`` initialised from a PRE-TRAINED checkpoint.
- The Transformer layers retain pre-trained weights.
- The CNN feature extractor is overridden with custom strides for Nanopore 
  and is randomly initialized (and left unfrozen).
- The classification head is re-initialized for our 9-token vocabulary.
- Parameter-efficient fine-tuning: LoRA is injected into the Transformer
  attention layers (``q_proj`` / ``v_proj``).
"""
from __future__ import annotations

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel  # re-exported for convenience
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
import logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 9          # <pad>, <s>, </s>, <unk>, A, C, G, T, <blank>
BLANK_TOKEN_ID: int = 8      # index of <blank> in VOCAB

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_nanopore_lora_model(pretrained_model_name: str = "facebook/wav2vec2-base") -> "PeftModel":
    """Build a LoRA-wrapped Wav2Vec2ForCTC model loaded from pre-trained weights.
    """
    
    # ------------------------------------------------------------------
    # 1. Load Pre-trained Config and Adapt for Nanopore
    # ------------------------------------------------------------------
    print(f"Loading configuration from {pretrained_model_name}...")
    config = Wav2Vec2Config.from_pretrained(pretrained_model_name)
    
    # Overwrite Vocabulary
    config.vocab_size = VOCAB_SIZE
    config.pad_token_id = 0
    config.ctc_loss_reduction = "mean"
    config.ctc_zero_infinity = True
    
    # Overwrite CNN architecture for Nanopore physics (~10x downsample)
    # Speech models use 7 CNN layers (320x downsample), which is too much for DNA.
    config.conv_dim = (512, 512, 512)
    config.conv_stride = (5, 2, 1)
    config.conv_kernel = (10, 3, 3)
    config.num_feat_extract_layers = len(config.conv_dim) # Ensure layer count matches
    
    # ------------------------------------------------------------------
    # 2. Load Pre-trained Weights (with mismatch handling)
    # ------------------------------------------------------------------
    print(f"Loading pre-trained weights from {pretrained_model_name}...")
    # NOTE: ignore_mismatched_sizes=True is the magic here.
    # It will print warnings that some weights (lm_head and conv layers) 
    # were not loaded and were newly initialized. THIS IS EXACTLY WHAT WE WANT.
    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    # ------------------------------------------------------------------
    # 3. Define and inject LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        # lm_head must be fully trainable since it's newly initialized
        modules_to_save=["lm_head"],
    )

    print("Injecting LoRA adapters ...")
    peft_model = get_peft_model(model, lora_config)

    # ------------------------------------------------------------------
    # 4. Unfreeze CNN layers
    # ------------------------------------------------------------------
    # Because the CNN was newly initialized (due to our custom strides),
    # it MUST be trained. If we leave it frozen, the model will output garbage.
    unfrozen: int = 0
    for name, param in peft_model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            param.requires_grad = True
            unfrozen += 1

    print(f"Unfrozen {unfrozen} CNN / feature-projection parameter tensors.")
    peft_model.print_trainable_parameters()

    return peft_model