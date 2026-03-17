import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model, TaskType

# 1. 像之前一样初始化模型配置 (针对纳米孔词表大小: A, C, G, T + special tokens = 9)
config = Wav2Vec2Config(
    vocab_size=9, 
    hidden_size=768,
    num_hidden_layers=12,
    # ... 其他配置参考之前的回答 ...
    pad_token_id=0,
)

model = Wav2Vec2ForCTC(config) 

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, # 对于 CTC 模型通常不用严格指定 SEQ_CLS
    r=8,                                   # LoRA 的秩 (Rank)，通常 8 或 16 就足够
    lora_alpha=32,                         # 缩放因子 (通常是 r 的 2~4倍)
    target_modules=["q_proj", "v_proj"],   # 目标模块：通常作用于 Attention 层的 Query 和 Value 投影
    lora_dropout=0.05,
    bias="none",
    # 【非常关键】指定不被冻结的模块！
    # 因为我们的词汇表不同，必须让最后的线性映射层 (lm_head) 保持完全可训练
    modules_to_save=["lm_head"] 
)

# 3. 将 LoRA 注入模型
peft_model = get_peft_model(model, lora_config)

# 4. 【针对纳米孔的特殊处理】：解冻 CNN 特征提取器 (Feature Extractor)
# 如果你的基座模型没见过纳米孔信号，这一步绝对不能省略！
# 遍历模型参数，把 feature_extractor (CNN层) 的 requires_grad 设为 True
for name, param in peft_model.named_parameters():
    if "feature_extractor" in name or "feature_projection" in name:
        param.requires_grad = True

peft_model.print_trainable_parameters()