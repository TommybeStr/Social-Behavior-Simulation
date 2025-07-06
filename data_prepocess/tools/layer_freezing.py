import torch
from transformers import AutoModelForCausalLM

class FrozenQwenModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.partial_pretrain,
            trust_remote_code=False
        )
        # 开启梯度检查点
        model.gradient_checkpointing_enable()

        total = len(model.qwen2.decoder.layers)
        freeze_n = total // 2
        for idx, layer in enumerate(model.qwen2.decoder.layers):
            if idx < freeze_n:
                for name, p in layer.named_parameters():
                    # 只冻结权重，不冻结 LayerNorm、bias、lm_head
                    if ("layer_norm" in name) or ("bias" in name):
                        continue
                    p.requires_grad = False
        return model