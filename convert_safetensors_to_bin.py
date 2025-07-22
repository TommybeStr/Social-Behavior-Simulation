#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoConfig

# ===== 配置区 =====
# 原始 safetensors 分片所在目录
ORIG_DIR = Path("/home/zss/Social_Behavior_Simulation/Qwen2.5-3B-Instruct")
# 导出目录
EXPORT_DIR = ORIG_DIR / "hf_export"
EXPORT_DIR.mkdir(exist_ok=True)
# ===================

def main():
    # 1. 读取原始 config.json
    config = AutoConfig.from_pretrained(ORIG_DIR)
    config.save_pretrained(EXPORT_DIR)

    # 2. 合并所有 safetensors 分片
    state_dict = {}
    for shard in sorted(ORIG_DIR.glob("*.safetensors")):
        print(f"Loading shard: {shard.name}")
        sd = load_file(str(shard), device="cpu")
        state_dict.update(sd)
    print(f"Total tensors loaded: {len(state_dict)}")

    # 3. 保存为 pytorch_model.bin
    bin_path = EXPORT_DIR / "pytorch_model.bin"
    torch.save(state_dict, bin_path)
    print(f"Saved merged model to: {bin_path}")

    # 4. 拷贝 tokenizer 文件（根据你的目录结构做调整）
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = ORIG_DIR / fname
        if src.exists():
            dst = EXPORT_DIR / fname
            dst.write_bytes(src.read_bytes())
            print(f"Copied {fname} to export dir.")

    print(f"\n✅ 导出完成！请将聊天脚本中的 MODEL_DIR 指向：\n  {EXPORT_DIR}")

if __name__ == "__main__":
    main()
