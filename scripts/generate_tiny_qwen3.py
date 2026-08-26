import shutil
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "TinyQwen3"


def main() -> None:
    torch.manual_seed(42)

    config = Qwen3Config(
        vocab_size=1024,  # 词表大小

        hidden_size=128,  # 隐藏层维度
        intermediate_size=384,  # FFN 中间层维度

        num_hidden_layers=2,  # Transformer 隐藏层数量

        num_attention_heads=4,  # 注意力头数量
        num_key_value_heads=2,  # 键值头数量
        head_dim=32,  # 注意力头维度

        max_position_embeddings=256,  # 最大位置编码长度

        rms_norm_eps=1e-6,  # RMS 归一化 epsilon
        rope_theta=1_000_000.0,  # RoPE 旋转位置编码 base

        hidden_act="silu",  # 隐藏层激活函数 SiLU

        attention_bias=False,  # 注意力是否偏置
        attention_dropout=0.0,  # 注意力 dropout

        use_cache=True,  # 是否使用 cache

        bos_token_id=1,  # 开始 token id
        eos_token_id=2,  # 结束 token id

        tie_word_embeddings=True,  # 输入 embedding 和输出 lm_head 是否共享词嵌入
    )

    model = Qwen3ForCausalLM(config)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 以 safetensors 格式保存模型
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )

    print(model)
    print()
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
