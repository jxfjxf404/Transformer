"""
============================================================
Transformer 从零实现 - 模块测试脚本
============================================================
对每个核心模块进行单元测试, 验证:
  1. 输入/输出形状正确性
  2. 前向传播不报错
  3. Mask 机制正确性
  4. 完整模型的端到端推理

用法:
  python test_modules.py
============================================================
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedding import InputEmbedding, PositionalEncoding
from src.attention import ScaledDotProductAttention, MultiHeadAttention
from src.feedforward import PositionWiseFeedForward
from src.encoder import EncoderLayer, TransformerEncoder
from src.decoder import DecoderLayer, TransformerDecoder
from src.transformer import Transformer
from src.mask import (
    create_padding_mask,
    create_subsequent_mask,
    create_combined_mask,
    create_encoder_mask,
    create_decoder_cross_mask,
)
from src.data import CopyTaskDataset, create_dataloader, SPECIAL_TOKENS


def test_input_embedding():
    """测试词嵌入层。"""
    print("\n[测试] InputEmbedding...")
    batch_size, seq_len, vocab_size, d_model = 4, 10, 100, 128

    embedding = InputEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = embedding(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")


def test_positional_encoding():
    """测试位置编码。"""
    print("\n[测试] PositionalEncoding...")
    batch_size, seq_len, d_model = 4, 20, 128

    pe = PositionalEncoding(d_model, max_len=100)
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"

    pe_buffer = pe.pe
    assert pe_buffer.shape == (1, 100, d_model), f"PE buffer 形状错误: {pe_buffer.shape}"

    assert torch.allclose(pe_buffer.max(), torch.tensor(1.0), atol=0.1), "PE 值域异常"
    assert torch.allclose(pe_buffer.min(), torch.tensor(-1.0), atol=0.1), "PE 值域异常"

    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")
    print(f"  [OK] PE buffer 形状: {pe_buffer.shape}, 值域: [{pe_buffer.min():.2f}, {pe_buffer.max():.2f}]")


def test_scaled_dot_product_attention():
    """测试缩放点积注意力。"""
    print("\n[测试] ScaledDotProductAttention...")
    batch_size, n_heads, seq_len_q, seq_len_k, d_k = 4, 8, 10, 15, 64

    attention_no_dropout = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(batch_size, n_heads, seq_len_q, d_k)
    k = torch.randn(batch_size, n_heads, seq_len_k, d_k)
    v = torch.randn(batch_size, n_heads, seq_len_k, d_k)

    out = attention_no_dropout(q, k, v)
    assert out.shape == (batch_size, n_heads, seq_len_q, d_k), f"形状错误: {out.shape}"
    print(f"  [OK] Q: {q.shape}, K: {k.shape}, V: {v.shape} -> 输出: {out.shape}")

    out_attn, attn_weights = attention_no_dropout(q, k, v, return_attention=True)
    assert attn_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size, n_heads, seq_len_q), atol=1e-5), (
        f"注意力权重和不为1, max_diff={((attn_weights.sum(dim=-1) - 1).abs().max().item()):.8f}"
    )
    print(f"  [OK] 注意力权重形状: {attn_weights.shape}, 每行和 = 1.0 (no dropout)")


def test_multihead_attention():
    """测试多头注意力。"""
    print("\n[测试] MultiHeadAttention...")
    batch_size, seq_len_q, seq_len_k, d_model, n_heads = 4, 10, 15, 128, 8

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_k, d_model)
    v = torch.randn(batch_size, seq_len_k, d_model)

    out = mha(q, k, v)
    assert out.shape == (batch_size, seq_len_q, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] Q: {q.shape}, K: {k.shape}, V: {v.shape} -> 输出: {out.shape}")

    out_attn, attn_weights = mha(q, k, v, return_attention=True)
    assert attn_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
    print(f"  [OK] 带注意力权重返回: {out_attn.shape}, 注意力权重: {attn_weights.shape}")


def test_feedforward():
    """测试前馈神经网络。"""
    print("\n[测试] PositionWiseFeedForward...")
    batch_size, seq_len, d_model, d_ff = 4, 10, 128, 512

    ff = PositionWiseFeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    out = ff(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入: {x.shape} -> 输出: {out.shape}")


def test_mask():
    """测试各种 mask 创建函数。"""
    print("\n[测试] Mask 机制...")

    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    pad_mask = create_padding_mask(seq, pad_idx=0)
    assert pad_mask.shape == (2, 1, 1, 5)
    assert pad_mask[0, 0, 0, 3].item() == True
    assert pad_mask[0, 0, 0, 0].item() == False
    print(f"  [OK] Padding mask 形状: {pad_mask.shape}")

    sub_mask = create_subsequent_mask(4)
    assert sub_mask.shape == (1, 1, 4, 4)
    assert sub_mask[0, 0, 1, 0].item() == False
    assert sub_mask[0, 0, 0, 1].item() == True
    print(f"  [OK] Subsequent mask 形状: {sub_mask.shape}")
    print(f"  [OK] 上三角掩码正确: 位置(1,0)可看={not sub_mask[0,0,1,0]}, 位置(0,1)可看={not sub_mask[0,0,0,1]}")

    combined = create_combined_mask(seq)
    assert combined.shape == (2, 1, 5, 5)
    print(f"  [OK] 组合 mask 形状: {combined.shape}")


def test_encoder_layer():
    """测试编码器单层。"""
    print("\n[测试] EncoderLayer...")
    batch_size, seq_len, d_model = 4, 10, 128

    layer = EncoderLayer(d_model, n_heads=8, d_ff=512)
    x = torch.randn(batch_size, seq_len, d_model)
    out = layer(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入: {x.shape} -> 输出: {out.shape}")

    mask = create_subsequent_mask(seq_len)
    out_masked = layer(x, mask)
    assert out_masked.shape == out.shape
    print(f"  [OK] 带 mask 的输出: {out_masked.shape}")


def test_encoder():
    """测试完整编码器。"""
    print("\n[测试] TransformerEncoder...")
    batch_size, seq_len, d_model = 4, 10, 128

    encoder = TransformerEncoder(num_layers=3, d_model=d_model, n_heads=8, d_ff=512)
    x = torch.randn(batch_size, seq_len, d_model)
    out = encoder(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入: {x.shape} -> 输出: {out.shape}")


def test_decoder_layer():
    """测试解码器单层。"""
    print("\n[测试] DecoderLayer...")
    batch_size, src_len, tgt_len, d_model = 4, 12, 10, 128

    layer = DecoderLayer(d_model, n_heads=8, d_ff=512)
    x = torch.randn(batch_size, tgt_len, d_model)
    enc_out = torch.randn(batch_size, src_len, d_model)
    out = layer(x, enc_out)

    assert out.shape == (batch_size, tgt_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入: {x.shape}, 编码器输出: {enc_out.shape} -> 输出: {out.shape}")


def test_decoder():
    """测试完整解码器。"""
    print("\n[测试] TransformerDecoder...")
    batch_size, src_len, tgt_len, d_model = 4, 12, 10, 128

    decoder = TransformerDecoder(num_layers=3, d_model=d_model, n_heads=8, d_ff=512)
    x = torch.randn(batch_size, tgt_len, d_model)
    enc_out = torch.randn(batch_size, src_len, d_model)
    out = decoder(x, enc_out)

    assert out.shape == (batch_size, tgt_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入: {x.shape}, 编码器输出: {enc_out.shape} -> 输出: {out.shape}")


def test_transformer():
    """测试完整 Transformer 模型。"""
    print("\n[测试] Transformer (完整模型)...")
    batch_size, src_len, tgt_len = 4, 10, 12
    vocab_size = 50
    d_model = 128

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        max_len=100,
        dropout=0.1,
    )

    src = torch.randint(4, vocab_size, (batch_size, src_len))
    tgt = torch.randint(4, vocab_size, (batch_size, tgt_len))

    logits = model(src, tgt)

    assert logits.shape == (batch_size, tgt_len, vocab_size), f"形状错误: {logits.shape}"
    print(f"  [OK] 源序列: {src.shape}, 目标序列: {tgt.shape} -> Logits: {logits.shape}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] 模型参数量: {n_params:,}")

    print("\n  [测试] 自回归生成...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            src, max_len=15, bos_idx=1, eos_idx=2
        )
    print(f"  [OK] 生成序列形状: {generated.shape}")
    assert generated.size(0) == batch_size
    assert generated.size(1) <= 15


def test_copy_dataset():
    """测试复制任务数据集。"""
    print("\n[测试] CopyTaskDataset...")
    dataset = CopyTaskDataset(num_samples=10, min_len=3, max_len=8, vocab_size=50)

    assert len(dataset) == 10
    src, tgt = dataset[0]
    assert src.dim() == 1
    assert tgt.dim() == 1
    assert tgt[0] == SPECIAL_TOKENS["<bos>"]
    assert tgt[-1] == SPECIAL_TOKENS["<eos>"]
    assert torch.equal(tgt[1:-1], src)
    print(f"  [OK] 数据集大小: {len(dataset)}, 源序列: {src.tolist()}, 目标序列: {tgt.tolist()}")

    loader = create_dataloader(dataset, batch_size=4)
    batch_src, batch_tgt = next(iter(loader))
    print(f"  [OK] 批次源序列: {batch_src.shape}, 批次目标序列: {batch_tgt.shape}")


def main():
    """运行所有测试。"""
    print("=" * 60)
    print("Transformer 模块单元测试")
    print("=" * 60)

    all_passed = True

    tests = [
        test_input_embedding,
        test_positional_encoding,
        test_scaled_dot_product_attention,
        test_multihead_attention,
        test_feedforward,
        test_mask,
        test_encoder_layer,
        test_encoder,
        test_decoder_layer,
        test_decoder,
        test_transformer,
        test_copy_dataset,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  [FAIL] 测试失败: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过! PASS")
    else:
        print("部分测试失败! FAIL")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())