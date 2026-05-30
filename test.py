"""
============================================================
Transformer 从零实现 - 测试脚本
============================================================
对每个核心模块进行单元测试，验证:
  1. 输入/输出形状正确性
  2. 前向传播不报错
  3. Mask 机制正确性
  4. 完整模型的端到端推理

用法:
  python test.py              # 运行所有模块测试
  python test.py --full       # 运行完整功能测试（包含训练验证）
============================================================
"""

import argparse
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
    print("\n[测试] InputEmbedding...")
    batch_size, seq_len, vocab_size, d_model = 4, 10, 100, 128

    embedding = InputEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = embedding(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")


def test_positional_encoding():
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
    print(f"  [OK] 注意力权重形状: {attn_weights.shape}, 每行和 = 1.0")


def test_multihead_attention():
    print("\n[测试] MultiHeadAttention...")
    batch_size, seq_len_q, seq_len_k, d_model, n_heads = 4, 10, 15, 128, 8

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)
    q = torch.randn(batch_size, seq_len_q, d_model)
    k = torch.randn(batch_size, seq_len_k, d_model)
    v = torch.randn(batch_size, seq_len_k, d_model)

    out = mha(q, k, v)
    assert out.shape == (batch_size, seq_len_q, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] Q: {q.shape}, K: {k.shape}, V: {v.shape} -> 输出: {out.shape}")


def test_feedforward():
    print("\n[测试] PositionWiseFeedForward...")
    batch_size, seq_len, d_model, d_ff = 4, 10, 128, 512

    ff = PositionWiseFeedForward(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    out = ff(x)

    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")


def test_encoder_layer():
    print("\n[测试] EncoderLayer...")
    batch_size, seq_len, d_model, n_heads, d_ff = 4, 10, 128, 8, 512

    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)

    out = encoder_layer(x)
    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")


def test_encoder():
    print("\n[测试] TransformerEncoder...")
    batch_size, seq_len, d_model, n_heads, d_ff, num_layers = 4, 10, 128, 8, 512, 3

    encoder = TransformerEncoder(num_layers, d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)

    out = encoder(x)
    assert out.shape == (batch_size, seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] 输入形状: {x.shape} -> 输出形状: {out.shape}")


def test_decoder_layer():
    print("\n[测试] DecoderLayer...")
    batch_size, src_seq_len, tgt_seq_len = 4, 10, 8
    d_model, n_heads, d_ff = 128, 8, 512

    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    enc_out = torch.randn(batch_size, src_seq_len, d_model)

    out = decoder_layer(x, enc_out)
    assert out.shape == (batch_size, tgt_seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] x: {x.shape}, enc_out: {enc_out.shape} -> 输出: {out.shape}")


def test_decoder():
    print("\n[测试] TransformerDecoder...")
    batch_size, src_seq_len, tgt_seq_len = 4, 10, 8
    d_model, n_heads, d_ff, num_layers = 128, 8, 512, 3

    decoder = TransformerDecoder(num_layers, d_model, n_heads, d_ff, dropout=0.1)
    x = torch.randn(batch_size, tgt_seq_len, d_model)
    enc_out = torch.randn(batch_size, src_seq_len, d_model)

    out = decoder(x, enc_out)
    assert out.shape == (batch_size, tgt_seq_len, d_model), f"形状错误: {out.shape}"
    print(f"  [OK] x: {x.shape}, enc_out: {enc_out.shape} -> 输出: {out.shape}")


def test_masks():
    print("\n[测试] Mask 机制...")

    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    pad_mask = create_padding_mask(seq, pad_idx=0)
    assert pad_mask.shape == (2, 1, 1, 5), f"padding mask 形状错误: {pad_mask.shape}"
    print(f"  [OK] Padding mask 形状: {pad_mask.shape}")

    sub_mask = create_subsequent_mask(4)
    assert sub_mask.shape == (1, 1, 4, 4), f"subsequent mask 形状错误: {sub_mask.shape}"
    print(f"  [OK] Subsequent mask 形状: {sub_mask.shape}")

    combined = create_combined_mask(seq)
    assert combined.shape == (2, 1, 5, 5), f"combined mask 形状错误: {combined.shape}"
    print(f"  [OK] Combined mask 形状: {combined.shape}")

    enc_mask = create_encoder_mask(seq)
    assert enc_mask.shape == (2, 1, 1, 5), f"encoder mask 形状错误: {enc_mask.shape}"
    print(f"  [OK] Encoder mask 形状: {enc_mask.shape}")

    cross_mask = create_decoder_cross_mask(seq)
    assert cross_mask.shape == (2, 1, 1, 5), f"cross mask 形状错误: {cross_mask.shape}"
    print(f"  [OK] Decoder cross mask 形状: {cross_mask.shape}")


def test_transformer_model():
    print("\n[测试] Transformer 完整模型...")
    batch_size, src_seq_len, tgt_seq_len = 4, 10, 8
    src_vocab_size, tgt_vocab_size, d_model = 100, 100, 128
    n_heads, num_layers, d_ff = 8, 2, 512

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_len=100,
        dropout=0.1,
    )

    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))

    logits = model(src, tgt)
    assert logits.shape == (batch_size, tgt_seq_len, tgt_vocab_size), f"形状错误: {logits.shape}"
    print(f"  [OK] src: {src.shape}, tgt: {tgt.shape} -> logits: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] 模型总参数量: {total_params:,}")

    generated = model.generate(src, max_len=20, bos_idx=1, eos_idx=2)
    assert generated.dim() == 2 and generated.size(0) == batch_size, f"generate 输出形状错误: {generated.shape}"
    print(f"  [OK] Generate 输出形状: {generated.shape}")


def test_dataset_and_dataloader():
    print("\n[测试] 数据集和 DataLoader...")
    dataset = CopyTaskDataset(num_samples=100, min_len=3, max_len=10, vocab_size=50)
    src, tgt = dataset[0]
    print(f"  [OK] CopyTaskDataset: src={src.shape}, tgt={tgt.shape}")

    dataloader = create_dataloader(
        dataset, batch_size=8, shuffle=True, pad_idx=SPECIAL_TOKENS["<pad>"]
    )
    batch = next(iter(dataloader))
    src_batch, tgt_batch = batch
    assert src_batch.dim() == 2 and tgt_batch.dim() == 2
    print(f"  [OK] DataLoader batch: src={src_batch.shape}, tgt={tgt_batch.shape}")


def test_full_features():
    print("\n" + "=" * 70)
    print("完整功能测试（含训练验证）")
    print("=" * 70)

    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    from create_test_data import create_test_data
    from src.translation_data import load_multi30k, create_dataloader as create_trans_dataloader, SPECIAL_TOKENS as T_SPECIAL
    from src.trainer import Trainer

    data_dir = "data/test_translation"
    create_test_data(data_dir)

    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = load_multi30k(data_dir)

    print(f"  英语词表大小: {len(src_vocab)}")
    print(f"  德语词表大小: {len(tgt_vocab)}")
    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        max_len=50,
        dropout=0.1,
        pad_idx=T_SPECIAL["<pad>"],
    )

    train_loader = create_trans_dataloader(train_dataset, batch_size=8, shuffle=True, pad_idx=T_SPECIAL["<pad>"])
    val_loader = create_trans_dataloader(val_dataset, batch_size=8, shuffle=False, pad_idx=T_SPECIAL["<pad>"])
    test_loader = create_trans_dataloader(test_dataset, batch_size=8, shuffle=False, pad_idx=T_SPECIAL["<pad>"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=1e-3,
        num_epochs=3,
        device=device,
        pad_idx=T_SPECIAL["<pad>"],
        warmup_steps=100,
        log_interval=10,
    )

    print("  开始训练 (3 epochs)...")
    trainer.train()

    print("  运行测试...")
    test_loss, test_acc, test_ppl = trainer.test()
    print(f"  测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}, 测试困惑度: {test_ppl:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    trainer.save_model("checkpoints/checkpoint_test.pth")
    trainer.export_history("results/history_test.json")
    trainer.plot_curves("figures/curves_test.png")
    print("  [OK] 模型和训练历史已保存")

    from src.experiment_utils import count_parameters
    total_params, params_by_module = count_parameters(model)
    print(f"\n  模型总参数量: {total_params:,}")

    print("\n  [OK] 完整功能测试通过!")


def main():
    parser = argparse.ArgumentParser(description="Transformer 模块测试")
    parser.add_argument("--full", action="store_true", help="运行完整功能测试（包含训练验证）")
    args = parser.parse_args()

    print("=" * 70)
    print("Transformer 模块单元测试")
    print("=" * 70)

    tests = [
        ("词嵌入层", test_input_embedding),
        ("位置编码", test_positional_encoding),
        ("缩放点积注意力", test_scaled_dot_product_attention),
        ("多头注意力", test_multihead_attention),
        ("前馈网络", test_feedforward),
        ("编码器层", test_encoder_layer),
        ("完整编码器", test_encoder),
        ("解码器层", test_decoder_layer),
        ("完整解码器", test_decoder),
        ("Mask 机制", test_masks),
        ("Transformer 模型", test_transformer_model),
        ("数据集和 DataLoader", test_dataset_and_dataloader),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    print("=" * 70)

    if args.full:
        print("\n运行完整功能测试...")
        test_full_features()


if __name__ == "__main__":
    main()