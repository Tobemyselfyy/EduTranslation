# src/train.py
import torch
import torch.nn as nn
from pathlib import Path

from tokenizer import BPETokenizer
from Dataset import EduDataset
from DataLoader import make_loader
from model import LSTMSeq2Seq

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "model.pt"
PRETRAIN_PATH = ROOT / "encoder_pretrain.pt"   # 由 pretrain_plm.py 生成


def main():
    # 1. 加载 tokenizer
    src_tok = BPETokenizer.load(PROC_DIR / "bpe_src.json")
    tgt_tok = BPETokenizer.load(PROC_DIR / "bpe_tgt.json")
    print("src vocab:", len(src_tok), "tgt vocab:", len(tgt_tok))

    # 2. Dataset + DataLoader
    ds = EduDataset(PROC_DIR / "train_ids.tsv")
    loader = make_loader(
        ds,
        src_pad_id=src_tok.pad_id(),
        tgt_pad_id=tgt_tok.pad_id(),
        batch_size=4,
        shuffle=True,
    )
    print("训练样本:", len(ds), "batch 数:", len(loader))

    # 3. 模型
    model = LSTMSeq2Seq(len(src_tok), len(tgt_tok), pad_id=tgt_tok.pad_id())

    # 3.5 用错位语言模型预训练好的编码器初始化（可选）
    # pretrain_plm.py 保存的权重只含 src_emb 和 encoder 两部分，
    # strict=False 表示模型其余部分（tgt_emb、decoder、fc）保持随机初始化。
    # 之后训练循环完全不变——接下来的训练就是对预训练编码器的「微调」
    if PRETRAIN_PATH.exists():
        model.load_state_dict(torch.load(PRETRAIN_PATH), strict=False)
        print("已加载 PLM 预训练编码器:", PRETRAIN_PATH)

    # 4. 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. 训练循环
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for src, tgt, src_mask, tgt_mask in loader:
            # Teacher forcing：输入去掉最后一位，标签去掉第一位
            # tgt 本来是 [bos, w1, w2, ..., wn, eos]
            # decoder_input = [bos, w1, ..., wn]
            # target        = [w1, w2, ..., wn, eos]
            decoder_input = tgt[:, :-1]
            target = tgt[:, 1:]

            logits = model(src, decoder_input)          # [B, L-1, V]
            loss = criterion(logits.transpose(1, 2), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"epoch {epoch+1:3d}/{epochs}, loss = {avg:.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print("模型已保存:", MODEL_PATH)


if __name__ == "__main__":
    main()