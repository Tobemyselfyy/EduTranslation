# src/pretrain_plm.py
# 错位（排列）语言模型预训练脚本
#
# 作用：在正式翻译训练之前，只用「源语言」数据（train_ids.tsv 的 src 列，单语即可），
# 用错位语言模型的目标把编码器预训练一遍，得到更好的初始化权重。
# 之后 train.py 加载这个权重再在平行语料上微调，这就是「预训练 → 微调」范式。
#
# 错位语言模型的核心只有两步，都在 permute_shift 函数里：
#   1) 随机排列：把每条句子的 token 顺序随机打乱（「错位」指的是这一步）；
#   2) 平移一位：打乱后的序列整体错开一位，用前文自回归预测下一个 token
#      —— 这一步和 train.py 里 teacher forcing 的错开一位完全相同。

import torch
import torch.nn as nn
from pathlib import Path

from tokenizer import BPETokenizer
from Dataset import EduDataset
from DataLoader import make_loader

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRETRAIN_PATH = ROOT / "encoder_pretrain.pt"


class EncoderLM(nn.Module):
    """预训练用的「编码器语言模型」。结构与 model.py 的 LSTMSeq2Seq 里
    src_emb + encoder 两部分完全一致，属性名对齐，权重才能直接对接。"""
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, pad_id=0):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)   # 语言模型头，保存时被过滤掉

    def forward(self, x):
        # x: [B, L] 乱序后的源 token id
        out, _ = self.encoder(self.src_emb(x))   # [B, L, H]
        return self.fc(out)                      # [B, L, V]，每个位置预测下一个词


def permute_shift(src, mask):
    """错位语言模型的核心：1) 随机排列非 pad token；2) 平移一位。"""
    B, L = src.shape
    xp = src.clone()                     # 复制一份，不修改原始 batch
    for b in range(B):
        valid = src[b][mask[b]]          # 取出这条句子里非 pad 的真实 token
        # 只打乱真实 token，pad 留在末尾，后面 ignore_index 才会正确忽略它们
        xp[b, :len(valid)] = valid[torch.randperm(len(valid))]
    return xp[:, :-1], xp[:, 1:]         # 平移一位：输入 [B, L-1] / 目标 [B, L-1]


def main():
    # 1. 加载源端 tokenizer（预训练只看源语言）
    src_tok = BPETokenizer.load(PROC_DIR / "bpe_src.json")
    print("src vocab:", len(src_tok))

    # 2. 复用数据封装管线：EduDataset 读 train_ids.tsv（只用 src 列）
    ds = EduDataset(PROC_DIR / "train_ids.tsv")
    loader = make_loader(
        ds,
        src_pad_id=src_tok.pad_id(),
        tgt_pad_id=src_tok.pad_id(),     # tgt 列用不到，传什么都行
        batch_size=4,
        shuffle=True,
    )
    print("预训练样本:", len(ds), "batch 数:", len(loader))

    # 3. 预训练模型：编码器 + 语言模型头
    lm = EncoderLM(len(src_tok), pad_id=src_tok.pad_id())

    # 4. 损失与优化器：预测目标是「下一个 token」，pad 位置不算损失
    criterion = nn.CrossEntropyLoss(ignore_index=src_tok.pad_id())
    optimizer = torch.optim.Adam(lm.parameters(), lr=1e-3)

    # 5. 预训练循环：结构和 train.py 几乎一样，唯一区别是多调一次 permute_shift 做「错位」
    epochs = 20
    for epoch in range(epochs):
        lm.train()
        total_loss = 0
        for src, tgt, src_mask, tgt_mask in loader:
            perm_input, perm_target = permute_shift(src, src_mask)
            logits = lm(perm_input)              # [B, L-1, V]
            loss = criterion(logits.transpose(1, 2), perm_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch+1:3d}/{epochs}, loss = {avg:.4f}")

    # 6. 只保存编码器部分：过滤掉语言模型头 fc 的权重，
    #    剩下的键名 src_emb.* / encoder.* 正好与 LSTMSeq2Seq 对应
    enc_state = {
        k: v for k, v in lm.state_dict().items()
        if k.startswith(("src_emb", "encoder"))
    }
    torch.save(enc_state, PRETRAIN_PATH)
    print("预训练编码器已保存:", PRETRAIN_PATH)


if __name__ == "__main__":
    main()
