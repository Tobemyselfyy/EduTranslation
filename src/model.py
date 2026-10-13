# src/model.py
import torch
import torch.nn as nn


class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_dim=64, hidden_dim=128, pad_id=0):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.src_emb(src)                   # [B, L_src, D]
        _, (h, c) = self.encoder(src_emb)             # h,c: [1, B, H]

        tgt_emb = self.tgt_emb(tgt)                   # [B, L_tgt, D]
        out, _ = self.decoder(tgt_emb, (h, c))        # [B, L_tgt, H]

        logits = self.fc(out)                         # [B, L_tgt, V]
        return logits
