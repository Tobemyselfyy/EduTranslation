# src/collate_fn.py
import torch
from torch.nn.utils.rnn import pad_sequence


def make_collate_fn(src_pad_id, tgt_pad_id):
    def collate(batch):
        src_list = [torch.tensor(x[0], dtype=torch.long) for x in batch]
        tgt_list = [torch.tensor(x[1], dtype=torch.long) for x in batch]

        src = pad_sequence(src_list, batch_first=True, padding_value=src_pad_id)
        tgt = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_id)

        src_mask = (src != src_pad_id)
        tgt_mask = (tgt != tgt_pad_id)

        return src, tgt, src_mask, tgt_mask

    return collate