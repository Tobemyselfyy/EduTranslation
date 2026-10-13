# src/dataloader.py
from torch.utils.data import DataLoader
from collate_fn import make_collate_fn


def make_loader(dataset, src_pad_id, tgt_pad_id, batch_size=8, shuffle=False):
    collate = make_collate_fn(src_pad_id, tgt_pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )