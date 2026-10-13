# src/dataset.py
from torch.utils.data import Dataset


def read_id_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src_str, tgt_str = line.split("\t")
            src_ids = [int(x) for x in src_str.split()]
            tgt_ids = [int(x) for x in tgt_str.split()]
            pairs.append((src_ids, tgt_ids))
    return pairs


class EduDataset(Dataset):
    def __init__(self, id_tsv_path, max_len=128):
        self.pairs = read_id_tsv(id_tsv_path)
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        src_ids, tgt_ids = self.pairs[i]
        return src_ids[:self.max_len], tgt_ids[:self.max_len]