# src/main.py
from pathlib import Path
from tokenizer import BPETokenizer
from Dataset import EduDataset
from DataLoader import make_loader

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

IDS_PATH = PROC_DIR / "train_ids.tsv"
SRC_TOK_PATH = PROC_DIR / "bpe_src.json"
TGT_TOK_PATH = PROC_DIR / "bpe_tgt.json"


def main():
    ds = EduDataset(IDS_PATH)
    print("数据集大小:", len(ds))
    print("第一条样本:", ds[0])

    src_tok = BPETokenizer.load(SRC_TOK_PATH)
    tgt_tok = BPETokenizer.load(TGT_TOK_PATH)

    loader = make_loader(
        ds,
        src_pad_id=src_tok.pad_id(),
        tgt_pad_id=tgt_tok.pad_id(),
        batch_size=4,
        shuffle=True,
    )

    for src, tgt, src_mask, tgt_mask in loader:
        print("src shape:", src.shape)
        print("tgt shape:", tgt.shape)
        print("src_mask shape:", src_mask.shape)
        print("tgt_mask shape:", tgt_mask.shape)
        print()
        print("src[0]:", src[0].tolist())
        print("src_mask[0]:", src_mask[0].tolist())
        print("decode src[0]:", src_tok.decode(src[0].tolist()))
        break


if __name__ == "__main__":
    main()