# src/preprocess.py
import os
from pathlib import Path
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "train.tsv"
PROC_DIR = ROOT / "data" / "processed"

IDS_PATH = PROC_DIR / "train_ids.tsv"
SRC_TOK_PATH = PROC_DIR / "bpe_src.json"
TGT_TOK_PATH = PROC_DIR / "bpe_tgt.json"


def read_tsv(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line.split("\t")
            pairs.append((src, tgt))
    return pairs


def get_tokenizer(path, texts):
    if os.path.exists(path):
        return BPETokenizer.load(path)
    tok = BPETokenizer()
    tok.train(texts)
    tok.save(path)
    return tok


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_tsv(RAW_PATH)
    print("原始样本数:", len(pairs))

    src_tok = get_tokenizer(SRC_TOK_PATH, [p[0] for p in pairs])
    tgt_tok = get_tokenizer(TGT_TOK_PATH, [p[1] for p in pairs])
    print("src vocab:", len(src_tok), "tgt vocab:", len(tgt_tok))

    with open(IDS_PATH, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            src_ids = src_tok.encode(src, eos=True)#src 是完整输入，不需要开头标记
            tgt_ids = tgt_tok.encode(tgt, bos=True, eos=True)
            f.write(" ".join(map(str, src_ids)) + "\t" +
                    " ".join(map(str, tgt_ids)) + "\n")

    print("已写出:", IDS_PATH)


if __name__ == "__main__":
    main()