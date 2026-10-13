# src/translate.py
import torch
from pathlib import Path
from tokenizer import BPETokenizer
from model import LSTMSeq2Seq

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

SRC_TOK_PATH = PROC_DIR / "bpe_src.json"
TGT_TOK_PATH = PROC_DIR / "bpe_tgt.json"
MODEL_PATH = ROOT / "model.pt"


def greedy_decode(model, src_tok, tgt_tok, text, max_len=50):
    """
    贪心解码：每一步都选概率最大的 token，直到 <eos> 或达到 max_len。
    """
    model.eval()

    with torch.no_grad():
        # 1. 源句编码
        src_ids = src_tok.encode(text, eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long)   # [1, L_src]

        # 2. 从 <bos> 开始
        bos = tgt_tok.bos_id()
        eos = tgt_tok.eos_id()

        tgt_ids = [bos]

        # 3. 逐步生成
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long)   # [1, L_cur]
            logits = model(src_tensor, tgt_tensor)                    # [1, L_cur, V]

            # 取最后一个位置的预测
            next_id = logits[0, -1].argmax().item()

            if next_id == eos:
                break
            tgt_ids.append(next_id)

        return tgt_ids


def main():
    # 1. 加载 tokenizer
    src_tok = BPETokenizer.load(SRC_TOK_PATH)
    tgt_tok = BPETokenizer.load(TGT_TOK_PATH)

    # 2. 加载模型
    model = LSTMSeq2Seq(len(src_tok), len(tgt_tok), pad_id=tgt_tok.pad_id())
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH))
        print("已加载模型:", MODEL_PATH)
    else:
        print("模型未训练，输出会是乱码。先跑 train.py")

    # 3. 翻译一句话
    text = "什么是机器学习？"
    out_ids = greedy_decode(model, src_tok, tgt_tok, text)
    result = tgt_tok.decode(out_ids)
    print("源句:", text)
    print("翻译:", result)


if __name__ == "__main__":
    main()