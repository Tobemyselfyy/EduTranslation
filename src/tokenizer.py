# src/tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import BertPreTokenizer

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"


class BPETokenizer:
    def __init__(self):
        self.tok = Tokenizer(BPE(unk_token=UNK))
        self.tok.pre_tokenizer = BertPreTokenizer()

    def train(self, texts, vocab_size=200):   # 词表上限；实际大小取决于语料量
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[PAD, UNK, BOS, EOS],
        )
        self.tok.train_from_iterator(texts, trainer=trainer)

    def encode(self, text, bos=False, eos=False):
        ids = self.tok.encode(text).ids
        if bos:
            ids = [self.tok.token_to_id(BOS)] + ids
        if eos:
            ids = ids + [self.tok.token_to_id(EOS)]
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def pad_id(self):
        return self.tok.token_to_id(PAD)

    def bos_id(self):
        return self.tok.token_to_id(BOS)

    def eos_id(self):
        return self.tok.token_to_id(EOS)

    def save(self, path):
        self.tok.save(str(path))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.tok = Tokenizer.from_file(str(path))
        return obj

    def __len__(self):
        return self.tok.get_vocab_size()