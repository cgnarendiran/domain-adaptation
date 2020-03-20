import os
from io import open
import torch


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):

        # Build Bert Tokenizer:
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained("bert-base-cased", cache_dir=path)
        self.dictionary = tokenizer.vocab
        # self.dictionary = Dictionary()
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        # self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))
        with open(os.path.join(path, 'train.txt'), 'r', encoding="utf8") as f:
            train_data = f.read()
            self.train = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_data)))
        with open(os.path.join(path, 'train.txt'), 'r', encoding="utf8") as f:
            valid_data = f.read()
            self.valid = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(valid_data)))

    # def tokenize(self, path):
    #     """Tokenizes a text file."""
    #     assert os.path.exists(path)
    #     # Add words to the dictionary
    #     with open(path, 'r', encoding="utf8") as f:
    #         for line in f:
    #             words = line.split() + ['<eos>']
    #             for word in words:
    #                 self.dictionary.add_word(word)

    #     # Tokenize file content
    #     with open(path, 'r', encoding="utf8") as f:
    #         idss = []
    #         for line in f:
    #             words = line.split() + ['<eos>']
    #             ids = []
    #             for word in words:
    #                 ids.append(self.dictionary.word2idx[word])
    #             idss.append(torch.tensor(ids).type(torch.int64))
    #         ids = torch.cat(idss)

    #     return ids
