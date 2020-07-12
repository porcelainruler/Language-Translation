from torchtext.datasets import Multi30k
from torchtext.data import Field

import spacy
import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


# To get Pre-processed Parallel Dataset for (De - Eng)
def get_dataset(flag_tv: bool = True, flag_test: bool = False, flag_all: bool = False):
    if flag_test:
        return test_data
    elif flag_all:
        return train_data, valid_data, test_data

    return train_data, valid_data


# To Get Constructed Vocab
def get_vocab():
    return SRC, TRG
