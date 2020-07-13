import codecs
import math
from Model.EngHindiDataPreprocess import config
from nltk import word_tokenize


def load_data_sp(path):
    with codecs.open(path, encoding='utf-8') as f:
        data = f.read().split('\n')
        print('Num of Lines in Data:', len(data))

    return data


def tokenizer(data: list, flag: bool = True, max_length: int = math.inf):
    assert type(data) == list, 'Raw Data should be in list data type'
    print(max_length)

    token_list = list()
    for line in data:
        if flag:
            tokens = word_tokenize(line)
        else:
            tokens = ['<sos>'] + line.split(' ') + ['<eos>']

        token_list.append(tokens)

        if len(token_list) > max_length:
            break
    return token_list


# Vocab Code for Hindi
hin_vocab_intToText = dict()
hin_vocab_textToInt = dict()

# Vocab Code for English
eng_vocab_intToText = dict()
eng_vocab_textToInt = dict()

# Setting up some Special Token value for Hindi Vocab
hin_vocab_textToInt[config.SOS_TOKEN] = config.SOS_TOKEN_IDX
hin_vocab_textToInt[config.EOS_TOKEN] = config.EOS_TOKEN_IDX
hin_vocab_textToInt[config.PAD_TOKEN] = config.PAD_TOKEN_IDX
hin_vocab_textToInt[config.UNK_TOKEN] = config.UNK_TOKEN_IDX

hin_vocab_intToText[config.SOS_TOKEN_IDX] = config.SOS_TOKEN
hin_vocab_intToText[config.EOS_TOKEN_IDX] = config.EOS_TOKEN
hin_vocab_intToText[config.PAD_TOKEN_IDX] = config.PAD_TOKEN
hin_vocab_intToText[config.UNK_TOKEN_IDX] = config.UNK_TOKEN

# Setting up some Special Token value for Hindi Vocab
eng_vocab_textToInt[config.SOS_TOKEN] = config.SOS_TOKEN_IDX
eng_vocab_textToInt[config.EOS_TOKEN] = config.EOS_TOKEN_IDX
eng_vocab_textToInt[config.PAD_TOKEN] = config.PAD_TOKEN_IDX
eng_vocab_textToInt[config.UNK_TOKEN] = config.UNK_TOKEN_IDX

eng_vocab_intToText[config.SOS_TOKEN_IDX] = config.SOS_TOKEN
eng_vocab_intToText[config.EOS_TOKEN_IDX] = config.EOS_TOKEN
eng_vocab_intToText[config.PAD_TOKEN_IDX] = config.PAD_TOKEN
eng_vocab_intToText[config.UNK_TOKEN_IDX] = config.UNK_TOKEN

count_eng = 4
count_hin = 4


def create_hindi_vocab(data: list):
    global count_hin

    for arr in data:
        for token in arr:
            if token in hin_vocab_textToInt.keys():
                continue
            else:
                hin_vocab_textToInt[token] = count_hin
                hin_vocab_intToText[count_hin] = token
                count_hin += 1

    return hin_vocab_textToInt, hin_vocab_intToText


def create_eng_vocab(data: list):
    global count_eng

    for arr in data:
        for token in arr:
            if token in eng_vocab_textToInt.keys():
                continue
            else:
                eng_vocab_textToInt[token] = count_eng
                eng_vocab_intToText[count_eng] = token
                count_eng += 1

    return eng_vocab_textToInt, eng_vocab_intToText


def convert_seq_to_int(data: list, flag: bool):
    arr = list()
    for line in data:
        tok_line = list()
        for token in line:
            if flag:
                if token in eng_vocab_textToInt.keys():
                    tok_line.append(eng_vocab_textToInt[token])
                else:
                    tok_line.append(eng_vocab_textToInt['<unk>'])
            else:
                if token in hin_vocab_textToInt.keys():
                    tok_line.append(hin_vocab_textToInt[token])
                else:
                    tok_line.append(hin_vocab_textToInt['<unk>'])

        arr.append(tok_line)

    return arr


# hin_vocab_read = load_data_sp('monolingual.hi')
hin_read = load_data_sp('IITB.en-hi.hi')
HIN_TOKEN_FORM = tokenizer(hin_read, flag=False)
create_hindi_vocab(HIN_TOKEN_FORM)


eng_read = load_data_sp('IITB.en-hi.en')
ENG_TOKEN_FORM = tokenizer(eng_read, flag=True, max_length=100000)
create_eng_vocab(ENG_TOKEN_FORM)

print('Vocab Creation Done for Both')


def vocab_creator(vocab_dict: dict, flag: bool):
    if flag:
        out = codecs.open('hindi_vocab.txt', encoding='utf-8', mode='w')
    else:
        out = codecs.open('english_vocab.txt', encoding='utf-8', mode='w')

    for key in vocab_dict.keys():
        out.write(f'{key}_:_{vocab_dict[key]}')
        out.write('\n')

    out.close()


vocab_creator(hin_vocab_textToInt, flag=True)
vocab_creator(eng_vocab_textToInt, flag=False)

# print('English Vocab:', eng_vocab_textToInt)
# print('Hindi Vocab:', hin_vocab_textToInt)

# ENG_DATA = convert_seq_to_int(ENG_TOKEN_FORM, flag=True)
# HIN_DATA = convert_seq_to_int(HIN_TOKEN_FORM, flag=False)
