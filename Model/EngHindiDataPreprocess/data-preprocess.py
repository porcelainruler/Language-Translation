import gzip
import codecs
import math
from Model.EngHindiDataPreprocess import config
from Model.EngHindiDataPreprocess import HindiTokenizer as HIN_Tok


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
            tokens = ['<sos>'] + line.split(' ') + ['<eos>']
        else:
            tokens = line.split(' ')

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

count = 4


def create_hindi_vocab(data: list):
    global count

    for arr in data:
        for token in arr:
            if token in hin_vocab_textToInt.keys():
                continue
            else:
                hin_vocab_textToInt[token] = count
                hin_vocab_intToText[count] = token
                count += 1

    return hin_vocab_textToInt, hin_vocab_intToText


def create_eng_vocab(data: list):
    global count

    for arr in data:
        for token in arr:
            if token in eng_vocab_textToInt.keys():
                continue
            else:
                eng_vocab_textToInt[token] = count
                eng_vocab_intToText[count] = token
                count += 1

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
'''
hin_read = load_data_sp('IITB.en-hi.hi')
HIN_TOKEN_FORM = tokenizer(hin_read, flag=True, max_length=10000)
create_hindi_vocab(HIN_TOKEN_FORM)
'''
HIN_T = HIN_Tok.Tokenizer()
text = HIN_T.read_from_file(filename='test_hin.hi')
HIN_TOKEN_FORM = HIN_T.tokenize()

print(HIN_TOKEN_FORM)

exit(0)

eng_read = load_data_sp('IITB.en-hi.en')
ENG_TOKEN_FORM = tokenizer(eng_read, flag=True, max_length=10000)
create_eng_vocab(ENG_TOKEN_FORM)


print('English Vocab:', eng_vocab_textToInt)
print('Hindi Vocab:', hin_vocab_textToInt)

ENG_DATA = convert_seq_to_int(ENG_TOKEN_FORM, flag=True)
HIN_DATA = convert_seq_to_int(HIN_TOKEN_FORM, flag=False)

print('English Vocab Size:', max(eng_vocab_intToText.keys()))
print('Hindi Vocab Size:', max(hin_vocab_intToText.keys()))

print(ENG_DATA)
print(HIN_DATA)


