import codecs
import math
from Model.EngHindiDataPreprocess import config, EnglishTokenizer as ENG_Tok, HindiTokenizer as HIN_TOK, \
    IndicTokenizer as IND_TOK
from nltk import word_tokenize
from Model.EngHindiDataPreprocess import config


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


print('Vocab Creation in Progress...')

# English Dataset Tokenization and Vocab Creation
ENG_TOKENIZER = ENG_Tok.EnglishTokenizer()
eng_read = ENG_TOKENIZER.read_from_file(path='mlc_train.hi-en.en')
ENG_TOKENS = ENG_TOKENIZER.tokenize()
create_eng_vocab(ENG_TOKENS)


# Hindi Dataset Tokenization and Vocab Creation
hin_read = IND_TOK.get_sentences(filepath='mlc_train.hi-en.hi')
HIN_TOKENS = IND_TOK.get_token(filepath='mlc_train.hi-en.hi')
create_hindi_vocab(HIN_TOKENS)

# Printing Vocab Size
# print('English Vocab Size:', count_eng, 'Hindi Vocab Size:', count_hin)

print('----------------------Vocab Creation Done for Both----------------------')


def vocab_creator(vocab_dict: dict, flag: bool):
    if flag:
        out = codecs.open('hindi_vocab.txt', encoding='utf-8', mode='w')
    else:
        out = codecs.open('english_vocab.txt', encoding='utf-8', mode='w')

    for key in vocab_dict.keys():
        out.write(f'{key}_:_{vocab_dict[key]}')
        out.write('\n')

    out.close()


# Vocab txt File Creation for both English and Hindi                                <For Vocab Creation in txt>
# vocab_creator(eng_vocab_textToInt, flag=False)
# vocab_creator(hin_vocab_textToInt, flag=True)

# Vocab Checker:
# print('English Vocab:', eng_vocab_textToInt)
# print('Hindi Vocab:', hin_vocab_textToInt)

print('Data Conversion to Integer in Progress...')

max_length = -math.inf


def max_length_updator(seq: list):
    global max_length

    for sent in seq:
        if len(sent) > max_length:
            max_length = len(sent)


def padding_seq(seq: list):
    global max_length

    new_seq = list()
    for idx in range(len(seq)):
        padding = [config.PAD_TOKEN_IDX]*int(max_length - len(seq[idx]))
        new_seq.append(seq[idx] + padding)

    return new_seq


# Sequence Tokens Convert to Integer Form
ENG_DATA = convert_seq_to_int(ENG_TOKENS, flag=True)
HIN_DATA = convert_seq_to_int(HIN_TOKENS, flag=False)

# Updating Max-Length for Dataset Padding
max_length_updator(ENG_DATA)
max_length_updator(HIN_DATA)

# Adding Padding to Dataset
ENG_DATA_PADDED = padding_seq(ENG_DATA)
HIN_DATA_PADDED = padding_seq(HIN_DATA)

print('Data Conversion to Integer Done...')

# Check for Correct Tokenization
# print(ENG_DATA[:20])
# print(HIN_DATA[:20])
