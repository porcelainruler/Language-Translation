import codecs

from indicnlp.tokenize import sentence_tokenize, indic_tokenize

file_read = ''

sentences = list()
tokens = list()


def get_sentences(filepath: str = ''):
    global file_read, sentences
    if filepath != '':
        file_read = codecs.open(filename=filepath, encoding='utf-8').read()
        file_read = file_read.replace('\u200d', ' ')
    else:
        file_read = codecs.open(filename='mlc_train.hi-en.hi', encoding='utf-8').read()
        file_read = file_read.replace('\u200d', ' ')

    # sentences = sentence_tokenize.sentence_split(file_read, lang='hi')
    sentences = file_read.split('\n')

    return sentences


def get_token(filepath: str = ''):
    global tokens

    if not sentences:
        get_sentences(filepath)

    tok_list = list()
    for sentence in sentences:
        sentence = sentence.replace('\u200d', ' ')
        sentence = sentence.replace('\n', ' ')
        toks = ['<sos>'] + indic_tokenize.trivial_tokenize(sentence, lang='hi') + ['<eos>']
        tok_list.append(toks)

    tokens = tok_list
    return tokens


# Testing
# print(get_sentences()[:20])
# print(get_token()[:20])
