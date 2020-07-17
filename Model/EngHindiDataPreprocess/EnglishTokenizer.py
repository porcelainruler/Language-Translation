import nltk
import codecs


class EnglishTokenizer:
    def __init__(self, sent: str = ''):
        self.text = ''
        self.tokens = list()
        self.sent_list = sent.split('\n')

    def read_from_file(self, path):
        file = codecs.open(path, encoding='utf-8', errors='ignore').read()
        self.sent_list = file.split('\n')

        return self.sent_list

    def sent_clean(self, src: str) -> str:
        # Data Specific Corrections:
        src = src.replace(' _ ', '')
        src = src.replace('_ ', '')
        src = src.replace(' - ', ' ')
        src = src.replace('...', ' ')

        # Numeric and {12th --> 1, 2, th}
        src = src.replace('0th', '0th ')
        src = src.replace('1th', '1th ')
        src = src.replace('2th', '2th ')
        src = src.replace('3th', '3th ')
        src = src.replace('4th', '4th ')
        src = src.replace('5th', '5th ')
        src = src.replace('6th', '6th ')
        src = src.replace('7th', '7th ')
        src = src.replace('8th', '8th ')
        src = src.replace('9th', '9th ')

        # To be Checked for Size and Accuracy    --Temporary Removed
        # src = src.replace('0', '0 ')
        # src = src.replace('1', '1 ')
        # src = src.replace('2', '2 ')
        # src = src.replace('3', '3 ')
        # src = src.replace('4', '4 ')
        # src = src.replace('5', '5 ')
        # src = src.replace('6', '6 ')
        # src = src.replace('7', '7 ')
        # src = src.replace('8', '8 ')
        # src = src.replace('9', '9 ')

        # Punctuation Replace
        src = src.replace('\\', ' ')
        src = src.replace('/', ' ')
        src = src.replace('|', ' ')
        src = src.replace(',', ' ')
        src = src.replace('\"', ' ')
        src = src.replace('(', ' ')
        src = src.replace(')', ' ')
        src = src.replace(':', ' ')
        src = src.replace("‘‘", ' ')
        src = src.replace("''", ' ')
        src = src.replace(".", ' ')
        src = src.replace('-', ' ')
        src = src.replace('_', ' ')
        src = src.replace('%', ' ')
        src = src.replace('~', ' ')

        '''
        # Preposition Merge case ofyou --> of, you
        prep_dict = ['of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'like',
                     'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under',
                     'around', 'among']

        for prep in prep_dict:
            src = src.replace(f' {prep}', f' {prep} ')
            src = src.replace(f'{prep} ', f' {prep} ')
        '''

        # Contractions Case
        # Not
        src = src.replace('n\'t', ' not')

        # Pronouns
        src = src.replace('\'m', ' am')
        src = src.replace('\'re', ' are')
        src = src.replace('\'ve', ' have')
        src = src.replace('\'s', ' is')
        src = src.replace('’m', ' am')
        src = src.replace('’re', ' are')
        src = src.replace('’ve', ' have')
        src = src.replace('’s', ' is')

        # Did, Would, Had
        did_dict = {'who\'d': 'who did', 'what\'d': 'what did', 'when\'d': 'when did', 'where\'d': 'where did',
                    'why\'d': 'why did', 'how\'d': 'how did', 'which\'d': 'which did'}
        for tok in did_dict.keys():
            src = src.replace(tok, did_dict[tok])
        src = src.replace('\'d', ' would')

        # Other Word
        other_word_dict = {'gimme': 'give me', '\'cause': 'because', 'cuz': 'because', 'finna': 'fixing to',
                           'imma': 'i am going to', 'gonna': 'going to', 'hafta': 'have to', 'woulda': 'would have',
                           'coulda': 'could have', 'shoulda': 'should have', 'ma\'am': 'madam', 'howdy': 'how do you',
                           'let\'s': 'let us', 'y\'ll': 'you all'}
        for tok in other_word_dict.keys():
            src = src.replace(tok, other_word_dict[tok])

        src = src.replace('\'', ' ')

        return src

    def tokenize(self) -> list:
        assert len(self.sent_list) != 0, 'Please Either provide Sentence to Class constructor while initialization or' \
                                         ' call read_from_file function'

        ff_check = codecs.open('checker_eng.txt', encoding='utf-8', mode='w')
        tok_list = list()
        for sent in self.sent_list:
            sent = sent.lower()

            ff_check.write(sent)
            ff_check.write('\n')

            sent = self.sent_clean(sent)
            tokens = ['<sos>'] + nltk.word_tokenize(sent) + ['<eos>']

            ff_check.write(' '.join(tokens))
            ff_check.write('\n')
            ff_check.write('\n')

            tok_list.append(tokens)

        self.tokens = tok_list
        return self.tokens


# Testing:
# source = "What'the/fuck is I'm this"
# tok = EnglishTokenizer(source)
# print(tok.tokenize())
# print(token)
# print(tok)
