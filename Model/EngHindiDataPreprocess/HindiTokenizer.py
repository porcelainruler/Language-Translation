# -*- coding: utf-8 -*-
import codecs
import re
from string import digits


class Tokenizer:
    '''class for tokenizer'''

    def __init__(self, text: str = None):
        if text is not None:
            self.text = text
            self.clean_text()
        else:
            self.text = None
        self.sentences = []
        self.tokens = []
        self.stemmed_word = []
        self.final_list = []
        self.final_tokens = []

    def read_from_file(self, filename):
        f = codecs.open(filename, encoding='utf-8')
        self.text = f.read()
        self.clean_text()
        return self.text

    def generate_sentences(self):
        '''generates a list of sentences'''
        text = self.text
        self.sentences = text.split('\n')

        return self.sentences

    def print_sentences(self, sentences=None):
        if sentences:
            for i in sentences:
                print(i.encode('utf-8'))
        else:
            for i in self.sentences:
                print(i.encode('utf-8'))

    def remove_integer(self):
        # using translate and digits
        # to remove numeric digits from string
        remove_digits = str.maketrans('', '', digits)
        res = self.text.translate(remove_digits)

        self.text = res

    def clean_text(self):
        '''not working'''
        text = self.text

        self.remove_integer()

        text = re.sub(r'(\d+)', r'', text)
        text = text.replace(',', ' ')
        text = text.replace('\"', ' ')
        text = text.replace('(', ' ')
        text = text.replace(')', ' ')
        text = text.replace('\"', ' ')
        text = text.replace(':', ' ')
        text = text.replace("\'", ' ')
        text = text.replace("‘‘", ' ')
        text = text.replace("’’", ' ')
        text = text.replace("''", ' ')
        text = text.replace(".", ' ')
        text = text.replace("|", ' ')
        text = text.replace("-", ' ')
        self.text = text

        return self.text

    def remove_only_space_words(self):

        tokens = filter(lambda tok: tok.strip(), self.tokens)
        tok_arr = list()
        for tok in tokens:
            tok_arr.append(tok)
        self.tokens = tok_arr

    def hyphenated_tokens(self):

        for each in self.tokens:
            if '-' in each:
                tok = each.split('-')
                self.tokens.remove(each)
                self.tokens.append(tok[0])
                self.tokens.append(tok[1])

    def tokenize(self):
        '''done'''
        if not self.sentences:
            self.generate_sentences()

        sentences_list = self.sentences
        tokens = []
        for each in sentences_list:
            each.replace('\r', '')
            word_list = each.split(' ')
            print(word_list)
            tokens.append(word_list)
        self.tokens = tokens
        # remove words containing spaces
        # self.remove_only_space_words()

        # remove hyphenated words
        # self.hyphenated_tokens()

        return self.tokens

    def print_tokens(self, print_list=None):
        '''done'''
        if print_list is None:
            for i in self.tokens:
                print(i.encode('utf-8'))
        else:
            for i in print_list:
                print(i.encode('utf-8'))

    def tokens_count(self):
        '''done'''
        return len(self.tokens)

    def sentence_count(self):
        '''done'''
        return len(self.sentences)

    def len_text(self):
        '''done'''
        return len(self.text)

    def concordance(self, word):
        '''done'''
        if not self.sentences:
            self.generate_sentences()
        sentence = self.sentences

        concordance_sent = []
        for each in sentence:
            each = each
            if word in each:
                concordance_sent.append(each)
        return concordance_sent

    def generate_freq_dict(self):
        '''done'''
        freq = {}
        if not self.tokens:
            self.tokenize()

        temp_tokens = self.tokens
        # doubt whether set can be used here or not
        for each in self.tokens:
            freq[each] = temp_tokens.count(each)

        return freq

    def print_freq_dict(self, freq):
        '''done'''
        for i in freq.keys():
            print(i.encode('utf-8'), ',', freq[i])

    def generate_stem_words(self, word):
        suffixes = {
            1: [u"ो", u"े", u"ू", u"ु", u"ी", u"ि", u"ा"],
            2: [u"कर", u"ाओ", u"िए", u"ाई", u"ाए", u"ने", u"नी", u"ना", u"ते", u"ीं", u"ती", u"ता", u"ाँ", u"ां", u"ों",
                u"ें"],
            3: [u"ाकर", u"ाइए", u"ाईं", u"ाया", u"ेगी", u"ेगा", u"ोगी", u"ोगे", u"ाने", u"ाना", u"ाते", u"ाती", u"ाता",
                u"तीं", u"ाओं", u"ाएं", u"ुओं", u"ुएं", u"ुआं"],
            4: [u"ाएगी", u"ाएगा", u"ाओगी", u"ाओगे", u"एंगी", u"ेंगी", u"एंगे", u"ेंगे", u"ूंगी", u"ूंगा", u"ातीं",
                u"नाओं", u"नाएं", u"ताओं", u"ताएं", u"ियाँ", u"ियों", u"ियां"],
            5: [u"ाएंगी", u"ाएंगे", u"ाऊंगी", u"ाऊंगा", u"ाइयाँ", u"ाइयों", u"ाइयां"],
        }
        for L in 5, 4, 3, 2, 1:
            if len(word) > L + 1:
                for suf in suffixes[L]:
                    # print type(suf),type(word),word,suf
                    if word.endswith(suf):
                        # print 'h'
                        return word[:-L]
        return word

    def generate_stem_dict(self):
        '''returns a dictionary of stem words for each token'''

        stem_word = {}
        if not self.tokens:
            self.tokenize()
        for each_token in self.tokens:
            temp = self.generate_stem_words(each_token)
            stem_word[each_token] = temp
            self.stemmed_word.append(temp)

        tokens = [i for i in self.stemmed_word]
        self.final_tokens = tokens
        return stem_word

    def remove_stop_words(self):
        f = codecs.open("rss.txt", encoding='utf-8')
        if not self.stemmed_word:
            self.generate_stem_dict()
        stopwords = [x.strip() for x in f.readlines()]
        tokens = [i for i in self.stemmed_word if unicode(i) not in stopwords]
        self.final_tokens = tokens
        return tokens


'''
if __name__ == "__main__":
    #t = Tokenizer(
    #    वाशिंगटन: दुनिया के सबसे शक्तिशाली देश के राष्ट्रपति बराक ओबामा ने प्रधानमंत्री नरेंद्र मोदी के संदर्भ में 'टाइम' पत्रिका में लिखा, "नरेंद्र मोदी ने अपने 
    #    बाल्यकाल में अपने परिवार की सहायता करने के लिए अपने पिता की चाय बेचने में मदद की थी। आज वह दुनिया के सबसे बड़े लोकतंत्र के नेता हैं और गरीबी 
    #    से प्रधानमंत्री तक की उनकी जिंदगी की कहानी भारत के उदय की गतिशीलता और क्षमता को परिलक्षित करती है।)
    # t=Tokenizer()
    # t.read_from_file('sample.txt')
    # print type(t.text)
    # y=clean(t.text)
    # print y
    sent = t.clean_text()
    print(sent)

    sent = t.generate_sentences()
    print(sent)
    sent = t.tokenize()
    print(sent)
    freq_dict = t.generate_freq_dict()
    print(freq_dict)
    s = t.concordance('बातों')
    f = t.generate_stem_dict()
    print(f)
    # for i in f.keys():
    # 	print i.encode('utf-8'),f[i].encode('utf-8')
    # z = t.remove_stop_words()
    t.print_tokens(t.final_tokens)
    print(t.sentence_count(), t.tokens_count(), t.len_text())
'''