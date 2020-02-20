import nltk
from nltk.book import text1
from nltk import sent_tokenize
from nltk import word_tokenize


def read_text(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    print(type(raw))
    print(len(raw))
    print(raw[:100])
    print()


def tokenize_word(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)
    print(type(tokens))
    print(len(tokens))
    print(tokens[:10])
    print()


def tokenize_sent(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    sent_tokens = sent_tokenize(raw)
    print(type(sent_tokens))
    print(len(sent_tokens))
    print(sent_tokens[:10])
    print()


def convert_to_nltk(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)
    text = nltk.Text(tokens)
    print(text)
    print()


def freq_dist(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)
    fdist1 = nltk.FreqDist(tokens)
    print(fdist1)
    print(fdist1.most_common(50))

    fdist2 = nltk.FreqDist(text1)
    print(fdist2)
    print(fdist2.most_common(50))


read_text('158-0.txt')
tokenize_word('158-0.txt')
tokenize_sent('158-0.txt')
convert_to_nltk('158-0.txt')
freq_dist('158-0.txt')



