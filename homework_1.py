import nltk
from nltk.book import text1
from nltk import sent_tokenize
from nltk import word_tokenize


# Open and read the text
def read_text(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    print(type(raw))
    print(len(raw))
    print(raw[:100])
    print()


# Word segmentation using word_tokenize()
def tokenize_word(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)
    print(type(tokens))
    print(len(tokens))
    print(tokens[:10])
    print()


# Sentence segmentation using sent_tokenize()
def tokenize_sent(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    sent_tokens = sent_tokenize(raw)
    print(type(sent_tokens))
    print(len(sent_tokens))
    print(sent_tokens[:10])
    print()


# Convert to a nltk Text (text = nltk.Text(tokens))
def convert_to_nltk(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)
    text = nltk.Text(tokens)
    print(text)
    print()


# Use nltk.FreqDist() to print the most common words in book and “Moby Dick”(text1)
def freq_dist(file):
    file = open(file, encoding='utf_8_sig')
    raw = file.read()
    tokens = word_tokenize(raw)

    freq_words1 = nltk.FreqDist(tokens)
    print(freq_words1)
    print(freq_words1.most_common(50))

    freq_words2 = nltk.FreqDist(text1)
    print(freq_words2)
    print(freq_words2.most_common(50))


# Call the functions
read_text('158-0.txt')
tokenize_word('158-0.txt')
tokenize_sent('158-0.txt')
convert_to_nltk('158-0.txt')
freq_dist('158-0.txt')
