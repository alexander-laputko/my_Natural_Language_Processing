import nltk
from nltk.book import text1
from nltk import sent_tokenize
from nltk import word_tokenize

f = open('158-0.txt', encoding='utf_8_sig')
raw = f.read()
print(type(raw))
print(len(raw))
print(raw[:100])
print()

tokens = word_tokenize(raw)
print(type(tokens))
print(len(tokens))
print(tokens[:10])
print()

sent_tokens = sent_tokenize(raw)
print(type(sent_tokens))
print(len(sent_tokens))
print(sent_tokens[:10])
print()

text = nltk.Text(tokens)

fdist1 = nltk.FreqDist(tokens)
print(fdist1)
print(fdist1.most_common(50))

fdist2 = nltk.FreqDist(text1)
print(fdist2)
print(fdist2.most_common(50))