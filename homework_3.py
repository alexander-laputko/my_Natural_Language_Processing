import nltk, gensim
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import codecs
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import gutenberg

print(gutenberg.fileids())


# Word embedding training
def train_model(fileid):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(gutenberg.sents(fileid), min_count=5, size=300,
                                  workers=4, window=10, sg=1, negative=5, iter=10)


# Task 1: Train embeddings on 1 books from gutenberg
model = train_model('austen-emma.txt')

# Task 1: save the models on disk
model.save("austen-emma.model")  # binary format
model.wv.save_word2vec_format("austen-emma.vec", binary=False)  # text / vec format

# Task 2: Do a few searches on example words with .most_similiar() of gensim
print(model.most_similar('Emma'))
print()
print(model.most_similar('Miss'))
print()
print(model.most_similar('great'))
print()


# Task 3: Visualize the model
def viz(pca=True):
    wv, vocabulary = load_embeddings("austen-emma.vec")

    if pca:
        pca = PCA(n_components=2, whiten=True)
        Y = pca.fit(wv[:300, :]).transform(wv[:300, :])
    else:
        tsne = TSNE(n_components=2, random_state=0)
        Y = tsne.fit_transform(wv[:200, :])

    np.set_printoptions(suppress=True)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        if label.lower() not in stopwords.words('english'):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in if len(line.strip().split()) != 2])

        wv = np.loadtxt(wv)

    return wv, vocabulary


viz(pca=False)
viz(pca=True)


# Task 5: Try different settings, different window sizes (5,10) and see how the evaluation measure changes
def train_model2(fileid):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(gutenberg.sents(fileid), min_count=1, size=300,
                                  workers=4, window=10, sg=1, negative=5, iter=10)


model2 = train_model2('austen-emma.txt')

print(model2.most_similar('Emma'))


def train_model3(fileid):
    """
        training a gensim model, see also: https://radimrehurek.com/gensim/models/word2vec.html
    """
    # min-count: only include words in the model with a min-count
    return gensim.models.Word2Vec(gutenberg.sents(fileid), min_count=1, size=300,
                                  workers=4, window=5, sg=1, negative=5, iter=10)


model3 = train_model3('austen-emma.txt')

print(model3.most_similar('Emma'))
