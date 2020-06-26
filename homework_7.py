import nltk
from nltk.corpus import brown
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Task1: Choose any dataset with sequence data -- eg. some given training datasets for part-of-speech tagging
# (for example brown corpus in NLTK) -- but you can also use some Russian language corpus or whatever you want.
nltk.download('brown')

data = []

# Converting our data to a convenient dimension.
for sent in brown.tagged_sents():
    words = []
    tags = []
    for word, tag in sent:
        words.append(word)
        tags.append(tag)
    data.append((words, tags))


# Create methods that transform the texts into numbers for feeding to the input of the neural network.
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


word_to_ix = {}
tag_to_ix = {}
for sent, tags in data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

# Train and test our data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# define pos tagger LSTM
# Create the model
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


EMBEDDING_DIM = 64
HIDDEN_DIM = 64
EPOCHS = 15

# train the classifier
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)).cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(EPOCHS):
    print('{} epoch of {}'.format(epoch + 1, EPOCHS))
    losses = []
    for sentence, tags in train_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix).cuda()
        targets = prepare_sequence(tags, tag_to_ix).cuda()

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print('Loss: ', sum(losses) / len(losses))

# 3. Evaluate the quality of the model. Before training split your dataset into a train and test part.
# And evaluate on the test part. Use a reasonable metric for evaluation!
with torch.no_grad():
    predicted_tags = []
    total_tags = 0
    correct_tags = 0
    for sentence, tags in test_data:
        # Get the data
        inputs = prepare_sequence(sentence, word_to_ix).cuda()
        tag_scores = model(inputs)
        targets = prepare_sequence(tags, tag_to_ix).cuda()
        predicted_tags = [ts.index(max(ts)) for ts in tag_scores.tolist()]
        total_tags += len(targets)
        for i, tag in enumerate(predicted_tags):
            if tag == targets[i]:
                correct_tags += 1
    print('Accuracy: ', correct_tags / total_tags)
