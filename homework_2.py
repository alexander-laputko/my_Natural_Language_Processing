from collections import defaultdict

from nltk import sent_tokenize, word_tokenize


# Task 1: Select a book from gutenberg https://www.gutenberg.org/
def read_text(file):
    file = open(file, encoding='utf_8_sig')
    text = file.read()


# Task 2: Split your assigned gutenberg book into paragraphs -- we will treat these paragraphs as single documents
# in the remainder of the task
def split_on_paragraphs(file):
    SENT_COUNT = 20
    file = open(file, encoding='utf_8_sig')
    text = file.read()

    sentences = sent_tokenize(text)
    paragraphs = [' '.join(sentences[k:k + SENT_COUNT]) for k in range(0, len(sentences), SENT_COUNT)]

    # print(type(paragraphs))
    print('Have paragraphs in total: ', len(paragraphs))
    # show the first paragraph
    # print('First paragraph:\n', paragraphs[0])

    # Task 2: Create a positional index, with the paragraphs being the documents
    def index_paragraph(paragraph):
        result = defaultdict(list)
        for n, token in enumerate(word_tokenize(paragraph)):
            result[token.lower()].append(n)
        return result

    index = {f"paragraph_{k}": index_paragraph(p) for k, p in enumerate(paragraphs)}
    print(index['paragraph_0'])

    # Task 4: Implement simple search for 2 word phrase queries ( eg: “Arnold Schwarzenegger”)
    # check all word positions
    def search_in_paragraph(request: str, paragraph_key: str) -> list:
        tokens = [w.lower() for w in word_tokenize(request)]
        return [pos for pos in index[paragraph_key][tokens[0]]
                if search_recursive(paragraph_key, tokens, pos + 1, 1)]

    # Recursively iterate over words in request.
    def search_recursive(paragraph_key, tokens, pos, token_n):
        if token_n == len(tokens):
            return True
        # Check if the word is in the correct position.
        if tokens[token_n] in index[paragraph_key] and pos in index[paragraph_key][tokens[token_n]]:
            return search_recursive(paragraph_key, tokens, pos + 1, token_n + 1)

    # Create a dictionary with paragraphs as keys and positions of found phrases as values.
    def search(request: str) -> dict:
        results = {k: search_in_paragraph(request, k) for k in index.keys()}
        return {k: v for k, v in results.items() if v}

    # Task 5: Do some example searches to show that the positional index works
    print(search("Emma Woodhouse"))
    print(search("comfortable home"))


split_on_paragraphs('158-0.txt')