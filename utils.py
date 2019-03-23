import os
import re
import numpy as np

MAX_SEQUENCE_LENGTH = 113
EMBEDDING_DIM = 100 # 50 or 100 or 200 or 300
PAD_ID = 0
UNK_ID = 1


def word_preprocessor(word):
    word = re.sub(r'\d+', '1', re.sub(r"[-|.|,|\?|\!]+", '', word))
    word = word.lower()
    if word != '':
        return word
    else:
        return '.'

    
def load_data(path, word_preprocessor=word_preprocessor):
    tags = []
    words = []
    data = {'words': [], 'tags': []}
    with open(path) as f:
        for line in f.readlines()[2:]:
            if line != '\n':
                parts = line.replace('\n', '').split(' ')
                words.append(word_preprocessor(parts[0]))
                tags.append(parts[-1])
            else:
                data['words'].append(words)
                data['tags'].append(tags)
                words, tags = [], []

    return data


def make_vocab(sentences, tags=False):
    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    idd = max([PAD_ID, UNK_ID]) + 1
    for sen in sentences:
        for word in sen:
            if word not in vocab:
                vocab[word] = idd
                idd += 1
                
    return vocab


def make_sequences(list_of_words, vocab, word_preprocessor=None):
    sequences = []
    for words in list_of_words:
        seq = []
        for word in words:
            if word_preprocessor:
                word = word_preprocessor(word)
            seq.append(vocab.get(word, UNK_ID))
        sequences.append(seq)
    return sequences


def make_embedding_tensor(glova_path, words_vocab):
    """
        We use GloVe 6B 100d.
        You can download it from: https://nlp.stanford.edu/projects/glove/
    """
    embeddings_index = {}
    with open(os.path.join(glova_path, f"glove.6B.{EMBEDDING_DIM}d.txt")) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_tensor = np.zeros((len(words_vocab) + 1, EMBEDDING_DIM))
    for word, i in words_vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_tensor[i] = embedding_vector
    
    return embedding_tensor
