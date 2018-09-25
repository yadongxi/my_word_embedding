import math
import sys
import numpy as np


class Corpus:
    def __init__(self, filename):
        all_tokens = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                tokens = line.split()
                for token in tokens:
                    token = token.lower()

                    if len(token) > 1 and token.isalnum():
                        all_tokens.append(token)
            self.tokens = all_tokens
            self.save_to_file(filename)

    def save_to_file(self, filename):
        with open('preprocessed_' + filename, 'w', encoding="utf-8") as f:
            line = ''
            i = 1
            for token in self.tokens:
                if i % 20 == 0:
                    line += token
                    f.write('{}\n'.format(line))
                    line = ''
                else:
                    line += token + ' '
                i += 1

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, corpus, min_count):
        self.words = []
        self.wtoi = {}
        self.min_count = min_count
        self.build_vocabulary(corpus)
        self.frequency_filter()

    def build_vocabulary(self, corpus):
        words = []
        wtoi = {}

        i = 0
        for token in corpus:
            if token not in wtoi:
                wtoi[token] = len(words)
                words.append(Word(token))
            words[wtoi[token]].count += 1

            i += 1
        self.words = words
        self.wtoi = wtoi  # word to index

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.wtoi

    def indices(self, tokens):
        return [self.wtoi[token] if token in self.wtoi else self.wtoi['UNK'] for token in tokens]

    def frequency_filter(self):
        tmp = []
        tmp.append(Word('UNK'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < self.min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda token: token.count, reverse=True)

        # Update wtoi
        wtoi = {}
        for i, token in enumerate(tmp):
            wtoi[token.word] = i

        self.words = tmp
        self.wtoi = wtoi


class TableForNegativeSamples:
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power)
                    for t in vocab])  # Normalizing constants

        table_size = 1e8
        table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0  # Cumulative probability
        i = 0
        for j, word in enumerate(vocab):
            p += float(math.pow(word.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, W0, filename):
    with open(filename, 'w', encoding="utf-8") as f:
        for token, vector in zip(vocab, W0):
            vector_str = ' '.join([str(s) for s in vector])
            file_pointer.write('{} {}\n'.format(word, vector_str))


if __name__ == '__main__':
    file_name = "train.txt"
    # Number of negative examples
    k_negative_sampling = 5
    #context window
    window = 5
    # emebdding_dim
    embedding_dim = 100
    # frequency freshhold
    min_count = 15
    # learning rate
    lr = 0.01
    # get corpus
    corpus = Corpus(file_name)
    # build vocabulary
    vocab = Vocabulary(corpus, min_count)
    table = TableForNegativeSamples(vocab)

    # Initialize weight
    W0 = np.random.uniform(low=-0.5/embedding_dim, high=0.5 /
                           embedding_dim, size=(len(vocab), embedding_dim))
    W1 = np.zeros(shape=(len(vocab), embedding_dim))

    global_word_count = 0
    word_count = 0
    last_word_count = 0

    tokens = vocab.indices(corpus)

    for token_idx, token in enumerate(tokens):
        if word_count % 10000 == 0:
            global_word_count += (word_count - last_word_count)
            last_word_count = word_count
            print("Training: {} of {}".format(global_word_count, len(corpus)))

        # Randomize window size, where win is the max window size
        current_window = np.random.randint(low=1, high=window+1)
        context_start = max(token_idx - current_window, 0)
        context_end = min(token_idx + current_window + 1, len(tokens))
        context = tokens[context_start:token_idx] + \
            tokens[token_idx+1:context_end]  # Turn into an iterator?

        for context_word in context:
            # Init etow with zeros
            etow = np.zeros(embedding_dim)
            classifiers = [(token, 1)] + [(target, 0)
                                          for target in table.sample(k_negative_sampling)]
            for target, label in classifiers:
                z = np.dot(W0[context_word], W1[target])
                p = sigmoid(z)
                g = lr * (label - p)
                # Error to backpropagate to W0
                etow += g * W1[target]
                W1[target] += g * W0[context_word]  # Update W1

            # Update W0
            W0[context_word] += etow

        word_count += 1

    global_word_count += (word_count - last_word_count)

    # Save model to file
    save(vocab, W0, "embedding.txt")
