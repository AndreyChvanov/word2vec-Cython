import numpy as np
import re

MAX_CODE_LENGTH = 40


class VocabWord:
    def __init__(self, cn, point, word, code, codelen, index):
        self.cn = cn
        self.point = point
        self.word = word
        self.code = code
        self.codelen = codelen
        self.index = index


class Vocab:
    def __init__(self):
        self.vocab_size = 0
        self.vocab = []
        self.search_dict = {}

    @staticmethod
    def __process_lines(lines):
        line = lines.lower()
        line = re.sub(r"[^a-z0-9., ]", "", line)
        return line

    def add_word_to_vocab(self, word):
        self.vocab.append(VocabWord(cn=None,
                                    point=np.zeros(MAX_CODE_LENGTH, dtype=np.uint32),
                                    word=None,
                                    code=np.zeros(MAX_CODE_LENGTH, dtype=np.uint8),
                                    codelen=None,
                                    index=np.int32(self.vocab_size)))

        self.vocab[self.vocab_size].cn = 0
        self.vocab[self.vocab_size].word = word
        self.search_dict[word] = self.vocab_size
        self.vocab_size += 1
        return self.vocab_size - 1

    def search_vocab(self, word):
        try:
            i = self.search_dict[word]
            return i
        except KeyError:
            return -1

    def sort_vocab(self):
        self.vocab = [word_item for word_item in self.vocab if word_item.cn >= 5]
        self.vocab = sorted(self.vocab, key=lambda x: x.cn, reverse=True)
        self.search_dict = {item.word: i for i, item in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        for i in range(self.vocab_size):
            self.vocab[i].index = i

    def create_binary_tree(self):
        code = np.zeros(MAX_CODE_LENGTH, dtype=np.uint32)
        point = np.zeros(MAX_CODE_LENGTH, dtype=np.uint32)
        count = np.zeros(self.vocab_size * 2 + 1, dtype=np.uint32)
        binary = np.zeros(self.vocab_size * 2 + 1, dtype=np.uint32)
        parent_node = np.zeros(self.vocab_size * 2 + 1, dtype=np.uint32)
        for a in range(self.vocab_size):
            count[a] = self.vocab[a].cn
        for a in range(self.vocab_size, self.vocab_size * 2):
            count[a] = 1e9
        pos1 = self.vocab_size - 1
        pos2 = self.vocab_size
        for a in range(self.vocab_size - 1):
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1i = pos1
                    pos1 -= 1
                else:
                    min1i = pos2
                    pos2 += 1
            else:
                min1i = pos2
                pos2 += 1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2i = pos1
                    pos1 -= 1
                else:
                    min2i = pos2
                    pos2 += 1
            else:
                min2i = pos2
                pos2 += 1
            count[self.vocab_size + a] = count[min1i] + count[min2i]
            parent_node[min1i] = self.vocab_size + a
            parent_node[min2i] = self.vocab_size + a
            binary[min2i] = 1
        for a in range(self.vocab_size):
            b = a
            i = 0
            while True:
                code[i] = binary[b]
                point[i] = b
                i += 1
                b = parent_node[b]
                if b == self.vocab_size * 2 - 2:
                    break
            self.vocab[a].codelen = i
            self.vocab[a].point[0] = self.vocab_size - 2
            for b_ in range(0, i):
                self.vocab[a].code[i - b_ - 1] = code[b_]
                self.vocab[a].point[i - b_] = point[b_] - self.vocab_size

    def learn_vocab_from_file(self, file):
        file = self.__process_lines(file)
        self.vocab_size = 0
        for word in file.split(' '):
            if word == '':
                continue
            i = self.search_vocab(word)
            if i == -1:
                a = self.add_word_to_vocab(word)
                self.vocab[a].cn = 1
            else:
                self.vocab[i].cn += 1
        self.sort_vocab()
        self.create_binary_tree()
        return file

    def encode_sentence(self, sentence):
        indexes = []
        for word in sentence:
            if word == '':
                continue
            i = self.search_vocab(word)
            if i != -1:
                indexes.append(self.vocab[i])
            else:
                indexes.append(None)
        return indexes


