import numpy as np
import logging
from gensim import matutils
import numpy as np

REAL = np.float32

# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger()


class Model:
    def __init__(self, vocab, vocab_size, hidden_size):
        self.vocab = vocab
        self.index2word = {word_item.index: word_item.word for word_item in self.vocab.vocab}
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.w0 = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.hidden_size)) / self.hidden_size
        self.w1 = np.zeros((self.hidden_size, self.vocab_size))
        self.window = 7
        self.w0 = self.w0.astype(REAL)
        self.w1 = self.w1.astype(REAL)
        # self.w0 = self.w0.reshape(-1)
        # self.w1 = self.w1.reshape(-1)

    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            # logger.info("precomputing L2-norms of word weight vectors")
            self.w0norm = np.vstack(matutils.unitvec(vec) for vec in self.w0).astype(REAL)

    def most_similar(self, positive=[], negative=[], topn=10):

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, str) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, str) else word for word in negative]
        all_words, mean = set(), []
        for word, weight in positive + negative:
            word_index = self.vocab.search_vocab(word)
            if word_index != 1:
                mean.append(weight * matutils.unitvec(self.w0[self.vocab.vocab[word_index].index]))
                all_words.add(self.vocab.vocab[word_index].index)
            else:
                pass
                # logger.warning("word '%s' not in vocabulary; ignoring it" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(REAL)
        dists = np.dot(self.w0norm, mean)
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]

    def accuracy(self, questions, restrict_vocab=30000):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        self.init_sims()
        ok_vocab = sorted(self.vocab.vocab, key=lambda item: -item.cn)[:restrict_vocab]
        ok_vocab = {item.word: item for item in ok_vocab}
        ok_index = set(v.index for v in ok_vocab.values())

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                pass
                 # logger.info("%s: %.1f%% (%i/%i)" %
                 #            (section['section'], 100.0 * correct / (correct + incorrect),
                 #             correct, correct + incorrect))


        sections, section = [], None
        for line_no, line in enumerate(open(questions)):
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    # log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.lower() for word in
                                         line.split()]
                except:
                    # logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                    pass
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    # logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
                    continue

                # find the most likely prediction, ignoring OOV words and input words
                predicted, ignore = None, set(self.vocab.vocab[self.vocab.search_vocab(v)].index for v in [a, b, c])
                for index in np.argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            pass
                            # logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1
        if section:
            # store the last section, too
            sections.append(section)
            # log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct'] for s in sections),
                 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        return sections
