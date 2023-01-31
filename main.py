import pyximport
import numpy as np
from model import Model
from vocab import Vocab
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

pyximport.install(setup_args={"include_dirs": np.get_include()})

from pure_cython import train_sentence

if __name__ == "__main__":

    with open('data/text8', 'r') as f:
        file = f.read()

    start = time.time()
    vocab = Vocab()
    lines = vocab.learn_vocab_from_file(file)
    print(f'vocab created, vocab size: {len(vocab.vocab)}, time {time.time() - start}')
    lines = lines.split(" ")
    x = vocab.encode_sentence(lines)

    model = Model(vocab=vocab, vocab_size=vocab.vocab_size, hidden_size=200)


    sentences = []
    for i in range(0, len(x), 1000):
        try:
            sentences.append(x[i:i + 1000])
        except:
            sentences.append(x[i:])

    lrs = np.linspace(0.025, 0.0001, len(sentences))
    bar = tqdm(range(0, len(sentences), 1))

    iters_numbs = []
    acc_list = []

    for i in bar:
        cur_sentence = sentences[i]
        train_sentence(model, cur_sentence, alpha=lrs[i])
        if i % 5000 == 0 or i == len(sentences) - 1:
            acc = model.accuracy('data/questions-words.txt')
            total = acc[-1]
            print(f" correct : {total['correct']}/ {total['correct'] + total['incorrect']}")
            acc_value = round(total['correct'] / (total['correct'] + total['incorrect']), 2)
            bar.set_description(f'Accuracy - {100 * acc_value} %')
            iters_numbs.append(i)
            acc_list.append(acc_value)

    plt.plot(iters_numbs, acc_list)
    plt.show()
