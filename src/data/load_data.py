import pandas as pd
import numpy as np
import tensorflow.keras as keras

def load_data(filename):
    names = ['sent', 'tweet']
    data = pd.read_csv(filename, encoding="ISO-8859-1", names=names, delimiter=',')
    data = data.iloc[np.random.permutation(len(data))]
    data = data.reset_index(drop=True)
    print(data.head())
    data['words'] = data['tweet'].str.split()
    y = data['sent'].tolist()
    x = data['words'].tolist()

    return x, y

def load_vocabulary(x, z):
    words = set()
    for sample in x:
        words = words.union(sample)

    for sample in z:
        words = words.union(sample)

    vocabulary = {'<pad>': 0}
    for i, word in enumerate(words):
        vocabulary[word] = i + 1

    return vocabulary

def prepare():
    x, train_y = load_data('../data/processed/train.csv')
    z, test_y = load_data('../data/processed/test.csv')
    vocab = load_vocabulary(x, z)

    train_x = []
    for sample in x:
        train_x.append([vocab[word] for word in sample])

    train_x = keras.preprocessing.sequence.pad_sequences(train_x, padding='post')
    train_y = np.asarray(train_y, dtype=np.uint8)
    test_x = []
    for sample in z:
        test_x.append([vocab[word] for word in sample])

    test_x = keras.preprocessing.sequence.pad_sequences(test_x, padding='post')
    test_y = np.asarray(test_y, dtype=np.uint8)
    return train_x, train_y, test_x, test_y, vocab