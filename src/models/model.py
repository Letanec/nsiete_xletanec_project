import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.initializers import Constant

class SentimentClassifier(keras.Model):

    def __init__(self, vocab_size, embedding_size, lstm_units, num_classes):
        super(SentimentClassifier, self).__init__()

        self.emb = Embedding(
            input_dim=vocab_size,
            output_dim=64,
            mask_zero=True)

        self.lstm = Bidirectional(LSTM(
            units=64))

        self.dense = Dense(
            units=1,
            activation='sigmoid')

    def call(self, x):
        mask = self.emb.compute_mask(x)
        x = self.emb(x)
        x = self.lstm(x, mask=mask)
        x = self.dense(x)
        return x
