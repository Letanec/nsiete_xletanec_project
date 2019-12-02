import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout, GaussianNoise
from tensorflow.keras.initializers import Constant

class SentimentClassifier(keras.Model):

    def __init__(self, vocab_size, embedding_size, lstm_units, num_classes):
        super(SentimentClassifier, self).__init__()

        self.emb = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            mask_zero=True)

        self.gauss = GaussianNoise(stddev=0.3)

        self.dropout1 = Dropout(rate=0.3)

        self.lstm1 = Bidirectional(LSTM(
            return_sequences=True,
            units=lstm_units))

        self.lstm2 = Bidirectional(LSTM(
            units=lstm_units))

        self.dropout2 = Dropout(rate=0.4)

        self.dense = Dense(
            units=num_classes,
            activation='sigmoid')

    def call(self, x):
        mask = self.emb.compute_mask(x)
        x = self.emb(x)
        x = self.gauss(x)
        x = self.dropout1(x)
        x = self.lstm1(x, mask=mask)
        x = self.lstm2(x, mask=mask)
        x = self.dropout2(x)
        x = self.dense(x)
        return x
