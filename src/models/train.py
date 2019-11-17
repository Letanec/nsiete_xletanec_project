import tensorflow.keras as keras
import os
import datetime
from src.models import model as m

def train_model(train_x, train_y, test_x, test_y, vocab):

    model = m.SentimentClassifier(len(vocab), 300, 64, 2)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("../logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1,
            profile_batch=0)
    ]

    model.fit(
        x=train_x,
        y=train_y,
        batch_size=100,
        epochs=5,
        callbacks=callbacks,
        validation_data=(test_x, test_y))