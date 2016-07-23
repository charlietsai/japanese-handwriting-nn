"""Example job for running a neural network."""

import h5py
import numpy as np

from preprocessing.make_keras_input import data
from models import M16


def load_model_weights(name, model):
    try:
        model.load_weights('weights/M16-hiragana_weights.h5')
    except:
        print "Can't load weights!"


def save_model_weights(name, model):
    try:
        model.save_weights(name)
    except:
        print "failed to save classifier weights"
    pass

X_train, y_train, X_test, y_test = data(mode='hiragana')
n_output = y_train.shape[1]

model = M16(n_output=n_output, input_shape=(1, 64, 64))

load_model_weights('weights/M16-hiragana_weights.h5', model)

adam = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam)

model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=16,
          show_accuracy=True)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=16,
                            show_accuracy=True,
                            verbose=0)
print "Training size: ", X_train.shape[0]
print "Test size: ", X_test.shape[0]
print "Test Score: ", score
print "Test Accuracy: ", acc
save_model_weights('weights/M16-hiragana_weights.h5', model)
