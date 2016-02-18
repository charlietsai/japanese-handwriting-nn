#!/opt/rh/python27/root/usr/bin/python
#SBATCH -p iric
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err
#SBATCH --time=1200:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=ALL
#SBATCH  --mail-user=SUNETID@stanford.edu
#SBATCH --ntasks-per-node=16

import numpy as np
import struct
import skflow
import tensorflow as tf
from PIL import Image, ImageEnhance
from preprocessing.data_utils import get_ETL8B_data
from sklearn import datasets, metrics, cross_validation
from sklearn.utils import shuffle


def exp_decay(global_step):
    return tf.train.exponential_decay(0.1, global_step, 100, 0.70)

writersPerChar = 160

characters, labels = np.array([]), np.array([])

for i in range(1,4):
    if i == 3:
       max_records = 315
    else:
       max_records = 319
    
    chars, labs = get_ETL8B_data(i, range(0,max_records), writersPerChar, vectorize=True, resize=(28,28))
    characters = np.concatenate((characters,chars), axis=0)
    labels = np.concatenate((labels,labs), axis=0)

for dataset in range(7,14):
    if dataset < 10:
        dataset = '0'+str(dataset)
    chars, labs = get_ETL8B_data(dataset, range(0,8), writersPerChar, database='ETL1C', vectorize=True, resize=(28,28))
    characters = np.concatenate((characters,chars), axis=0)
    labels = np.concatenate((labels,labs), axis=0)

# rename labels from 0 to n_labels-1
unique_labels = list(set(labels))
labels_dict = {unique_labels[i]:i for i in range(len(unique_labels))}
new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

characters_shuffle, new_labels_shuffle = shuffle(characters, new_labels, random_state=0)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(characters_shuffle,
                                                                     new_labels_shuffle,
                                                                     test_size=0.2,
                                                                     random_state=42)


n_classes = len(set(labels))
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[500,500,500],
                                            n_classes=n_classes,
                                            steps=100000,
                                            learning_rate=exp_decay,
                                           )

# Fit and predict.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_train)
score = metrics.accuracy_score(y_train, y_pred)
print('Training Accuracy: {0:f}'.format(score))

y_pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print('Test Accuracy: {0:f}'.format(score))