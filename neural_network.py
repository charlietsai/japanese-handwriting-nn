import struct
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from preprocessing.data_utils import get_ETL8B_data
from sklearn import datasets, metrics, cross_validation
import skflow
import tensorflow as tf

def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

def conv_model(X, y):
    # reshape X to 4d tensor with 2nd and 3rd dimensions being image width and height
    # final dimension being the number of color channels
    X = tf.reshape(X, [-1, 28, 28, 1])
    # first conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    # second conv layer will compute 64 features for each 5x5 patch
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # densely connected layer with 1024 neurons
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


# Load dataset.
# Hiragana data set with subset of writers
characters, labels = get_ETL8B_data(1, range(0,75), 160, vectorize=True, resize=(28,28))

# rename labels from 0 to n_labels-1
unique_labels = list(set(labels))
labels_dict = {unique_labels[i]:i for i in range(len(unique_labels))}
new_labels = np.array([labels_dict[l] for l in labels], dtype=np.int32)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(characters,
                                                                     new_labels,
                                                                     test_size=0.3,
                                                                     random_state=42)
# Training and predicting
n_classes = len(set(labels))
classifier = skflow.TensorFlowEstimator(model_fn=conv_model,
                                        n_classes=n_classes,
                                        batch_size=100,
                                        steps=2000,
                                        learning_rate=0.001)
# Fit and predict.
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {0:f}'.format(score))