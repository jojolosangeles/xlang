from keras.datasets import mnist, cifar10, imdb
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


def mnist_style_train(model, train_data, train_labels, val_data, val_labels):
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    hist = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    return hist


def mnist_style_load(loadfn, as_onehot=True):
    (X_train, y_train), (X_test, y_test) = loadfn()
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    X_train, y_train, X_val, y_val = X_train[:50000], y_train[:50000], X_train[50000:], y_train[50000:]
    if as_onehot:
        y_train = tf.one_hot(indices=y_train, depth=10)
        y_test = tf.one_hot(indices=y_test, depth=10)
        y_val = tf.one_hot(indices=y_val, depth=10)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


#
# RESULTS = (X_train, y_train), (X_val, y_val), (X_test, y_test)
#


def hello_mnist(as_onehot=True):
    return mnist_style_load(mnist.load_data, as_onehot)


def hello_cifar(as_onehot=True):
    return mnist_style_load(cifar10.load_data, as_onehot)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def hello_imdb(num_words=10000, vectorized=True):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2)
    if vectorized:
        train_data = vectorize_sequences(train_data, dimension=num_words)
        train_labels = np.asarray(train_labels).astype('float32')
        val_data = vectorize_sequences(val_data, dimension=num_words)
        val_labels = np.asarray(val_labels).astype('float32')
        test_data = vectorize_sequences(test_data, dimension=num_words)
        test_labels = np.asarray(test_labels).astype('float32')
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


#
# Loaders supported
#

hello_dict = {
    "mnist": hello_mnist,
    "cifar": hello_cifar,
    "imdb": hello_imdb
}


def is_hello(name):
    return name in hello_dict


