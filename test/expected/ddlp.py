import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from keras.applications import VGG16
from keras.datasets import mnist


#
# 28*28 => mnist_p53 => 10 probabilities
# - dense 512 relu
#
def model_mnist_p53():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


#
# Example using generated methods
#
def mnist_p53():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_mnist_p53()
    model = model_mnist_p53()
    hist = train_mnist_p53(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 784 => mnist_p63 => 10 probabilities
# - dense 32 relu
#
def model_mnist_p63():
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


#
# Example using generated methods
#
def mnist_p63():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_mnist_p63()
    model = model_mnist_p63()
    hist = train_mnist_p63(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p72 => probability
# - dense 16 relu
# - dense
#
def model_imdb_p72():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p72():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p72()
    model = model_imdb_p72()
    hist = train_imdb_p72(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => reuters_p80 => 46 probabilities
# - dense 64 relu
# - dense
#
def model_reuters_p80():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    return model


#
# Example using generated methods
#
def reuters_p80():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_reuters_p80()
    model = model_reuters_p80()
    hist = train_reuters_p80(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => reuters_p84 => 46 probabilities
# - dense 64 relu
# - dense 4
#
def model_reuters_p84():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    return model


#
# Example using generated methods
#
def reuters_p84():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_reuters_p84()
    model = model_reuters_p84()
    hist = train_reuters_p84(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# features => boston_p86 => float
# - dense 64 relu
# - dense
#
def model_boston_p86(n_features):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def boston_p86():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_boston_p86()
    model = model_boston_p86()
    hist = train_boston_p86(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p105 => probability
# - dense 16 relu
# - dense
#
def model_imdb_p105():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p105():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p105()
    model = model_imdb_p105()
    hist = train_imdb_p105(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p105b => probability
# - dense 4 relu
# - dense
#
def model_imdb_p105b():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p105b():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p105b()
    model = model_imdb_p105b()
    hist = train_imdb_p105b(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p106 => probability
# - dense 512 relu
# - dense
#
def model_imdb_p106():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p106():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p106()
    model = model_imdb_p106()
    hist = train_imdb_p106(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p108 => probability
# - dense 16 relu l2 0.001
# - dense
#
def model_imdb_p108():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p108():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p108()
    model = model_imdb_p108()
    hist = train_imdb_p108(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p110 => probability
# - dense 16 relu
# - dropout 0.5
# - dense
# - dropout
#
def model_imdb_p110():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p110():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p110()
    model = model_imdb_p110()
    hist = train_imdb_p110(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 28x28x1 => mnist_p120 => 10 probabilities
# - conv 32 3x3 relu
# - maxpool 2x2
# - conv 64
# - maxpool
# - conv
# - flatten
# - dense 64 relu
#
def model_mnist_p120():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


#
# Example using generated methods
#
def mnist_p120():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_mnist_p120()
    model = model_mnist_p120()
    hist = train_mnist_p120(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 150x150x3 => catdog_p134 => probability
# - conv 32 3x3 relu
# - maxpool 2x2
# - conv 64
# - maxpool
# - conv 128
# - maxpool
# - conv 128
# - maxpool
# - flatten
# - dense 512 relu
#
def model_catdog_p134():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def catdog_p134():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_catdog_p134()
    model = model_catdog_p134()
    hist = train_catdog_p134(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 150x150x3 => catdog_p141 => probability
# - conv 32 3x3 relu
# - maxpool 2x2
# - conv 64
# - maxpool
# - conv 128
# - maxpool
# - conv 128
# - maxpool
# - flatten
# - dropout 0.5
# - dense 512 relu
#
def model_catdog_p141():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def catdog_p141():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_catdog_p141()
    model = model_catdog_p141()
    hist = train_catdog_p141(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 4*4*512 => catdog_p148 => probability
# - dense 256 relu
# - dropout 0.5
#
def model_catdog_p148():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(4*4*512,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def catdog_p148():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_catdog_p148()
    model = model_catdog_p148()
    hist = train_catdog_p148(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 150x150x3 => catdog_p150 => probability
# - convbase VGG16 imagenet
# - flatten
# - dense 256 relu
#
def model_catdog_p150():
    model = models.Sequential()
    model.add(VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def catdog_p150():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_catdog_p150()
    model = model_catdog_p150()
    hist = train_catdog_p150(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p187 => probability
# - embed 8 input_length=20
# - flatten
#
def model_imdb_p187():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 8, input_length=20))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p187():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p187()
    model = model_imdb_p187()
    hist = train_imdb_p187(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p191 => probability
# - embed 100 input_length=100
# - flatten
# - dense 32 relu
#
def model_imdb_p191():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 100, input_length=100))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p191():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p191()
    model = model_imdb_p191()
    hist = train_imdb_p191(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p200 => probability
# - embed 32
# - simpleRNN 32
#
def model_imdb_p200():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 32))
    model.add(layers.SimpleRNN(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p200():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p200()
    model = model_imdb_p200()
    hist = train_imdb_p200(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p205 => probability
# - embed 32
# - LSTM 32
#
def model_imdb_p205():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 32))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p205():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p205()
    model = model_imdb_p205()
    hist = train_imdb_p205(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# timed 32x7 => jena_p213 => float
# - flatten
# - dense 32 relu
#
def model_jena_p213():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32,7)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p213():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p213()
    model = model_jena_p213()
    hist = train_jena_p213(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p215 => float
# - GRU 32
#
def model_jena_p215():
    model = models.Sequential()
    model.add(layers.GRU(32, input_shape=(None,7)))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p215():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p215()
    model = model_jena_p215()
    hist = train_jena_p215(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p217 => float
# - GRU 32 dropout=0.2 recurrent_dropout=0.2
#
def model_jena_p217():
    model = models.Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None,7)))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p217():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p217()
    model = model_jena_p217()
    hist = train_jena_p217(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p218 => float
# - GRU 32 dropout=0.2 recurrent_dropout=0.2
# - GRU 64
#
def model_jena_p218():
    model = models.Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=(None,7)))
    model.add(layers.GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p218():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p218()
    model = model_jena_p218()
    hist = train_jena_p218(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p220 => probability
# - embed 128
# - LSTM 32
#
def model_imdb_p220():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p220():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p220()
    model = model_imdb_p220()
    hist = train_imdb_p220(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# text 10k => imdb_p221 => probability
# - embed 32
# - bidirectional LSTM 32
#
def model_imdb_p221():
    model = models.Sequential()
    model.add(layers.Embedding(10000, 32))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


#
# Example using generated methods
#
def imdb_p221():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_imdb_p221()
    model = model_imdb_p221()
    hist = train_imdb_p221(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p227 => float
# - embed 128 input_length=500
# - conv 32 7 relu
# - maxpool 5
# - conv
# - global maxpool
#
def model_jena_p227():
    model = models.Sequential()
    model.add(layers.Embedding(7, 128, input_length=500))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p227():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p227()
    model = model_jena_p227()
    hist = train_jena_p227(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p228 => float
# - conv 32 5 relu
# - maxpool 3
# - conv
# - maxpool
# - conv
# - global maxpool
#
def model_jena_p228():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None,7)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p228():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p228()
    model = model_jena_p228()
    hist = train_jena_p228(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# time 7 => jena_p230 => float
# - conv 32 5 relu
# - maxpool 3
# - conv
# - GRU 32 dropout=0.1 recurrent_dropout=0.5
#
def model_jena_p230():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None,7)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    return model


#
# Example using generated methods
#
def jena_p230():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_jena_p230()
    model = model_jena_p230()
    hist = train_jena_p230(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 64x64x3 => image_p262 => 10 probabilities
# with conv=separableconv
# - conv 32 3x3 relu
# - conv 64
# - maxpool 2x2
# - conv 64
# - conv 128
# - maxpool 2x2
# - conv 64
# - conv 128
# - global_average_pool
# - dense 32 relu
#
def model_image_p262():
    model = models.Sequential()
    model.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
    model.add(layers.SeparableConv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.SeparableConv2D(64, (3,3), activation='relu'))
    model.add(layers.SeparableConv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.SeparableConv2D(64, (3,3), activation='relu'))
    model.add(layers.SeparableConv2D(128, (3,3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


#
# Example using generated methods
#
def image_p262():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_image_p262()
    model = model_image_p262()
    hist = train_image_p262(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)


#
# 60x26 => nietz_p275 => 26 probabilities
# - LSTM 128
#
def model_nietz_p275():
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=(60,26)))
    model.add(layers.Dense(26, activation='softmax'))
    return model


#
# Example using generated methods
#
def nietz_p275():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_nietz_p275()
    model = model_nietz_p275()
    hist = train_nietz_p275(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=0, keys=[], hist_dict=hist.history)
    return model.evaluate(test_data, test_labels, return_dict=True)
