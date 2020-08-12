from keras import models
from keras import layers
from keras import regularizers
from keras.applications import VGG16


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
# 784 => mnist_p63 => 10 probabilities
# - dense 32 relu
#
def model_mnist_p63():
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(784,)))
    model.add(layers.Dense(10, activation='softmax'))
    return model


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
