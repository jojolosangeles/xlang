from keras import models
from keras import layers


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
