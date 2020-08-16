from keras import models
from keras import layers
from keras import regularizers
from keras.applications import VGG16
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


#
# load reuters
# - validate with 1000
# - train rmsprop accuracy with 10 epochs batch 512
#
def data_reuters_test_1():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    val_data = train_data[:1000]
    val_labels = train_labels[:1000]
    train_data = train_data[1000:]
    train_labels = train_labels[1000:]
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


#
# text 10k => reuters_test_1 => 46 probabilities
# - dense 64 relu
# - dense
#
def model_reuters_test_1():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    return model
