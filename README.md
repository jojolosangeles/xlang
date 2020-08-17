## Xlang

This is based on the book Deep Learning with Python, which has some very concise code for many different
types of problems.  

One thing I learned from this book is that once the incoming information needs to be designed in a way
that the learning model can handle.  

xlang extends that concept (sort of) with a minimalist description of the information needed to
load/train/evaluate a keras model.  The minimalist description is processed into Python
code.

Here's the reuters example:

Text file: reuters.x
```yaml
text 10k => reuters_test_1 => 46 probabilities
- dense 64 relu
- dense

load reuters
- validate with 1000

train rmsprop accuracy
- 10 epochs, show loss accuracy
- batch 512
```

Code below generated from above text file: python create_model.py models/reuters.x
```python
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import regularizers
from keras.applications import VGG16
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


#
# load reuters
# - validate with 1000
#
def data_reuters_test_1():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    train_data = vectorize_sequences(train_data, 10000)
    test_data = vectorize_sequences(test_data, 10000)
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


#
# train rmsprop accuracy
# - 10 epochs, show loss accuracy
# - batch 512
#
def train_reuters_test_1(model, x_train, y_train, x_val, y_val):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
    return hist


#
# Example using generated methods
#
def reuters_test_1():
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = data_reuters_test_1()
    model = model_reuters_test_1()
    hist = train_reuters_test_1(model, train_data, train_labels, val_data, val_labels)
    show_hist(epochs=10, keys=['loss', 'accuracy'], hist_dict=hist.history)
```
