def show_hist(epochs, keys, hist_dict):
    plt.clf()
    plt.figure(figsize=(11,8))
    rows = len(keys)
    cols = 2
    epochs = range(1, epochs+1)
    for i,key in enumerate(keys):
        plt.subplot(rows, cols, i+1)
        training_value = hist_dict[key]
        validation_value = hist_dict[f"val_{key}"]
        plt.plot(epochs, training_value, 'bo', label=f"Training {key}")
        plt.plot(epochs, validation_value, 'b', label=f"Validation {key}")
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
    plt.show()

def vectorize_sequences(sequences, dimension=10000):
    n_sequences = len(sequences)
    results = np.zeros((n_sequences, dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


import os

imdb_dir = '/Users/johannesjohannsen/ml/dlip/imdb/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-1:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences()
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:]
y_val = labels[training_samples:]

from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results
vectorize_sequences([s1,s2,s3], dimension=11)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
x_train[0]
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#
# text 10k => imdb_test_1 => probability
# - dense 16 relu
# - dense
#
def model_imdb_test_1():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
model = model_imdb_test_1()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
hist = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history_dict = hist.history

plotit(history_dict, ['loss','accuracy'])
