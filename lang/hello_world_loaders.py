from keras.datasets import mnist, cifar10


def mnist_style_train(model, train_data, train_labels, val_data, val_labels):
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    hist = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    return hist


def mnist_style_load(loadfn):
    (X_train, y_train), (X_test, y_test) = loadfn()
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    X_train, y_train, X_val, y_val = X_train[:50000], y_train[:50000], X_train[50000:], y_train[50000:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def hello_mnist():
    return mnist_style_load(mnist.load_data)


def hello_cifar():
    return mnist_style_load(cifar10.load_data)


hello_dict = {
    "mnist": hello_mnist,
    "cifar": hello_cifar
}


def is_hello(name):
    return name in hello_dict


