from keras.datasets import mnist


def hello_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, y_train, X_val, y_val = X_train[:50000], y_train[:50000], X_train[50000:], y_train[50000:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


hello_dict = {
    "mnist": hello_mnist
}


def is_hello(name):
    return name in hello_dict


