from lang.xperiment import Xperiment
import sys

fileName = sys.argv[1]


class Xbuilder:
    def __init__(self):
        self.xperiments = []
        self.xcurrent = None
        self.layer_type_replacements = {}
        self.context = None

    def process_line(self, line):
        is_header, is_layer, is_with, is_loader_line, is_train_model_line = self.parse(line)

        if is_header:
            data = line.split("=>")
            data = [d.strip() for d in data]
            self.xcurrent = Xperiment(data[1], data[0], data[2])
            self.xperiments.append(self.xcurrent)

        if is_with:
            data = line.split()
            data = data[1].split("=")
            self.layer_type_replacements[data[0]] = data[1]

        if is_header or is_layer or is_with or is_loader_line or is_train_model_line:
            if is_loader_line:
                self.xcurrent.load_data.lines.append(line)
            elif is_train_model_line:
                self.xcurrent.train_model.lines.append(line)
            else:
                self.xcurrent.model_architecture.lines.append(line)

        if is_loader_line:
            self.xcurrent.load_data.process_loader_line(line)
        if is_layer:
            self.xcurrent.model_architecture.process_layer_line(line, self.layer_type_replacements)
        if is_train_model_line:
            self.xcurrent.train_model.process_training_line(line)

    def parse(self, line):
        # is it a Model Architecture line?
        data = line.split("=>")
        is_model_architecture = len(data) == 3
        if is_model_architecture:
            self.context = "Model Architecture"

        # is it a Load Data line?
        data = line.split()
        is_loader = len(data) == 2 and data[0] == 'load'
        if is_loader:
            self.context = "Load Data"

        # is it a Train Model line?
        is_train_model = len(data) > 2 and data[0] == 'train'
        if is_train_model:
            self.context = "Train Model"

        return is_model_architecture, \
               line.startswith("- ") and self.context == "Model Architecture", \
               line.startswith("with "), \
               is_loader or (line.startswith("- ") and self.context == "Load Data"),\
               is_train_model or (line.startswith("- ") and self.context == "Train Model")

    @staticmethod
    def imports():
        print("import numpy as np")
        print("import matplotlib.pyplot as plt")
        print("from keras import models")
        print("from keras import layers")
        print("from keras import regularizers")
        print("from keras.applications import VGG16")

    @staticmethod
    def gen_load_code(xperiment):
        x = xperiment
        if x.load_data.dataset_name is not None:
            print(f"from keras.datasets import {x.load_data.dataset_name}")
            print(f"from keras.utils.np_utils import to_categorical")
            x.load_data.print_lines_as_comment()
            print(f"def data_{x.model_name}():")
            print(f"    (train_data, train_labels), (test_data, test_labels) = {x.load_data.dataset_name}.load_data(num_words={x.model_architecture.num_words})")
            print(f"    train_data = vectorize_sequences(train_data, {x.load_data.max_sequence})")
            print(f"    test_data = vectorize_sequences(test_data, {x.load_data.max_sequence})")
            print(f"    train_labels = to_categorical(train_labels)")
            print(f"    test_labels = to_categorical(test_labels)")
            if x.load_data.validation_size is not None:
                print(f"    val_data = train_data[:{x.load_data.validation_size}]")
                print(f"    val_labels = train_labels[:{x.load_data.validation_size}]")
                print(f"    train_data = train_data[{x.load_data.validation_size}:]")
                print(f"    train_labels = train_labels[{x.load_data.validation_size}:]")
            else:
                print(f"    val_data = None")
                print(f"    val_labels = None")
            print("    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)")

    @staticmethod
    def gen_model_code(xperiment):
        x = xperiment
        x.model_architecture.print_lines_as_comment()
        print(f"def model_{x.model_name}({x.model_architecture.model_builder_params()}):")
        print("    model = models.Sequential()")
        for layer in x.model_architecture.layers:
            params = ", ".join(layer.params + layer.kwparams)
            print(f"    model.add({layer.class_name}({params}))")
        print("    return model")

    @staticmethod
    def gen_training_code(xperiment):
        x = xperiment
        if x.train_model.is_valid():
            x.train_model.print_lines_as_comment()
            print(f"def train_{x.model_name}(model, x_train, y_train, x_val, y_val):")
            print(f"    model.compile(optimizer='{x.train_model.optimizer}', loss='{x.train_model.loss}', metrics={str(x.train_model.metrics)})")
            print(f"    hist = model.fit(x_train, y_train, epochs={x.train_model.epochs}, batch_size={x.train_model.batch_size}, validation_data=(x_val, y_val))")
            print(f"    return hist")

    @staticmethod
    def gen_example(xperiment):
        x = xperiment
        print("")
        print("")
        print("#")
        print("# Example using generated methods")
        print("#")
        print(f"def {x.model_name}():")
        print(f"    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = data_{x.model_name}()")
        print(f"    model = model_{x.model_name}()")
        print(f"    hist = train_{x.model_name}(model, train_data, train_labels, val_data, val_labels)")
        print(f"    show_hist(epochs={x.train_model.epochs}, keys={str(x.train_model.show_keys)}, hist_dict=hist.history)")

    def gen_model(self):
        self.imports()
        for x in self.xperiments:
            x.prepare_to_code()
            self.gen_load_code(x)
            self.gen_model_code(x)
            self.gen_training_code(x)
            self.gen_example(x)



lines = open(fileName, "r").readlines()
python_file = fileName.endswith(".py")
builder = Xbuilder()
for line in lines:
    line = line.strip()
    if python_file:
        if len(line) > 2:
            line = line[2:]
    builder.process_line(line)
builder.gen_model()



