from xlate.xperiment import Xperiment
from xlate import xlate
import sys

fileName = sys.argv[1]
lines = open(fileName, "r").readlines()


class Xbuilder:
    def __init__(self):
        self.xperiments = []
        self.xcurrent = None
        self.layer_type_replacements = {}

    def process_line(self, line):
        if len(line) > 2 and line[0] == '#':
            line = line[2:]
            is_header, is_layer, is_with = self.parse(line)

            if is_header:
                data = line.split("=>")
                data = [d.strip() for d in data]
                self.xcurrent = Xperiment(data[1], data[0], data[2])
                self.xperiments.append(self.xcurrent)

            if is_with:
                data = line.split()
                data = data[1].split("=")
                self.layer_type_replacements[data[0]] = data[1]

            if is_header or is_layer or is_with:
                self.xcurrent.lines.append(line)

            if is_layer:
                line = line[2:]
                data = line.split()
                layer_type = data[0]
                if layer_type in self.layer_type_replacements:
                    layer_type = self.layer_type_replacements[layer_type]
                class_name,params,attrs = self.xcurrent.layer_memory.get_params(layer_type, data)
                layer = self.xcurrent.add_layer(xlate.token_val(class_name, data), params, attrs)
                if layer_type == "bidirectional":
                    data = data[1:]
                    layer_type = data[0]
                    class_name, params, attrs = self.xcurrent.layer_memory.get_params(layer_type, data)
                    param_str = ", ".join(xlate.params_as_code(params))
                    layer.params.append(f"{class_name}({param_str})")

    def parse(self, line):
        data = line.split("=>")
        return len(data) == 3, line.startswith("- "), line.startswith("with ")

    def gen_code(self):
        print("from keras import models")
        print("from keras import layers")
        print("from keras import regularizers")
        print("from keras.applications import VGG16")
        for x in self.xperiments:
            x.prepare_to_code()
            print("")
            print("")
            print("#")
            for line in x.lines:
                print(f"# {line}")
            print("#")
            print(f"def model_{x.model_name}({x.model_builder_params()}):")
            print("    model = models.Sequential()")
            for layer in x.layers:
                params = ", ".join(layer.params + layer.kwparams)
                print(f"    model.add({layer.class_name}({params}))")
            print("    return model")


builder = Xbuilder()
for line in lines:
    line = line.strip()
    builder.process_line(line)
builder.gen_code()



