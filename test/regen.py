from xlate.xperiment import Xperiment
from xlate import xlate
import sys

fileName = sys.argv[1]
lines = open(fileName, "r").readlines()


class Xbuilder:
    def __init__(self):
        self.xperiments = []
        self.xcurrent = None

    def process_line(self, line):
        if len(line) > 2 and line[0] == '#':
            line = line[2:]
            is_header, is_layer = self.parse(line)

            if is_header:
                data = line.split("=>")
                data = [d.strip() for d in data]
                self.xcurrent = Xperiment(data[1], data[0], data[2])
                self.xperiments.append(self.xcurrent)

            if is_header or is_layer:
                self.xcurrent.lines.append(line)

            if is_layer:
                line = line[2:]
                data = line.split()
                layer_type = data[0]
                class_name,params,attrs = self.xcurrent.layer_memory.get_params(layer_type, data)
                self.xcurrent.add_layer(class_name, params, attrs)

    def parse(self, line):
        data = line.split("=>")
        return len(data) == 3, line.startswith("- ")

    def gen_code(self):
        print("from keras import models")
        print("from keras import layers")
        for x in self.xperiments:
            x.prepare_to_code()
            print("")
            print("")
            print("#")
            for line in x.lines:
                print(f"# {line}")
            print("#")
            print(f"def model_{x.model_name}():")
            print("    model = models.Sequential()")
            for layer in x.layers:
                params = ", ".join(layer.params + layer.kwparams)
                print(f"    model.add(layers.{layer.class_name}({params}))")
            print("    return model")


builder = Xbuilder()
for line in lines:
    line = line.strip()
    builder.process_line(line)
builder.gen_code()



