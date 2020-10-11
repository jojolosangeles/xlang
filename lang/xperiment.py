from lang import args
from lang import xlate
from lang.dimension import implicit_dimension, reset_default_dimension
from lang.output_spec import OutputSpec


class Xlayer:
    def __init__(self, class_name, params, kwparams):
        self.class_name = class_name
        self.params = params if params else []
        self.kwparams = kwparams
        print(self)

    def __repr__(self):
        return f"XLayer({self.class_name}, {len(self.params)} params, {len(self.kwparams)} kwparaams)"


layer_config = {
    # layer_type: ( class_name,
    #               expects_flattened_input,
    #               is_dimensional,
    #               param_value_offsets,
    #               attr_values_start_offset,
    #               fixed_params )
    "bidirectional": ("layers.Bidirectional", False, False, [], 1, None),
    "conv": ("layers.Conv", False, True, [1, 2], 3, None),
    "convbase": ("Token_1", False, False, [], 2, "include_top=False"),
    "dense": ("layers.Dense", True, False, [1], 2, None),
    "embed": ("layers.Embedding", False, False, [1], 2, None),
    "dropout": ("layers.Dropout", False, False, [1], None, None),
    "flatten": ("layers.Flatten", False, False, [], None, None),
    "global": ("layers.GlobalMaxPooling", False, True, [], None, None),
    "global_average_pool": ("layers.GlobalAveragePooling", False, True, [], None, None),
    "GRU": ("layers.GRU", False, False, [1], 2, None),
    "LSTM": ("layers.LSTM", False, False, [1], None, None),
    "maxpool": ("layers.MaxPooling", False, True, [1], None, None),
    "separableconv": ("layers.SeparableConv", False, True, [1, 2], 3, None),
    "simpleRNN": ("layers.SimpleRNN", False, False, [1], None, None)
}


def layer_class_name(layer_type):
    return layer_config[layer_type][0]


def expects_flattened_input(layer_type):
    return layer_config[layer_type][1]


class LayerMemory:
    def __init__(self):
        self.remembered_param_values = {}
        self.remembered_attr_values = {}
        for layer_type in layer_config:
            self.remembered_param_values[layer_type] = []
            self.remembered_attr_values[layer_type] = []

    def get_params(self, layer_type, tokens):
        class_name, _, is_dimensional, param_indices, attr_start_index, custom_attrs = layer_config[layer_type]
        param_values = self.param_memory(layer_type, tokens, param_indices)
        attr_values = []
        if attr_start_index != None:
            attr_values = self.attr_memory(layer_type, tokens, list(range(attr_start_index, len(tokens))))
        if is_dimensional:
            class_name = f"{class_name}{implicit_dimension(param_values)}D"
        if custom_attrs:
            attr_values.append(custom_attrs)
        return class_name, param_values, attr_values

    def param_memory(self, layer_type, tokens, indices):
        self.remembered_param_values[layer_type] = self.memory(self.remembered_param_values[layer_type], tokens,
                                                               indices)
        return self.remembered_param_values[layer_type]

    def attr_memory(self, layer_type, tokens, indices):
        self.remembered_attr_values[layer_type] = self.memory(self.remembered_attr_values[layer_type], tokens, indices)
        return self.remembered_attr_values[layer_type]

    def memory(self, remembered_values, tokens, indices):
        specified_values = [tokens[i] for i in indices if i < len(tokens)]
        if len(remembered_values) > len(specified_values):
            specified_values.extend(remembered_values[len(specified_values):])
        return specified_values


class Lines:
    def __init__(self):
        self.lines = []

    def print_lines_as_comment(self):
        print("")
        print("")
        print("#")
        for line in self.lines:
            print(f"# {line}")
        print("#")


class ModelArchitecture(Lines):
    def __init__(self, input_spec, output_spec):
        # track lines specifying architecture
        Lines.__init__(self)

        # the input and output data specifications for this architecture
        self.input_spec = input_spec
        self.output_spec = output_spec

        # TODO: looks wrong, dimension handling needs simplification
        reset_default_dimension()

        # the input_spec is converted to the input_shape needed by the model
        # the output_spec is converted to the output_layer of the model
        self.input_shape = args.as_kwarg_str('input_shape', xlate.as_shape(input_spec))
        self.num_words = xlate.as_num_words(input_spec)
        self.output_layer = Xlayer(self.output_spec.layer_class_name, self.output_spec.param_values,
                                   self.output_spec.attrs)

        # the layers in this architecture, the layer_memory allows layer specifications to
        # repeat values passed to previous layers without being explicit.  For example,
        #
        # - dense 32 relu
        # - dense 64 relu
        # - dense 64 relu
        #
        # can be represented as:
        #
        # - dense 32 relu
        # - dense 64
        # - dense
        #
        self.layers = []
        self.layer_memory = LayerMemory()

    def add_layer(self, class_name, param_values=None, attrs={}):
        xlayer = Xlayer(
            class_name,
            args.args_as_list(param_values),
            [args.as_kwarg_str(name, val) for name, val in args.as_kwarg_list(attrs)]
        )
        self.layers.append(xlayer)
        return xlayer

    def process_layer_line(self, line, layer_type_replacements):
        line = line[2:]
        data = line.split()
        layer_type = data[0]
        if layer_type in layer_type_replacements:
            layer_type = layer_type_replacements[layer_type]
        class_name, params, attrs = self.layer_memory.get_params(layer_type, data)
        if not self.layers and expects_flattened_input(layer_type):
            self.add_layer(layer_class_name("flatten"))
        layer = self.add_layer(xlate.token_val(class_name, data), params, attrs)
        if layer_type == "bidirectional":
            data = data[1:]
            layer_type = data[0]
            class_name, params, attrs = self.layer_memory.get_params(layer_type, data)
            param_str = ", ".join(args.args_as_list(params))
            layer.params.append(f"{class_name}({param_str})")

    def prepare_to_code(self):
        # stacked GRU layers require return_sequences=True
        for i, layer in enumerate(self.layers[:-1]):
            if "GRU" in layer.class_name and "GRU" in self.layers[i + 1].class_name:
                layer.kwparams.append('return_sequences=True')

        # embedding puts shape as first parameter
        if "Embedding" in self.layers[0].class_name:
            n = 0
            data = self.input_spec.split()
            for d in data:
                n = max(n, xlate.as_number(d))
            self.layers[0].params.insert(0, str(n))
        else:
            self.layers[0].kwparams.append(self.input_shape)
        self.layers.append(self.output_layer)

    def model_builder_params(self):
        if self.input_spec == "features":
            return "n_features"
        else:
            return ""


class LoadData(Lines):
    def __init__(self):
        # track lines specifying data loading
        Lines.__init__(self)
        self.dataset_name = None
        self.validation_size = None
        self.max_sequence = None

    def process_loader_line(self, line):
        data = line.split()
        if len(data) == 2 and data[0] == "load":
            self.dataset_name = data[1]
        if len(data) == 4 and data[0] == "-" and data[1] == "validate" and data[2] == "with":
            self.validation_size = data[3]


class TrainModel(Lines):
    def __init__(self):
        Lines.__init__(self)
        self.optimizer = "sgd"
        self.loss = "binary_crossentropy"
        self.metrics = ['accuracy']
        self.epochs = 10
        self.batch_size = 0
        self.show_keys = []

    def is_valid(self):
        return self.optimizer is not None

    def process_training_line(self, line):
        data = line.split()
        if data[0] == "train":
            for x in data[1:]:
                if self.optimizer is None:
                    self.optimizer = x
                else:
                    self.metrics.append(x)
        elif data[2].startswith("epochs"):
            self.epochs = data[1]
        elif data[1].startswith("batch"):
            self.batch_size = data[2]
        self.set_show_keys(data)

    def set_show_keys(self, data):
        show_found = False
        for x in data:
            if x == "show":
                show_found = True
            elif show_found:
                self.show_keys.append(x)


class Xperiment:
    """
    An experiment has these sections:

      Loading Data
      Model Architecture
      Training the Model
      Evaluating the Model

    Each section has it's own configuration lines, and section-specific data needed to generate the code.
    """

    def __init__(self, model_name, input_spec, output_spec):
        self.load_data = LoadData()
        self.output_spec = OutputSpec(output_spec, layer_config)
        self.model_architecture = ModelArchitecture(input_spec, self.output_spec)
        self.train_model = TrainModel()
        self.evaluate_model = None

        self.model_name = model_name

        # data source for this experiment
        self.dataset_name = None
        self.loader_lines = []

    def prepare_to_code(self):
        self.model_architecture.prepare_to_code()
        self.train_model.loss = self.output_spec.loss_fn
        self.load_data.max_sequence = self.model_architecture.num_words
