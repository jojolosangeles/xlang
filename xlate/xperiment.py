from xlate import xlate

class Xlayer:
    def __init__(self, class_name, params, kwparams):
        self.class_name = class_name
        self.params = params
        self.kwparams = kwparams


layer_config = {
    # layer_type: ( class_name, param_values, attr_values )
    "conv": ("Conv", True, [1, 2], [3]),
    "dense": ("Dense", False, [1], [2]),
    "dropout": ("Dropout", False, [1], []),
    "flatten": ("Flatten", False, [], []),
    "maxpool": ("MaxPooling", True, [1], [])
}


class LayerMemory:
    def __init__(self):
        self.remembered_param_values = {}
        self.remembered_attr_values = {}
        for layer_type in layer_config:
            self.remembered_param_values[layer_type] = []
            self.remembered_attr_values[layer_type] = []

    def get_params(self, layer_type, tokens):
        class_name, is_dimensional, param_indices, attr_indices = layer_config[layer_type]
        param_values = self.param_memory(layer_type, tokens, param_indices)
        attr_values = self.attr_memory(layer_type, tokens, attr_indices)
        if is_dimensional:
            class_name = f"{class_name}{xlate.implicit_dimension(param_values)}D"
        return class_name, param_values, attr_values

    def param_memory(self, layer_type, tokens, indices):
        self.remembered_param_values[layer_type] = self.memory(self.remembered_param_values[layer_type], tokens, indices)
        return self.remembered_param_values[layer_type]

    def attr_memory(self, layer_type, tokens, indices):
        self.remembered_attr_values[layer_type] = self.memory(self.remembered_attr_values[layer_type], tokens, indices)
        return self.remembered_attr_values[layer_type]

    def memory(self, remembered_values, tokens, indices):
        specified_values = [tokens[i] for i in indices if i < len(tokens)]
        if len(remembered_values) > len(specified_values):
            specified_values.extend(remembered_values[len(specified_values):])
        return specified_values


#
#  Xperiment(model_name, input, output)
#    input_shape = as_kwparam('input_shape', as_shape(input))
#    output_layer = as_output_layer(output)
#    layers = []
#
#    add_layer(layer_type, param_values, attrs)
#      layers.add(make_layer(
#        as_layer_class(layer_type, implicit_dimension(param_values)),
#        [ as_param(val) for val in param_values ],
#        [ as_kwparam(name, val) for name, val in as_kwparam_list(attrs) ]
#      ))
#
#    prepare_to_code()
#      layers[0].kwparams.add(input_shape)
#      layers.add(output_layer)
#
class Xperiment:
    def __init__(self, model_name, input_spec, output_spec):
        self.model_name = model_name
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.input_shape = xlate.as_kwparam('input_shape', xlate.as_shape(input_spec))
        self.output_layer = Xlayer(*xlate.as_output_layer_params(output_spec))
        self.lines = []
        self.layers = []
        self.layer_memory = LayerMemory()

    def add_layer(self, class_name, param_values, attrs):
        self.layers.append(Xlayer(
            class_name,
            [xlate.as_param(val) for val in param_values],
            [xlate.as_kwparam(name, val) for name, val in xlate.as_kwparam_list(attrs)]
        ))

    def prepare_to_code(self):
        self.layers[0].kwparams.append(self.input_shape)
        self.layers.append(self.output_layer)
