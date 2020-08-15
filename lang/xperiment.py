from lang import args
from lang import layers
from lang import xlate
from lang.dimension import implicit_dimension, reset_default_dimension


class Xlayer:
    def __init__(self, class_name, params, kwparams):
        self.class_name = class_name
        self.params = params
        self.kwparams = kwparams


layer_config = {
    # layer_type: ( class_name, is_dimensional param_value_offsets, attr_values_start_offset, fixed_params )
    "bidirectional": ("layers.Bidirectional", False, [], 1, None),
    "conv": ("layers.Conv", True, [1, 2], 3, None),
    "convbase": ("Token_1", False, [], 2, "include_top=False"),
    "dense": ("layers.Dense", False, [1], 2, None),
    "embed": ("layers.Embedding", False, [1], 2, None),
    "dropout": ("layers.Dropout", False, [1], None, None),
    "flatten": ("layers.Flatten", False, [], None, None),
    "global": ("layers.GlobalMaxPooling", True, [], None, None),
    "global_average_pool": ("layers.GlobalAveragePooling", True, [], None, None),
    "GRU": ("layers.GRU", False, [1], 2, None),
    "LSTM": ("layers.LSTM", False, [1], None, None),
    "maxpool": ("layers.MaxPooling", True, [1], None, None),
    "separableconv": ("layers.SeparableConv", True, [1, 2], 3, None),
    "simpleRNN": ("layers.SimpleRNN", False, [1], None, None)
}


class LayerMemory:
    def __init__(self):
        self.remembered_param_values = {}
        self.remembered_attr_values = {}
        for layer_type in layer_config:
            self.remembered_param_values[layer_type] = []
            self.remembered_attr_values[layer_type] = []

    def get_params(self, layer_type, tokens):
        class_name, is_dimensional, param_indices, attr_start_index, custom_attrs = layer_config[layer_type]
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
#    model_builder_params() => s=""
#      if input == "features":
#        s = n_features
#
#
#  as_output_layer_params(output_spec) => class_name,param_values,attrs
#    tokens = output_spec.split()
#    for t in tokens:
#      if t in class_selectors:
#        layer_type, params, attrs = class_selectors[t]
#        param_values = [ token_val(t, tokens) for t in tokens ]
#
def as_output_layer_params(output_spec):
    layer_type, param_values, attrs = None, None, None
    tokens = output_spec.split()
    for t in tokens:
        if t in layers.class_selectors:
            layer_type, params, attrs = layers.class_selectors[t]
            param_values = [xlate.token_val(t, tokens) for t in params]
            attrs = [args.as_kwarg_str(name, val) for name, val in args.as_kwarg_list(attrs)]

    class_name, is_dimensional, param_value_offsets, attr_values_start_offset, fixed_attrs = layer_config[layer_type]
    if fixed_attrs:
        attrs.extend(fixed_attrs)
    return xlate.as_layer_class(class_name, is_dimensional, implicit_dimension(param_values)), param_values, attrs


class Xperiment:
    def __init__(self, model_name, input_spec, output_spec):
        self.model_name = model_name
        self.input_spec = input_spec
        self.output_spec = output_spec
        reset_default_dimension()
        self.input_shape = args.as_kwarg_str('input_shape', xlate.as_shape(input_spec))
        self.output_layer = Xlayer(*as_output_layer_params(output_spec))
        self.lines = []
        self.layers = []
        self.layer_memory = LayerMemory()

    def add_layer(self, class_name, param_values, attrs):
        xlayer = Xlayer(
            class_name,
            args.args_as_list(param_values),
            [args.as_kwarg_str(name, val) for name, val in args.as_kwarg_list(attrs)]
        )
        self.layers.append(xlayer)
        return xlayer

    def prepare_to_code(self):
        # stacked GRU layers require return_sequences=True
        for i,layer in enumerate(self.layers[:-1]):
            if "GRU" in layer.class_name and "GRU" in self.layers[i+1].class_name:
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
