from xlate import xlate

class Xlayer:
    def __init__(self, class_name, params, kwparams):
        self.class_name = class_name
        self.params = params
        self.kwparams = kwparams


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

    def add_layer(self, layer_type, param_values, attrs):
        self.layers.append(Xlayer(
            xlate.as_layer_class(layer_type, xlate.implicit_dimension(param_values)),
            [xlate.as_param(val) for val in param_values],
            [xlate.as_kwparam(name, val) for name, val in xlate.as_kwparam_list(attrs)]
        ))

    def prepare_to_code(self):
        self.layers[0].kwparams.append(self.input_shape)
        self.layers.append(self.output_layer)
