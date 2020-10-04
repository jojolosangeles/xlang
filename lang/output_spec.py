from lang.dimension import implicit_dimension
from lang import args
from lang import xlate
from lang import layers


class OutputSpec:
    def __init__(self, output_spec, layer_config):
        self.output_spec = output_spec
        self.layer_class_name, self.param_values, self.attrs, self.loss_fn = self.as_output_layer_params(layer_config)

    def as_output_layer_params(self, layer_config):
        layer_type, param_values, attrs, loss_fn = None, None, None, None
        tokens = self.output_spec.split()
        for t in tokens:
            if t in layers.class_selectors:
                layer_type, params, attrs, loss_fn = layers.class_selectors[t]
                param_values = [xlate.token_val(t, tokens) for t in params]
                attrs = [args.as_kwarg_str(name, val) for name, val in args.as_kwarg_list(attrs)]

        class_name, _, is_dimensional, param_value_offsets, attr_values_start_offset, fixed_attrs = layer_config[layer_type]
        if fixed_attrs:
            attrs.extend(fixed_attrs)
        return xlate.as_layer_class(class_name, is_dimensional, implicit_dimension(param_values)), param_values, attrs, loss_fn
