

#
#  dimension(d) => dim=1
#    if not d[0].isalpha():
#      dim = max(dim, len(d.split('x')))
#
def dimension(d):
    dim = 1
    if not d[0].isalpha():
        dim = max(dim, len(d.split('x')))
    return dim


#
#  implicit_dimension(param_values) => dim=1
#    for p in param_values:
#      dim = max(dim, dimension(p))
#
def implicit_dimension(param_values):
    dim = 1
    for p in param_values:
        dim = max(dim, dimension(p))
    return dim


#
#  maybe_quote(val) => val
#    if val[0].isalpha():
#      val = f"'{val}'"
#
def maybe_quote(val):
    if val[0].isalpha():
        val = f"'{val}'"
    return val


#
#  as_param(p) => p
#    if dimension(p) > 1:
#      p = f"({p.replace('x', ',')})"
#
def as_param(p):
    if dimension(p) > 1:
        p = f"({p.replace('x', ',')})"
    return p


#
#  as_kwparam(name, val) => result
#    val = as_param(val)
#    val = maybe_quote(val)
#    result = f"{name}={val}"
#
def as_kwparam(name, val):
    val = as_param(val)
    val = maybe_quote(val)
    result = f"{name}={val}"
    return result


# Map layer type-s into class names
layer_type_class = {
    "conv": "Conv",
    "dense": "Dense",
    "dropout": "Dropout",
    "flatten": "Flatten",
    "maxpool": "MaxPooling"
}

dimensional_layers = [ "conv", "maxpool" ]


#
#  as_layer_class(layer_type, dimensionality) => class_name
#    class_name = layer_type_class[layer_type]
#    if layer_type in dimensional_layers:
#      class_name = f"{class_name}{dimensionality}D"
#
def as_layer_class(layer_type, dimensionality):
    class_name = layer_type_class[layer_type]
    if layer_type in dimensional_layers:
        class_name = f"{class_name}{dimensionality}D"
    return class_name


#
#  as_number(s) => n=0
#    if s.endswith("k"):
#      n = int(s[:-1])*1000
#    elif s.isnumeric():
#      n = int(s)
#
def as_number(s):
    n = 0
    if s.endswith("k"):
        n = int(s[:-1] ) *1000
    elif s.isnumeric():
        n = int(s)
    return n


#
#  as_shape(input_spec) => shape_str
#    shape_str = input_spec
#    data = input_spec.split()
#    for d in data:
#      if dimension(d) > 1:
#        shape_str = f"({d.replace('x', ',')})"
#      n = as_number(d)
#      if n > 0:
#        shape_str = f"({n},)"
#
def as_shape(input_spec):
    shape_str = input_spec
    data = input_spec.split()
    for d in data:
        if dimension(d) > 1:
            shape_str = f"({d.replace('x', ',')})"
        n = as_number(d)
        if n > 0:
            shape_str = f"({n},)"
    return shape_str


#
#  token_val(t, tokens) => t
#    if t.startswith("Token_"):
#      t = tokens[ int(t[6:]) ]
#
def token_val(t, tokens):
    if t.startswith("Token_"):
        t = tokens[int(t[6:])]
    return t


class_selectors = {
    "probability": ("dense", ["1"], ["sigmoid"]),
    "probabilities": ("dense", ["Token_0"], ["softmax"]),
    "float": ("dense", ["1"], [])
}


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
        if t in class_selectors:
            layer_type, params, attrs = class_selectors[t]
            param_values = [token_val(t, tokens) for t in params]
            attrs = [as_kwparam(name, val) for name, val in as_kwparam_list(attrs)]
    return as_layer_class(layer_type, implicit_dimension(param_values)), param_values, attrs


param_indices = {
    "conv": [1, 2],
    "dense": [1],
    "dropout": [1],
    "flatten": [],
    "maxpool": [1]
}

remembered_param_values = {
    "conv": [],
    "dense": [],
    "dropout": [],
    "flatten": [],
    "maxpool": []
}

attr_indices = {
    "conv": [3],
    "dense": [2],
    "dropout": [],
    "flatten": [],
    "maxpool": []
}

remembered_attr_values = {
    "conv": [],
    "dense": [],
    "dropout": [],
    "flatten": [],
    "maxpool": []
}


def memory(remembered_values, layer_type, tokens, indices):
    specified_values = [tokens[i] for i in indices if i < len(tokens)]
    if len(remembered_values) > len(specified_values):
        specified_values.extend(remembered_values[len(specified_values):])
    return specified_values


def param_memory(layer_type, tokens, indices):
    remembered_param_values[layer_type] = memory(remembered_param_values[layer_type], layer_type, tokens, indices)
    return remembered_param_values[layer_type]


def attr_memory(layer_type, tokens, indices):
    remembered_attr_values[layer_type] = memory(remembered_attr_values[layer_type], layer_type, tokens, indices)
    return remembered_attr_values[layer_type]


#
#  get_params(layer_type, tokens) => params, attrs
#    indices = param_indices[layer_type]
#    params,attrs = memory(layer_type, tokens, indices)
#
def get_params(layer_type, tokens):
    pindices = param_indices[layer_type]
    params = param_memory(layer_type, tokens, pindices)
    aindices = attr_indices[layer_type]
    attrs = attr_memory(layer_type, tokens, aindices)
    return params,attrs


attr_name = {
    "relu": "activation",
    "selu": "activation",
    "sigmoid": "activation",
    "softmax": "activation"
}


#
#  as_kwparam_list(attrs) => generate name,val
#    for attr in attrs:
#      yield attr_name[attr],attr
#
def as_kwparam_list(attrs):
    for attr in attrs:
        yield attr_name[attr],attr