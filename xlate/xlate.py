
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
last_explicit_dimension = 1


def reset_dimensions():
    global last_explicit_dimension
    last_explicit_dimension = 1


def implicit_dimension(param_values):
    global last_explicit_dimension
    dim = 1
    for p in param_values:
        dim = max(dim, dimension(p))
    if dim > 1:
        last_explicit_dimension = dim
    if len(param_values) == 0:
        dim = last_explicit_dimension
    return dim


#
#  maybe_quote(val) => val
#    if val[0].isalpha():
#      val = f"'{val}'"
#
def maybe_quote(val):
    if val[0].isalpha() and not '(' in val and val != "True" and val != "False":
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


def params_as_code(param_values):
    return [as_param(val) for val in param_values]


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
dimensional_layers = [ "conv", "maxpool" ]


#
#  as_layer_class(layer_type, dimensionality) => class_name
#    class_name = layer_type_class[layer_type]
#    if layer_type in dimensional_layers:
#      class_name = f"{class_name}{dimensionality}D"
#
def as_layer_class(class_name, is_dimensional, dimensionality, tokens=[]):
    if is_dimensional:
        class_name = f"{class_name}{dimensionality}D"
    if class_name.startswith("Token_"):
        class_name = tokens[int(class_name[6:])]
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
    if input_spec == "features":
        input_spec = "n_features"
    shape_str = f"({input_spec},)"
    data = input_spec.split()
    if data[0] == 'time':
        shape_str = f"(None,{data[1]})"
    else:
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





attr_values = {
    "activation": { "relu", "selu", "sigmoid", "softmax" },
    "kernel_regularizer": { "l1", "l2" },
    "weights": { "imagenet" }
}


def attr_name(val):
    for attr_name in attr_values:
        if val in attr_values[attr_name]:
            return attr_name
    return ""


def attr_val(val, possible_params):
    if val == "l2":
        return f"regularizers.l2({possible_params[0]})"
    return val


#
#  as_kwparam_list(attrs) => generate name,val
#    for attr in attrs:
#      yield attr_name[attr],attr
#
def as_kwparam_list(attrs):
    for i,attr in enumerate(attrs):
        name = attr_name(attr)
        val = attr_val(attr, attrs[(i+1):])
        if name:
            yield name, val
        elif '=' in attr:
            data = attr.split('=')
            yield data[0], data[1]
