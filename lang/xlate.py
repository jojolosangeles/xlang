"""
There are two types of parameters, corresponding to args and kwargs

When an 'arg' value is provided, it may be transformed.

Any numbers separated by 'x' become a shape-tuple string.  '3x3' becomes '(3,3)'

An number ending in 'k' or 'm' adds 3 or 6 zeros to the end of the number.  '3k' becomes '3000', '3m' becomes '3000000'

Any string value is quoted.

>>> maybe_quote("3")
'3'
>>> maybe_quote("relu")
"'relu'"
>>> maybe_quote("fn(3)")
'fn(3)'
>>> maybe_quote("True")
'True'
>>> maybe_quote("False")
'False'
>>> maybe_quote("true")
"'true'"
>>> maybe_quote("false")
"'false'"
>>> maybe_quote("")
''

Strings that represent numbers need to be converted to their full number strings.

>>> as_number("3")
3
>>> as_number("10k")
10000
>>> as_number("3m")
3000000
>>> as_number("haha")
0
>>> as_number("")
0

>>> as_number_str("3")
'3'
>>> as_number_str("10k")
'10000'
>>> as_number_str("3m")
'3000000'
>>> as_number_str("haha")
'haha'
>>> as_number_str("")
''

"""
from lang.dimension import dimension


#
#  maybe_quote(val) => val
#    if len(val) > 0 and val[0].isalpha() and not '(' in val and val != "True" and val != "False":
#      val = f"'{val}'"
#
def maybe_quote(val):
    # if the value could be a word, could not be a function or bool val, add quotes
    if len(val) > 0 and val[0].isalpha() and not '(' in val and val != "True" and val != "False":
        val = f"'{val}'"
    return val


#
#  as_number_str(s) => s
#    if len(s) and s[-1] in conv_dict:
#      return f"{s[:-1]}{conv_dict[s[-1]]}"
#    else:
#      return s
#
conv_dict = {"m": "000000", "k": "000"}


def as_number(s):
    if len(s) == 0:
        return 0
    if s.isnumeric():
        return int(s)
    elif s[:-1].isnumeric and s[-1] in conv_dict:
        return int(f"{s[:-1]}{conv_dict[s[-1]]}")
    else:
        return 0


def as_number_str(s):
    n = as_number(s)
    if n > 0:
        return f"{n}"
    else:
        return s


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
dimensional_layers = ["conv", "maxpool"]


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
    "activation": {"relu", "selu", "sigmoid", "softmax"},
    "kernel_regularizer": {"l1", "l2"},
    "weights": {"imagenet"}
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
    for i, attr in enumerate(attrs):
        name = attr_name(attr)
        val = attr_val(attr, attrs[(i + 1):])
        if name:
            yield name, val
        elif '=' in attr:
            data = attr.split('=')
            yield data[0], data[1]
