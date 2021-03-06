from lang.dimension import dimension
from lang.args import as_number


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


def as_num_words(input_spec):
    data = input_spec.split()
    result = 0
    for d in data:
        n = as_number(d)
        if n > result:
            result = n
    return None if result == 0 else result

#
#  token_val(t, tokens) => t
#    if t.startswith("Token_"):
#      t = tokens[ int(t[6:]) ]
#
def token_val(t, tokens=None):
    if tokens and t.startswith("Token_"):
        t = tokens[int(t[6:])]
    return t



