"""
There are two types of parameters, corresponding to args and kwargs

When an 'arg' value is provided, it may be transformed.

Any numbers separated by 'x' become a shape-tuple string.  '3x3' becomes '(3,3)'

An number ending in 'k' or 'm' adds 3 or 6 zeros to the end of the number.  '3k' becomes '3000', '3m' becomes '3000000'

Any string value is quoted.



"""
from lang.dimension import dimension


#
#  maybe_quote(val) => val
#    if len(val) > 0 and val[0].isalpha() and not '(' in val and val != "True" and val != "False":
#      val = f"'{val}'"
#
def maybe_quote(val):
    """
    :param val: any value from model configuration
    :return: if the value is a string value, put it in quotes for code

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

    """
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
    """
    Strings that may represent numbers need to be converted to their numeric value

    :param s: any string from model configuration
    :return: a numeric value if possible, otherwise 0

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

    """
    if len(s) == 0:
        return 0
    if s.isnumeric():
        return int(s)
    elif s[:-1].isnumeric and s[-1] in conv_dict:
        return int(f"{s[:-1]}{conv_dict[s[-1]]}")
    else:
        return 0


def as_number_str(s):
    """
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

    :param s: possibly a string representing a number
    :return: string input unchanged, or as an expanded number string (e.g. "10k" becomes "10000")
    """
    n = as_number(s)
    if n > 0:
        return f"{n}"
    else:
        return s


#
#  as_arg_str(p) => p
#    if dimension(p) > 1:
#      p = f"({p.replace('x', ',')})"
#
def as_arg_str(p):
    if dimension(p) > 1:
        p = f"({p.replace('x', ',')})"
    return p


def args_as_list(param_values):
    return [as_arg_str(val) for val in param_values] if param_values else []


#
#  as_kwarg_str(name, val) => result
#    val = as_param(val)
#    val = maybe_quote(val)
#    result = f"{name}={val}"
#
def as_kwarg_str(name, val):
    val = as_arg_str(val)
    val = maybe_quote(val)
    result = f"{name}={val}"
    return result






#
#  token_val(t, tokens) => t
#    if t.startswith("Token_"):
#      t = tokens[ int(t[6:]) ]
#
def token_val(t, tokens):
    if t.startswith("Token_"):
        t = tokens[int(t[6:])]
    return t


kw_arg_values = {
    "activation": {"relu", "selu", "sigmoid", "softmax"},
    "kernel_regularizer": {"l1", "l2"},
    "weights": {"imagenet"}
}


def kw_name(val):
    """
    Convert a string value into it's corresponding parameter name+
    :param val:
    :return:
    """
    for kwname in kw_arg_values:
        if val in kw_arg_values[kwname]:
            return kwname
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
def as_kwarg_list(attrs):
    for i, attr in enumerate(attrs):
        name = kw_name(attr)
        val = attr_val(attr, attrs[(i + 1):])
        if name:
            yield name, val
        elif '=' in attr:
            data = attr.split('=')
            yield data[0], data[1]
