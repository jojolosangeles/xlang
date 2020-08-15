# python -m doctest -v dimension.py
"""
Some keras classes have dimension-specific variations, and end with "1D", "2D", "3D"

In xlang, the dimension is derived from the parameters.  In some cases, the dimension is not
available in the parameters, and the most recent dimension value needs to be used instead.

The default value for dimension is 1.

>>> dimension("a")
1
>>> dimension("x")
1
>>> dimension("3")
1
>>> dimension("3x3")
2
>>> dimension("150x3x3")
3


The implicit_dimension is derived from the parameters.

This assumes that "x" is special, and is a delimiter between dimensions

In one case, the dimension is based on a previous command.  For example, GlobalMaxPooling 1D/2D/3D does NOT
take any parameters.  The dimension in this case is based on the most recently calculated dimension.

>>> implicit_dimension([])
1
>>> implicit_dimension(["this", "is", "a", "2x2", "test"])
2
>>> implicit_dimension(["asdf", "3x3x5"])
3
>>> implicit_dimension([])
3
>>> reset_default_dimension()
>>> implicit_dimension([])
1
"""
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


last_explicit_dimension = 1


def reset_default_dimension():
    global last_explicit_dimension
    last_explicit_dimension = 1


#
#  implicit_dimension(param_values) => dim=1
#    global last_explicit_dimension
#    for p in param_values:
#      dim = max(dim, dimension(p))
#    if dim > 1:
#      last_explicit_dimension = dim
#    if len(param_values) == 0:
#      dim = last_explicit_dimension
#
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
