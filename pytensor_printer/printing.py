from typing import Any

import sympy as sp
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
from functools import partial
import pytensor
import pytensor.tensor as pt
import pytensor.scalar as ps

from pytensor.tensor.elemwise import Elemwise, DimShuffle

mapping = {
    sp.Add: pt.add,
    sp.Mul: pt.mul,
    sp.Abs: pt.abs,
    sp.sign: pt.sgn,
    sp.ceiling: pt.ceil,
    sp.floor: pt.floor,
    sp.log: pt.log,
    sp.exp: pt.exp,
    sp.sqrt: pt.sqrt,
    sp.cos: pt.cos,
    sp.acos: pt.arccos,
    sp.sin: pt.sin,
    sp.asin: pt.arcsin,
    sp.tan: pt.tan,
    sp.atan: pt.arctan,
    sp.atan2: pt.arctan2,
    sp.cosh: pt.cosh,
    sp.acosh: pt.arccosh,
    sp.sinh: pt.sinh,
    sp.asinh: pt.arcsinh,
    sp.tanh: pt.tanh,
    sp.atanh: pt.arctanh,
    sp.re: pt.real,
    sp.im: pt.imag,
    sp.arg: pt.angle,
    sp.erf: pt.erf,
    sp.gamma: pt.gamma,
    sp.loggamma: pt.gammaln,
    sp.Pow: pt.pow,
    sp.Eq: pt.eq,
    sp.Ne: pt.neq,
    sp.StrictGreaterThan: pt.gt,
    sp.StrictLessThan: pt.lt,
    sp.LessThan: pt.le,
    sp.GreaterThan: pt.ge,
    sp.And: pt.bitwise_and,  # bitwise
    sp.Or: pt.bitwise_or,  # bitwise
    sp.Not: pt.invert,  # bitwise
    sp.Xor: pt.bitwise_xor,  # bitwise
    sp.Max: pt.maximum,  # Sympy accept >2 inputs, Pytensor only 2
    sp.Min: pt.minimum,  # Sympy accept >2 inputs, Pytensor only 2
    sp.conjugate: pt.conj,
    sp.core.numbers.ImaginaryUnit: lambda: pt.complex(0, 1),
    # Matrices
    sp.MatAdd: Elemwise(ps.add),
    sp.HadamardProduct: Elemwise(ps.mul),
    sp.Trace: pt.linalg.trace,
    sp.Determinant: pt.linalg.det,
    sp.Inverse: pt.linalg.inv,
    sp.Transpose: DimShuffle((False, False), [1, 0]),
}


class PytensorPrinter(Printer):
    """ Code printer which creates Pytensor symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.aesara_code` or :func:`.aesara_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Pytensor variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Pytensor variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = "_pytensor"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        """ Get the cache key for a SymPy object.

        Parameters
        ==========

        s : sympy.core.basic.Basic
            SymPy object to get key for.

        name : str
            Name of object, if it does not have a ``name`` attribute.
        """

        if name is None:
            name = s.name

        return name, type(s), s.args, dtype, broadcastable

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        """
        Get the Pytensor variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = pt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        # Integers already taken care of below, interpret as float
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        return pt.stacklists([
            [self._print(arg, **kwargs) for arg in L]
            for L in X.tolist()
        ])

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = pt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = pt.dot(result, children[0])
        else:
            raise NotImplementedError('''Only non-negative integer
           powers of matrices can be handled by Pytensor at the moment''')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs)
                   for c in range(ncols)]
                  for r in range(nrows)]
        return pt.join(0, *[pt.join(1, *row) for row in blocks])

    def _print_slice(self, expr, **kwargs):
        return slice(*[self._print(i, **kwargs)
                       if isinstance(i, sp.Basic) else i
                       for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        return 3.141592653589793

    def _print_Piecewise(self, expr, **kwargs):
        import numpy as np
        e, cond = expr.args[0].args  # First condition and corresponding value

        # Print conditional expression and value for first condition
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)

        # One condition only
        if len(expr.args) == 1:
            # Return value if condition else NaN
            return pt.switch(p_cond, p_e, np.nan)

        # Return value_1 if condition_1 else evaluate remaining conditions
        p_remaining = self._print(sp.Piecewise(*expr.args[1:]), **kwargs)
        return pt.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        return pt.true_div(self._print(expr.p, **kwargs),
                           self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        return self._print(sp.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        from pytensor.gradient import Rop

        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = Rop(rv, var, pt.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        """ Convert a SymPy expression to a Pytensor graph variable.

        The ``dtypes`` and ``broadcastable`` arguments are used to specify the
        data type, dimension, and broadcasting behavior of the Pytensor variables
        corresponding to the free symbols in ``expr``. Each is a mapping from
        SymPy symbols to the value of the corresponding argument to
        ``pytensor.tensor.var.TensorVariable``.

        See the corresponding `documentation page`__ for more information on
        broadcasting in Pytensor.


        .. __: https://pytensor.readthedocs.io/en/latest/reference/tensor/broadcastable.html#broadcasting

        Parameters
        ==========

        expr : sympy.core.expr.Expr
            SymPy expression to print.

        dtypes : dict
            Mapping from SymPy symbols to Pytensor datatypes to use when creating
            new Pytensor variables for those symbols. Corresponds to the ``dtype``
            argument to ``pytensor.tensor.var.TensorVariable``. Defaults to ``'floatX'``
            for symbols not included in the mapping.

        broadcastable : dict
            Mapping from SymPy symbols to the value of the ``broadcastable``
            argument to ``pytensor.tensor.var.TensorVariable`` to use when creating Pytensor
            variables for those symbols. Defaults to the empty tuple for symbols
            not included in the mapping (resulting in a scalar).

        Returns
        =======

        pytensor.graph.basic.Variable
            A variable corresponding to the expression's value in a Pytensor
            symbolic expression graph.

        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def pytensor_code(expr, cache=None, **kwargs):
    """
    Convert a SymPy expression into a Pytensor graph variable.

    Parameters
    ==========

    expr : sympy.core.expr.Expr
        SymPy expression object to convert.

    cache : dict
        Cached Pytensor variables (see :class:`PytensorPrinter.cache
        <PytensorPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.PytensorPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.PytensorPrinter.doprint`.

    Returns
    =======

    pytensor.graph.basic.Variable
        A variable corresponding to the expression's value in a Pytensor symbolic
        expression graph.

    """
    if cache is None:
        cache = global_cache

    return PytensorPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    r"""
    Get value of ``broadcastables`` argument to :func:`.pytensor_code` from
    keyword arguments to :func:`.pytensor_function`.

    Included for backwards compatibility.

    Parameters
    ==========

    inputs
        Sequence of input symbols.

    dim : int
        Common number of dimensions for all inputs. Overrides other arguments
        if given.

    dims : dict
        Mapping from input symbols to number of dimensions. Overrides
        ``broadcastables`` argument if given.

    broadcastables : dict
        Explicit value of ``broadcastables`` argument to
        :meth:`.PytensorPrinter.doprint`. If not None function will return this value unchanged.

    Returns
    =======
    dict
        Dictionary mapping elements of ``inputs`` to their "broadcastables"
        values (tuple of ``bool``\ s).
    """
    if dim is not None:
        return {s: (False,) * dim for s in inputs}

    if dims is not None:
        maxdim = max(dims.values())
        return {
            s: (False,) * d + (True,) * (maxdim - d)
            for s, d in dims.items()
        }

    if broadcastables is not None:
        return broadcastables

    return {}


def pytensor_function(inputs, outputs, scalar=False, *,
                    dim=None, dims=None, broadcastables=None, **kwargs):
    """
    Create a Pytensor function from SymPy expressions.

    The inputs and outputs are converted to Pytensor variables using
    :func:`.pytensor_code` and then passed to ``pytensor.function``.

    Parameters
    ==========

    inputs
        Sequence of symbols which constitute the inputs of the function.

    outputs
        Sequence of expressions which constitute the outputs(s) of the
        function. The free symbols of each expression must be a subset of
        ``inputs``.

    scalar : bool
        Convert 0-dimensional arrays in output to scalars. This will return a
        Python wrapper function around the Pytensor function object.

    cache : dict
        Cached Pytensor variables (see :class:`PytensorPrinter.cache
        <PytensorPrinter>`). Defaults to the module-level global cache.

    dtypes : dict
        Passed to :meth:`.PytensorPrinter.doprint`.

    broadcastables : dict
        Passed to :meth:`.PytensorPrinter.doprint`.

    dims : dict
        Alternative to ``broadcastables`` argument. Mapping from elements of
        ``inputs`` to integers indicating the dimension of their associated
        arrays/tensors. Overrides ``broadcastables`` argument if given.

    dim : int
        Another alternative to the ``broadcastables`` argument. Common number of
        dimensions to use for all arrays/tensors.
        ``pytensor_function([x, y], [...], dim=2)`` is equivalent to using
        ``broadcastables={x: (False, False), y: (False, False)}``.

    Returns
    =======
    callable
        A callable object which takes values of ``inputs`` as positional
        arguments and returns an output array for each of the expressions
        in ``outputs``. If ``outputs`` is a single expression the function will
        return a Numpy array, if it is a list of multiple expressions the
        function will return a list of arrays. See description of the ``squeeze``
        argument above for the behavior when a single output is passed in a list.
        The returned object will either be an instance of
        ``pytensor.compile.function.types.Function`` or a Python wrapper
        function around one. In both cases, the returned value will have a
        ``pytensor_function`` attribute which points to the return value of
        ``pytensor.function``.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from pytensor_printer.printing import pytensor_function

    A simple function with one input and one output:

    >>> f1 = pytensor_function([x], [x**2 - 1], scalar=True)
    >>> f1(3)
    8.0

    A function with multiple inputs and one output:

    >>> f2 = pytensor_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
    >>> f2(3, 4, 2)
    5.0

    A function with multiple inputs and multiple outputs:

    >>> f3 = pytensor_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
    >>> f3(2, 3)
    [13.0, -5.0]

    See also
    ========

    dim_handling

    """

    # Pop off non-pytensor keyword args
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})

    broadcastables = dim_handling(
        inputs, dim=dim, dims=dims, broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(pytensor_code, cache=cache, dtypes=dtypes,
                   broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    # fix constant expressions as variables
    toutputs = [output if isinstance(output, pytensor.graph.basic.Variable) else pt.as_tensor_variable(output) for
                output in toutputs]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile pytensor func
    func = pytensor.function(tinputs, toutputs, **kwargs)

    is_0d = [o.variable.broadcastable == () for o in func.outputs]

    # No wrapper required
    if not scalar or not any(is_0d):
        func.pytensor_function = func
        return func

    # Create wrapper to convert 0-dimensional outputs to scalars
    def wrapper(*args):
        out = func(*args)
        # out can be array(1.0) or [array(1.0), array(2.0)]

        if is_sequence(out):
            return [o[()] if is_0d[i] else o for i, o in enumerate(out)]
        else:
            return out[()]

    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.pytensor_function = func
    return wrapper
