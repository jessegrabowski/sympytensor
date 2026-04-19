from functools import partial
from typing import Any

import pytensor
import pytensor.scalar as ps
import pytensor.tensor as pt
import sympy as sp
from pytensor.raise_op import CheckAndRaise
from pytensor.sparse.variable import SparseVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.variable import TensorVariable
from sympy.printing.printer import Printer
from pytensor import config
import numpy as np


mapping = {
    # Numbers
    sp.core.numbers.ImaginaryUnit: lambda: pt.complex(0, 1),
    # elemwise funcs
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
    # Matrices
    sp.MatAdd: Elemwise(ps.add),
    sp.HadamardProduct: Elemwise(ps.mul),
    sp.Trace: pt.linalg.trace,
    sp.Determinant: pt.linalg.det,
    sp.Inverse: pt.linalg.inv,
    sp.Transpose: pt.matrix_transpose,
}


def dod_to_csr(
    dod: dict[int, dict[int, Any]], shape: tuple[int, int]
) -> tuple[list, list[int], list[int], tuple[int, int]]:
    """Convert a dictionary-of-dictionaries sparse representation to compressed sparse row (CSR).

    Parameters
    ----------
    dod : dict of int to dict of int to value
        Sparse data in SymPy's dictionary-of-dictionaries format.
    shape : tuple of int
        Matrix shape ``(n_rows, n_cols)``.

    Returns
    -------
    data : list
        Non-zero values in row-major order.
    indices : list of int
        Column indices corresponding to each entry in `data`.
    indptr : list of int
        Row pointer array of length ``n_rows + 1``.
    shape : tuple of int
        The input `shape`, passed through unchanged.
    """
    n_rows, n_cols = shape

    data = []
    idxs = []
    pointers = [0]

    for row in range(n_rows):
        if row in dod:
            for col in sorted(dod[row].keys()):
                data.append(dod[row][col])
                idxs.append(col)
        pointers.append(len(data))

    return data, idxs, pointers, shape


class PytensorPrinter(Printer):
    """Code printer that converts SymPy expressions into PyTensor symbolic expression graphs.

    Parameters
    ----------
    cache : dict
        Cache dictionary to use.  If ``None`` (default) will use the global cache.  To create a printer which does
        not depend on or alter global state pass an empty dictionary.  Note: the dictionary is not copied on
        initialization of the printer and will be updated in-place, so using the same dict object when creating
        multiple printers or making multiple calls to :func:`as_tensor` or :func:`pytensor_function` means the cache
        is shared between all these applications.
    """

    printmethod = "_pytensor"

    def __init__(self, *args, **kwargs):
        self.cache = kwargs.pop("cache", {})
        super().__init__(*args, **kwargs)

    def _print(self, expr, **kwargs):
        """Override base _print to add fast path for numeric types."""
        if isinstance(expr, sp.Integer):
            return expr.p

        if isinstance(expr, sp.Basic) and expr.is_number and expr.is_real is not False:
            return float(expr.evalf())

        return super()._print(expr, **kwargs)

    def _get_key(
        self,
        s: sp.Basic,
        name: str | None = None,
        dtype: str | None = None,
        broadcastable: tuple | None = None,
    ) -> tuple:
        """Get the cache key for a SymPy object.

        Parameters
        ----------
        s : sympy.Basic
            SymPy object to get key for.
        name : str, optional
            Name of object, if it does not have a ``name`` attribute.
        dtype : str, optional
            PyTensor dtype string.
        broadcastable : tuple, optional
            Broadcastable pattern.
        """

        if name is None:
            name = s.name

        return name, type(s), s.args, dtype, broadcastable

    def _get_or_create(
        self,
        s: sp.Basic,
        name: str | None = None,
        dtype: str | None = None,
        broadcastable: tuple | None = None,
        shape: tuple | None = None,
    ) -> TensorVariable:
        """Get the PyTensor variable for a SymPy symbol from the cache, or create it if it does not exist."""

        # Defaults
        if name is None:
            name = s.name
        if dtype is None:
            dtype = "floatX"
        if broadcastable is None:
            broadcastable = ()
        if shape is None:
            shape = ()

        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)

        if key in self.cache:
            return self.cache[key]

        value = pt.tensor(name=name, dtype=dtype, broadcastable=broadcastable, shape=shape)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        dtype = kwargs.get("dtypes", {}).get(s)
        bc = kwargs.get("broadcastables", {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        name = str(type(s)) + "_" + str(s.args[0])
        dtype = kwargs.get("dtypes", {}).get(s)
        bc = kwargs.get("broadcastables", {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        try:
            op = mapping[type(expr)]
        except KeyError:
            raise NotImplementedError(
                f"SymPy type {type(expr).__name__} has no PyTensor mapping. "
                f"Add an entry to `mapping` or implement `_print_{type(expr).__name__}`."
            ) from None
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_MatrixSymbol(self, X, **kwargs):
        dtype = kwargs.get("dtypes", {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_ZeroMatrix(self, expr, **kwargs):
        rows, cols = expr.shape
        return pt.zeros((int(rows), int(cols)), dtype=pytensor.config.floatX)

    def _print_Idx(self, i, **kwargs):
        dtype = kwargs.get("dtypes", {}).get(i)
        if dtype is None:
            dtype = "int32"

        bc = kwargs.get("broadcastables", {}).get(i)
        if i.lower is None and i.upper is None:
            return self._get_or_create(i, dtype=dtype, broadcastable=bc)
        elif i.lower is None:
            valid_range = (0, int(i.upper.evalf() + 1))
        else:
            valid_range = (int(i.lower.evalf()), int(i.upper.evalf() + 1))

        i = self._get_or_create(i, dtype=dtype, broadcastable=bc)
        all_true_scalar = pt.all([pt.ge(i, valid_range[0]), pt.lt(i, valid_range[1])])
        msg = f"Index {i.name} out of valid range {valid_range[0]} - {valid_range[1]}"

        return CheckAndRaise(IndexError, msg)(i, all_true_scalar)

    def _partition_matrix_elements(self, X: sp.matrices.dense.DenseMatrix, **kwargs):
        """Partition matrix entries into a numeric base array and symbolic overlay lists.

        Parameters
        ----------
        X : sympy.matrices.dense.DenseMatrix
            SymPy dense matrix.
        **kwargs
            Additional arguments forwarded to :meth:`_print` for symbolic elements.

        Returns
        -------
        base : numpy.ndarray
            Array with numeric entries filled in, zeros elsewhere.
        sym_rows : list of int
            Row indices of symbolic entries.
        sym_cols : list of int
            Column indices of symbolic entries.
        sym_values : list of TensorVariable
            Printed PyTensor expressions for each symbolic entry.
        """
        nrows, ncols = X.shape
        base = np.zeros((nrows, ncols), dtype=config.floatX)
        sym_rows = []
        sym_cols = []
        sym_values = []

        for idx, val in enumerate(X.flat()):
            row, col = divmod(idx, ncols)
            if isinstance(val, sp.Basic) and val.is_number:
                base[row, col] = float(val.evalf())
            elif val != 0:
                sym_rows.append(row)
                sym_cols.append(col)
                sym_values.append(self._print(val, **kwargs))

        return base, sym_rows, sym_cols, sym_values

    def _print_DenseMatrix_setsubtensor(self, X: sp.matrices.dense.DenseMatrix, **kwargs) -> TensorVariable:
        """Convert dense matrix to PyTensor using a constant base with symbolic overlays.

        Fills all numeric values into a numpy array upfront, then applies a single ``set()`` operation for any
        symbolic entries.  This minimizes graph nodes to *O(n_symbolic)* instead of *O(n_nonzero)*.
        """
        base, sym_rows, sym_cols, sym_values = self._partition_matrix_elements(X, **kwargs)
        X_pt = pt.as_tensor_variable(base)

        if not sym_values:
            return X_pt

        return X_pt[pt.as_tensor(sym_rows), pt.as_tensor(sym_cols)].set(sym_values)

    def _print_DenseMatrix(self, X: sp.matrices.dense.DenseMatrix, **kwargs) -> TensorVariable:
        """Convert a SymPy dense matrix to a PyTensor variable.

        Parameters
        ----------
        X : sympy.matrices.dense.DenseMatrix
            SymPy dense matrix to convert.
        **kwargs
            Additional arguments passed to element printers (e.g. `dtypes`, `broadcastables`).

        Returns
        -------
        result : TensorVariable
            PyTensor variable representing the matrix.
        """
        try:
            elements = list(X.flat())
            if all(isinstance(elem, sp.Basic) and elem.is_number for elem in elements):
                arr = np.array([float(elem.evalf()) for elem in elements], dtype=config.floatX)
                return pt.as_tensor_variable(arr.reshape(X.shape))
        except (AttributeError, ValueError, TypeError):
            pass

        return self._print_DenseMatrix_setsubtensor(X, **kwargs)

    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_SparseMatrix(self, X: sp.SparseMatrix, **kwargs) -> SparseVariable:
        """Convert a SymPy sparse matrix to a PyTensor CSR sparse variable.

        Optimizes for the all-numeric case by bypassing printer dispatch.
        """
        dod = X.todod()
        data, idxs, pointers, shape = dod_to_csr(dod, shape=X.shape)

        if all(isinstance(d, sp.Basic) and d.is_number for d in data):
            data = [float(d.evalf()) for d in data]
        else:
            data = [self._print(d, **kwargs) for d in data]

        return pytensor.sparse.CSR(data, idxs, pointers, shape)

    _print_ImmutableSparseMatrix = _print_MutableSparseMatrix = _print_SparseMatrix

    def _print_IndexedBase(self, X, **kwargs):
        dtype = kwargs.get("dtypes", {}).get(X)
        shape = kwargs.get("shapes", None)
        bc = kwargs.get("broadcastable", None)

        if shape is None and bc is None:
            # No explicit shape/broadcastable provided — infer from the SymPy
            # object.  Use its declared shape when available, otherwise assume
            # a 1-d tensor with unknown length.
            if X.shape is not None:
                shape = tuple(int(x) if x is not None else None for x in X.shape)
            else:
                shape = (None,)
            # For IndexedBase the broadcastable signature matches the shape:
            # concrete dimensions are non-broadcastable (int), unknown ones are
            # None, which is exactly what _get_or_create expects.
            bc = shape
        elif shape is None:
            # Broadcastable was provided but shape was not — mirror it so both
            # arguments stay consistent.
            shape = bc

        return self._get_or_create(X, dtype=dtype, broadcastable=bc, shape=shape)

    def _print_Indexed(self, X, **kwargs):
        # Infer the shape of the indexed base.
        shape = X.base.shape
        if shape is not None:
            shape = tuple([int(x) if x is not None else None for x in X.shape])
        else:
            shape = (None,) * len(X.indices)

        bc = kwargs.get("broadcastables", {}).get(X.base, None)
        if bc is None:
            bc = shape
        indices = tuple([self._print(x) for x in X.indices])
        base = self._print(X.base, shape=shape, broadcastable=bc, **kwargs)

        return base[indices]

    @staticmethod
    def _build_reduction_slices(sum_args):
        """Build ``{index_name: slice}`` from Sum/Product limit triples."""
        return {var.name: slice(int(start), int(stop) + 1) for var, start, stop in sum_args}

    @staticmethod
    def _reduction_axes(dims_pt: list[TensorVariable], slice_dict: dict[str, slice]) -> tuple[tuple, tuple | None]:
        """Return ``(out_idx, reduce_axis)`` for a reduction.

        Parameters
        ----------
        dims_pt : list of TensorVariable
            PyTensor variables corresponding to the summand's index dimensions.
        slice_dict : dict of str to slice
            Mapping from index names to slices (built by :meth:`_build_reduction_slices`).
        """
        out_idx = []
        reduce_axis = []
        output_axis = 0
        for idx in dims_pt:
            val = slice_dict.get(idx.name, idx)
            out_idx.append(val)
            if isinstance(val, slice) and idx.name in slice_dict:
                reduce_axis.append(output_axis)
            if isinstance(val, slice):
                output_axis += 1
        return tuple(out_idx), tuple(reduce_axis) or None

    def _print_reduction(self, X, op: str = "sum", **kwargs) -> TensorVariable:
        """Convert SymPy Sum/Product with indexed summands to PyTensor reduction.

        Handles expressions like ``Sum(x[i, j, k], (i, 0, 10), (j, 0, 5))`` by:

        1. Building a slice dict from the reduction limits.
        2. Printing the summand to obtain graph inputs.
        3. Slicing the base array and reducing over the appropriate axes.

        Parameters
        ----------
        X : sympy.concrete.expr_with_limits.ExprWithLimits
            SymPy Sum or Product expression.
        op : str
            Reduction operation: ``"sum"`` or ``"prod"``.
        **kwargs
            Additional arguments passed to element printers.

        Returns
        -------
        result : TensorVariable
            PyTensor reduction result.
        """
        summand, *sum_args = X.args
        slice_dict = self._build_reduction_slices(sum_args)

        summand_pt = self._print(summand, **kwargs)
        inputs = list(pytensor.graph.graph_inputs([summand_pt]))
        inputs_by_name = {inp.name: inp for inp in inputs}

        # Preserve original index order from summand (graph traversal order is arbitrary)
        dims_pt = [inputs_by_name[idx.name] for idx in summand.indices]
        base = inputs_by_name[summand.base.name]

        out_idx, reduce_axis = self._reduction_axes(dims_pt, slice_dict)

        match op:
            case "sum":
                return pt.sum(base[out_idx], axis=reduce_axis)
            case "prod":
                return pt.prod(base[out_idx], axis=reduce_axis)
            case _:
                raise NotImplementedError(f"Unsupported reduction operation '{op}'. Supported: 'sum', 'prod'.")

    def _print_Sum(self, X, **kwargs) -> TensorVariable:
        """Convert SymPy Sum to PyTensor sum reduction."""
        return self._print_reduction(X, op="sum", **kwargs)

    def _print_Product(self, X, **kwargs) -> TensorVariable:
        """Convert SymPy Product to PyTensor prod reduction."""
        return self._print_reduction(X, op="prod", **kwargs)

    def _print_MatMul(self, expr, **kwargs):
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = pt.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        base_pt = self._print(expr.args[0], **kwargs)
        exp_val = self._print(expr.args[1], **kwargs)
        if not isinstance(exp_val, int) or exp_val < 1:
            raise NotImplementedError("Only positive integer matrix powers are supported by PyTensor.")
        result = base_pt
        for _ in range(exp_val - 1):
            result = pt.dot(result, base_pt)
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        nrows, ncols = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs) for c in range(ncols)] for r in range(nrows)]
        return pt.join(0, *[pt.join(1, *row) for row in blocks])

    def _print_slice(self, expr, **kwargs):
        return slice(
            *[self._print(i, **kwargs) if isinstance(i, sp.Basic) else i for i in (expr.start, expr.stop, expr.step)]
        )

    def _print_Piecewise(self, expr, **kwargs):
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

    def doprint(
        self,
        expr: sp.Expr,
        dtypes: dict[sp.Symbol, str] | None = None,
        broadcastables: dict[sp.Symbol, tuple[bool, ...]] | None = None,
    ) -> TensorVariable:
        """Convert a SymPy expression to a PyTensor graph variable.

        The `dtypes` and `broadcastables` arguments specify the data type, dimension, and broadcasting behavior of the
        PyTensor variables corresponding to the free symbols in `expr`.  Each is a mapping from SymPy symbols to the
        value of the corresponding argument to :class:`pytensor.tensor.variable.TensorVariable`.

        See the `PyTensor broadcasting docs`__ for more information.

        .. __: https://pytensor.readthedocs.io/en/latest/reference/tensor/broadcastable.html#broadcasting

        Parameters
        ----------
        expr : sympy.Expr
            SymPy expression to print.
        dtypes : dict of sympy.Symbol to str, optional
            Mapping from SymPy symbols to PyTensor dtype strings.  Defaults to ``'floatX'`` for symbols not included.
        broadcastables : dict of sympy.Symbol to tuple of bool, optional
            Mapping from SymPy symbols to broadcastable tuples.  Defaults to ``()`` (scalar) for symbols not included.

        Returns
        -------
        result : TensorVariable
            A variable corresponding to the expression's value in a PyTensor symbolic expression graph.
        """
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}

        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)


global_cache: dict[Any, Any] = {}


def as_tensor(
    expr: sp.Expr,
    cache: dict[Any, Any] | None = None,
    **kwargs,
) -> TensorVariable:
    """Convert a SymPy expression into a PyTensor graph variable.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression object to convert.
    cache : dict, optional
        Cached PyTensor variables (see :attr:`PytensorPrinter.cache`).  Defaults to the module-level global cache.
    **kwargs
        Forwarded to :meth:`PytensorPrinter.doprint` (e.g. `dtypes`, `broadcastables`).

    Returns
    -------
    result : TensorVariable
        A variable corresponding to the expression's value in a PyTensor symbolic expression graph.
    """
    if cache is None:
        cache = global_cache

    return PytensorPrinter(cache=cache, settings={}).doprint(expr, **kwargs)


def dim_handling(
    inputs: list[sp.Symbol],
    dim: int | None = None,
    dims: dict[sp.Symbol, int] | None = None,
    broadcastables: dict[sp.Symbol, tuple[bool, ...]] | None = None,
) -> dict[sp.Symbol, tuple[bool, ...]]:
    r"""Compute `broadcastables` argument to :meth:`PytensorPrinter.doprint` from convenience keyword arguments.

    Included for backwards compatibility.

    Parameters
    ----------
    inputs : list of sympy.Symbol
        Sequence of input symbols.
    dim : int, optional
        Common number of dimensions for all inputs.  Overrides other arguments if given.
    dims : dict of sympy.Symbol to int, optional
        Mapping from input symbols to number of dimensions.  Overrides `broadcastables` if given.
    broadcastables : dict of sympy.Symbol to tuple of bool, optional
        Explicit broadcastable values.  Returned unchanged if not ``None``.

    Returns
    -------
    result : dict of sympy.Symbol to tuple of bool
        Dictionary mapping elements of `inputs` to their broadcastable tuples.
    """
    if dim is not None:
        return {s: (False,) * dim for s in inputs}

    if dims is not None:
        maxdim = max(dims.values())
        return {s: (False,) * d + (True,) * (maxdim - d) for s, d in dims.items()}

    if broadcastables is not None:
        return broadcastables

    return {}


def pytensor_function(
    inputs: list[sp.Symbol],
    outputs: list[sp.Expr],
    scalar: bool = False,
    *,
    dim: int | None = None,
    dims: dict[sp.Symbol, int] | None = None,
    broadcastables: dict[sp.Symbol, tuple[bool, ...]] | None = None,
    **kwargs,
) -> callable:
    """Create a compiled PyTensor function from SymPy expressions.

    The inputs and outputs are converted to PyTensor variables using :func:`as_tensor` and then passed to
    :func:`pytensor.function`.

    Parameters
    ----------
    inputs : list of sympy.Symbol
        Sequence of symbols which constitute the inputs of the function.
    outputs : list of sympy.Expr
        Sequence of expressions which constitute the output(s) of the function.  The free symbols of each expression
        must be a subset of `inputs`.
    scalar : bool, optional
        Convert 0-dimensional arrays in output to scalars.  This will return a Python wrapper function around the
        PyTensor function object.
    cache : dict, optional
        Cached PyTensor variables (see :attr:`PytensorPrinter.cache`).  Defaults to the module-level global cache.
    dtypes : dict, optional
        Passed to :meth:`PytensorPrinter.doprint`.
    broadcastables : dict of sympy.Symbol to tuple of bool, optional
        Passed to :meth:`PytensorPrinter.doprint`.
    dims : dict of sympy.Symbol to int, optional
        Alternative to `broadcastables`.  Mapping from elements of `inputs` to integers indicating the dimension of
        their associated arrays/tensors.  Overrides `broadcastables` if given.
    dim : int, optional
        Another alternative to `broadcastables`.  Common number of dimensions to use for all arrays/tensors.
        ``pytensor_function([x, y], [...], dim=2)`` is equivalent to
        ``broadcastables={x: (False, False), y: (False, False)}``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pytensor.function`.

    Returns
    -------
    f : callable
        A callable object which takes values of `inputs` as positional arguments and returns an output array for each
        expression in `outputs`.  If `outputs` is a single expression the function returns a NumPy array; if it is a
        list of multiple expressions the function returns a list of arrays.  The returned object will either be an
        instance of :class:`pytensor.compile.function.types.Function` or a Python wrapper function around one.  In both
        cases, the returned value has a ``pytensor_function`` attribute pointing to the underlying compiled
        :func:`pytensor.function` result.

    See Also
    --------
    dim_handling

    Examples
    --------
    .. testcode:: python

        from sympy.abc import x, y, z
        from sympytensor.pytensor import pytensor_function

    A simple function with one input and one output:

    .. testcode:: python

        f1 = pytensor_function([x], [x**2 - 1], scalar=True)
        assert f1(3) == 8.0

    A function with multiple inputs and one output:

    .. testcode:: python

        f2 = pytensor_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)
        assert f2(3, 4, 2) == 5.0

    A function with multiple inputs and multiple outputs:

    .. testcode:: python

        f3 = pytensor_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)
        assert f3(2, 3) == [13.0, -5.0]
    """

    # Pop off non-pytensor keyword args
    cache = kwargs.pop("cache", {})
    dtypes = kwargs.pop("dtypes", {})

    broadcastables = dim_handling(
        inputs,
        dim=dim,
        dims=dims,
        broadcastables=broadcastables,
    )

    # Print inputs/outputs
    code = partial(as_tensor, cache=cache, dtypes=dtypes, broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))

    # fix constant expressions as variables
    toutputs = [
        output if isinstance(output, pytensor.graph.basic.Variable) else pt.as_tensor_variable(output)
        for output in toutputs
    ]

    if len(toutputs) == 1:
        toutputs = toutputs[0]

    # Compile pytensor func
    return pytensor.function(tinputs, toutputs, **kwargs)
