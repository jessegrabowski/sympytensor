import re

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.basic import Variable
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.variable import TensorVariable

xt, yt, zt = (pt.scalar(name, dtype="floatX") for name in "xyz")
Xt, Yt, Zt = (pt.tensor(n, dtype="floatX", shape=(None, None)) for n in "XYZ")

import sympy as sp
from sympy.abc import t, x, y, z
from sympy.core.singleton import S

from sympytensor.pytensor import as_tensor, dim_handling, pytensor_function

# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = (sp.MatrixSymbol(n, 4, 4) for n in "XYZ")

# For testing AppliedUndef
f_t = sp.Function("f")(t)


def fgraph_of(*exprs):
    """Transform SymPy expressions into Pytensor Computation.

    Parameters
    ==========
    exprs
        SymPy expressions

    Returns
    =======
    pytensor.graph.fg.FunctionGraph
    """

    outs = list(map(as_tensor, exprs))
    ins = list(pytensor.graph.basic.graph_inputs(outs))
    ins, outs = pytensor.graph.basic.clone(ins, outs)
    return pytensor.graph.fg.FunctionGraph(ins, outs)


def pytensor_simplify(fgraph):
    """Simplify a Pytensor Computation.

    Parameters
    ==========
    fgraph : pytensor.graph.fg.FunctionGraph

    Returns
    =======
    pytensor.graph.fg.FunctionGraph
    """
    mode = pytensor.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.rewrite(fgraph)
    return fgraph


def pt_eq(a, b):
    """Test two Pytensor objects for equality.

    Also accepts numeric types and lists/tuples of supported types.

    Note - debugprint() has a bug where it will accept numeric types but does
    not respect the "file" argument and in this case and instead prints the number
    to stdout and returns an empty string. This can lead to tests passing where
    they should fail because any two numbers will always compare as equal. To
    prevent this we treat numbers as a separate case.
    """
    numeric_types = (int, float, np.number)
    a_is_num = isinstance(a, numeric_types)
    b_is_num = isinstance(b, numeric_types)

    # Compare numeric types using regular equality
    if a_is_num or b_is_num:
        if not (a_is_num and b_is_num):
            return False

        return a == b

    # Compare sequences element-wise
    a_is_seq = isinstance(a, (tuple, list))
    b_is_seq = isinstance(b, (tuple, list))

    if a_is_seq or b_is_seq:
        if not (a_is_seq and b_is_seq) or type(a) != type(b):
            return False

        return list(map(pt_eq, a)) == list(map(pt_eq, b))

    # Otherwise, assume debugprint() can handle it
    astr = pytensor.printing.debugprint(a, file="str")
    bstr = pytensor.printing.debugprint(b, file="str")

    # Check for bug mentioned above
    for argname, argval, argstr in [("a", a, astr), ("b", b, bstr)]:
        if argstr == "":
            raise TypeError(
                "aesara.printing.debugprint(%s) returned empty string "
                "(%s is instance of %r)" % (argname, argname, type(argval))
            )

    return astr == bstr


@pytest.mark.parametrize("pt_obj, sp_obj", zip([xt, yt, zt, Xt, Yt, Zt], [x, y, z, X, Y, Z]))
def test_example_symbols(pt_obj, sp_obj):
    """
    Check that the example symbols in this module print to their Aesara
    equivalents, as many of the other tests depend on this.
    """
    assert pt_eq(pt_obj, as_tensor(sp_obj))


def test_Symbol():
    """Test printing a Symbol to a pytensor variable."""
    xx = as_tensor(x, broadcastables={x: ()})
    assert xx.broadcastable == ()
    assert xx.name == x.name


def test_MatrixSymbol():
    """Test printing a MatrixSymbol to a aesara variable."""
    XX = as_tensor(X)
    assert isinstance(XX, TensorVariable)
    assert XX.type.broadcastable == (False, False)


#
# @pytest.mark.parametrize('shape',  [(), (10,), (None,), (None, 10), (10, None), (None, None)])
# def test_MatrixSymbol_wrong_dims(shape):
#     """ Test MatrixSymbol with invalid broadcastable. """
#     with raises(ValueError):
#         pytensor_code(X, shapes={X:shape})


def test_AppliedUndef():
    """Test printing AppliedUndef instance, which works similarly to Symbol."""
    ftt = as_tensor(f_t)
    assert isinstance(ftt, TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == "f_t"


def test_add():
    expr = x + y
    comp = as_tensor(expr)
    assert comp.owner.op == pytensor.tensor.add


@pytest.mark.parametrize("f_sp, f_pt", [(sp.sin, pt.sin), (sp.tan, pt.tan)], ids=["sin", "tan"])
def test_trig(f_sp, f_pt):
    assert pt_eq(as_tensor(f_sp(x)), f_pt(xt))


def test_complex_expression():
    """Test printing a complex expression with multiple symbols."""
    expr = sp.exp(x**2 + sp.cos(y)) * sp.log(2 * z)
    comp = as_tensor(expr)
    expected = pt.exp(xt**2 + pt.cos(yt)) * pt.log(2 * zt)
    assert pt_eq(comp, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "int16", "int32", "int64"])
def test_dtype(dtype):
    """Test specifying specific data types through the dtype argument."""
    assert as_tensor(x, dtypes={x: dtype}).type.dtype == dtype


def test_floatX_dtype():
    assert as_tensor(x, dtypes={x: "floatX"}).type.dtype in ("float32", "float64")


def test_type_promotion_simple():
    assert as_tensor(x + 1, dtypes={x: "float32"}).type.dtype == "float32"


def test_type_promotion_mixed():
    assert as_tensor(x + y, dtypes={x: "float64", y: "float32"}).type.dtype == "float64"


@pytest.mark.parametrize("bc", [(False,), (True,), (False, False), (True, False)])
@pytest.mark.parametrize("s", [x, f_t])
def test_broadcastables(bc, s):
    # TODO: Matrix broadcasting?
    assert as_tensor(s, broadcastables={s: bc}, cache={}).broadcastable == bc


cases = [
    [(), (), ()],
    [(False,), (False,), (False,)],
    [(True,), (False,), (False,)],
    [(False, True), (False, False), (False, False)],
    [(True, False), (False, False), (False, False)],
]


@pytest.mark.parametrize("bc1, bc2, bc3", cases)
def test_broadcasting(bc1, bc2, bc3):
    """Test "broadcastable" attribute after applying element-wise binary op."""
    expr = x + y
    comp = as_tensor(expr, broadcastables={x: bc1, y: bc2})
    assert comp.broadcastable == bc3


def test_MatMul():
    expr = X * Y * Z
    expr_t = as_tensor(expr)
    assert isinstance(expr_t.owner.op, Dot)
    assert pt_eq(expr_t, Xt.dot(Yt).dot(Zt))


def test_Transpose():
    assert isinstance(as_tensor(X.T).owner.op, DimShuffle)


def test_MatAdd():
    expr = X + Y + Z
    assert isinstance(as_tensor(expr).owner.op, Elemwise)


def test_Rationals():
    assert pt_eq(as_tensor(sp.Integer(2) / 3), pt.true_div(2, 3))
    assert pt_eq(as_tensor(S.Half), pt.true_div(1, 2))


def test_Integers():
    assert as_tensor(sp.Integer(3)) == 3


def test_factorial():
    n = sp.Symbol("n")
    assert as_tensor(sp.factorial(n))


@pytest.mark.filterwarnings("ignore: A Supervisor feature is missing")
def test_Derivative():
    simp = lambda expr: pytensor_simplify(fgraph_of(expr))
    assert pt_eq(
        simp(as_tensor(sp.Derivative(sp.sin(x), x, evaluate=False))),
        simp(pytensor.grad(pt.sin(xt), xt)),
    )


def test_pytensor_function_single_output():
    f = pytensor_function([x, y], [x + y])
    assert f(2, 3) == 5


def test_pytensor_function_multiple_outputs():
    f = pytensor_function([x, y], [x + y, x - y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1


def test_pytensor_function_matches_numpy():
    f = pytensor_function([x, y], [x + y], dim=1, dtypes={x: "float64", y: "float64"})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = pytensor_function([x, y], [x + y], dtypes={x: "float64", y: "float64"}, dim=1)
    xx = np.arange(3).astype("float64")
    yy = 2 * np.arange(3).astype("float64")
    assert np.linalg.norm(f(xx, yy) - 3 * np.arange(3)) < 1e-9


@pytest.mark.parametrize("n_out, scalar", [(1, True), (1, False), (2, False)])
def test_pytensor_matrix_funciton_matches_numpy(n_out, scalar):
    m = sp.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])

    f = pytensor_function([x, y, z], [m] * n_out, scalar=scalar)
    output = f(1.0, 2.0, 3.0)
    if n_out == 1:
        output = np.expand_dims(output, 0)
    for out in output:
        np.testing.assert_allclose(out, expected)


def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True), y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}


@pytest.mark.parametrize(
    "kwargs, test_inputs, expected_result",
    [
        (
            dict(
                dim=1, on_unused_input="ignore", dtypes={x: "float64", y: "float64", z: "float64"}
            ),
            ([1, 2], [3, 4], [0, 0]),
            (np.asarray([4, 6])),
        ),
        (
            dict(
                dtypes={x: "float64", y: "float64", z: "float64"}, dim=1, on_unused_input="ignore"
            ),
            ([np.arange(3), 2 * np.arange(3), 2 * np.arange(3)]),
            (3 * np.arange(3)),
        ),
    ],
)
def test_addition_pytensor_kwargs_in_function_printer(kwargs, test_inputs, expected_result):
    f = pytensor_function([x, y, z], [x + y], **kwargs)
    assert np.linalg.norm(f(*test_inputs) - expected_result) < 1e-9


scalar_cases = [
    ([x, y], [x + y], None, [0]),  # Single 0d output
    ([X, Y], [X + Y], None, [2]),  # Single 2d output
    ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
    ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
    ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
]


@pytest.mark.parametrize(
    "inputs, outputs, in_dims, out_dims",
    scalar_cases,
    ids=["single 0d", "single 2d", "single 1d", "two 0d", "mixed"],
)
@pytest.mark.parametrize("scalar", [False, True])
def test_printing_scalar_function(inputs, outputs, in_dims, out_dims, scalar):
    """Test the "scalar" argument to aesara_function()."""
    from pytensor.compile.function.types import Function

    f = pytensor_function(inputs, outputs, dims=in_dims, scalar=scalar)

    # Check the pytensor_function attribute is set whether wrapped or not
    assert isinstance(f.pytensor_function, Function)

    # Feed in inputs of the appropriate size and get outputs
    in_values = [
        np.ones([1 if bc else 5 for bc in i.type.broadcastable])
        for i in f.pytensor_function.input_storage
    ]
    out_values = f(*in_values)
    if not isinstance(out_values, list):
        out_values = [out_values]

    # Check output types and shapes
    assert len(out_dims) == len(out_values)
    for d, value in zip(out_dims, out_values):
        if scalar and d == 0:
            # Should have been converted to a scalar value
            assert isinstance(value, np.number)

        else:
            # Otherwise should be an array
            assert isinstance(value, np.ndarray)
            assert value.ndim == d


def test_pytensor_function_raises_on_bad_kwarg():
    """
    Passing an unknown keyword argument to aesara_function() should raise an
    exception.
    """
    with pytest.raises(TypeError, match=re.escape("function() got an unexpected keyword argument")):
        pytensor_function([x], [x + 1], foobar=3)


def test_slice():
    assert as_tensor(slice(1, 2, 3)) == slice(1, 2, 3)

    def pt_eq_slice(s1, s2):
        for attr in ["start", "stop", "step"]:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                if not (a1 is None or a2 is None):
                    return False
            elif not pt_eq(a1, a2):
                return False
        return True

    dtypes = {x: "int32", y: "int32"}
    assert pt_eq_slice(as_tensor(slice(x, y), dtypes=dtypes), slice(xt, yt))
    assert pt_eq_slice(as_tensor(slice(1, x, 3), dtypes=dtypes), slice(1, xt, 3))


def test_MatrixSlice():
    cache = {}

    n = sp.Symbol("n", integer=True)
    X = sp.MatrixSymbol("X", n, n)

    Y = X[1:2:3, 4:5:6]
    Yt = as_tensor(Y, cache=cache)

    s = ScalarType(dtype="int64")
    assert tuple(Yt.owner.op.idx_list) == (slice(s, s, s), slice(s, s, s))
    assert Yt.owner.inputs[0] == as_tensor(X, cache=cache)
    assert all(Yt.owner.inputs[i].data == i for i in range(1, 7))

    k = sp.Symbol("k")
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = as_tensor(Y, dtypes={n: "int32", k: "int32"})
    assert Yt.owner.op.idx_list[0].stop == ScalarType("int32")


def test_BlockMatrix():
    n = sp.Symbol("n", integer=True)
    A, B, C, D = (sp.MatrixSymbol(name, n, n) for name in "ABCD")
    At, Bt, Ct, Dt = map(as_tensor, (A, B, C, D))
    Block = sp.BlockMatrix([[A, B], [C, D]])
    Blockt = as_tensor(Block)
    solutions = [
        pt.join(0, pt.join(1, At, Bt), pt.join(1, Ct, Dt)),
        pt.join(1, pt.join(0, At, Ct), pt.join(0, Bt, Dt)),
    ]
    assert any(pt_eq(Blockt, solution) for solution in solutions)


def test_DenseMatrix():
    from pytensor.tensor.basic import Join

    t = sp.Symbol("theta")
    for MatrixType in [sp.Matrix, sp.ImmutableMatrix]:
        X = MatrixType([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
        tX = as_tensor(X)
        assert isinstance(tX, TensorVariable)
        assert isinstance(tX.owner.op, Join)


# Pairs of objects which should be considered equivalent with respect to caching
pairs = [
    (x, sp.Symbol("x")),
    (X, sp.MatrixSymbol("X", *X.shape)),
    (f_t, sp.Function("f")(sp.Symbol("t"))),
]


@pytest.mark.parametrize("s1, s2", pairs)
def test_cache_basic(s1, s2):
    """Test single symbol-like objects are cached when printed by themselves."""
    cache = {}
    st = as_tensor(s1, cache=cache)

    # Test hit with same instance
    assert as_tensor(s1, cache=cache) is st

    # Test miss with same instance but new cache
    assert as_tensor(s1, cache={}) is not st

    # Test hit with different but equivalent instance
    assert as_tensor(s2, cache=cache) is st


def test_global_cache():
    """Test use of the global cache."""
    from sympytensor.pytensor import global_cache

    backup = dict(global_cache)
    try:
        # Temporarily empty global cache
        global_cache.clear()

        for s in [x, X, f_t]:
            st = as_tensor(s)
            assert as_tensor(s) is st

    finally:
        # Restore global cache
        global_cache.update(backup)


def test_cache_types_distinct():
    """
    Test that symbol-like objects of different types (Symbol, MatrixSymbol,
    AppliedUndef) are distinguished by the cache even if they have the same
    name.
    """
    symbols = [sp.Symbol("f_t"), sp.MatrixSymbol("f_t", 4, 4), f_t]

    cache = {}  # Single shared cache
    printed = {}

    for s in symbols:
        st = as_tensor(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    # Check all printed objects are distinct
    assert len(set(map(id, printed.values()))) == len(symbols)

    # Check retrieving
    for s, st in printed.items():
        assert as_tensor(s, cache=cache) is st


def test_symbols_are_created_once():
    """
    Test that a symbol is cached and reused when it appears in an expression
    more than once.
    """
    expr = sp.Add(x, x, evaluate=False)
    comp = as_tensor(expr)

    assert pt_eq(comp, xt + xt)
    assert not pt_eq(comp, xt + as_tensor(x))


def test_cache_complex():
    """
    Test caching on a complicated expression with multiple symbols appearing
    multiple times.
    """
    expr = x**2 + (y - sp.exp(x)) * sp.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = as_tensor(expr)

    # Iterate through variables in the Pytensor computational graph that the
    # printed expression depends on
    seen = set()
    for v in pytensor.graph.basic.ancestors([expr_t]):
        # Owner-less, non-constant variables should be our symbols
        if v.owner is None and not isinstance(v, pytensor.graph.basic.Constant):
            # Check it corresponds to a symbol and appears only once
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    # Check all were present
    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sp.Piecewise((0, x < 0), (x, x < 2), (1, True))  # ___/III
    result = as_tensor(expr)
    assert result.owner.op == pt.switch

    expected = pt.switch(xt < 0, 0, pt.switch(xt < 2, xt, 1))
    assert pt_eq(result, expected)

    expr = sp.Piecewise((x, x < 0))
    result = as_tensor(expr)
    expected = pt.switch(xt < 0, xt, np.nan)
    assert pt_eq(result, expected)

    expr = sp.Piecewise((0, sp.And(x > 0, x < 2)), (x, sp.Or(x > 2, x < 0)))
    result = as_tensor(expr)
    expected = pt.switch(pt.and_(xt > 0, xt < 2), 0, pt.switch(pt.or_(xt > 2, xt < 0), xt, np.nan))
    assert pt_eq(result, expected)


def test_Relationals():
    assert pt_eq(as_tensor(sp.Eq(x, y)), pt.eq(xt, yt))
    assert pt_eq(as_tensor(sp.Ne(x, y)), pt.neq(xt, yt))
    assert pt_eq(as_tensor(x > y), xt > yt)
    assert pt_eq(as_tensor(x < y), xt < yt)
    assert pt_eq(as_tensor(x >= y), xt >= yt)
    assert pt_eq(as_tensor(x <= y), xt <= yt)


def test_complexfunctions():
    from sympy.functions.elementary.complexes import conjugate

    atv = pt.as_tensor_variable
    cplx = pt.complex
    dtypes = {x: "complex128", y: "complex128"}
    xt, yt = as_tensor(x, dtypes=dtypes), as_tensor(y, dtypes=dtypes)

    assert pt_eq(as_tensor(y * conjugate(x), dtypes=dtypes), yt * (xt.conj()))
    assert pt_eq(as_tensor((1 + 2j) * x), xt * (atv(1.0) + atv(2.0) * cplx(0, 1)))


def test_constant_functions():
    tf = pytensor_function([], [1 + 1j])
    assert tf() == 1 + 1j


def test_indexedbase():
    cache = {}
    x = as_tensor(sp.IndexedBase("x"), cache=cache)
    assert x.name == "x"
    assert x.type.shape == (None,)
    assert len(cache) == 1


def test_indexedbase_with_declared_shape():
    cache = {}
    x = as_tensor(sp.IndexedBase("x", shape=(10, 10)), cache=cache)
    assert x.name == "x"
    assert x.type.shape == (10, 10)
    assert len(cache) == 1


def test_indexedbase_with_different_shapes_cache_separately():
    cache = {}
    x = as_tensor(sp.IndexedBase("x", shape=(10, 10)), cache=cache)
    y = as_tensor(sp.IndexedBase("x", shape=(10, 7)), cache=cache)
    assert x is not y
    assert len(cache) == 2


def test_indexedbase_with_index():
    i = sp.Idx("i", range=10)
    j = sp.Idx("j", range=2)

    cache = {}
    x = as_tensor(sp.IndexedBase("x")[i, j], cache=cache)
    assert x.type.shape == ()
    assert x.owner.inputs[0].ndim == 2
    assert len(cache) == 3

    i_pt, j_pt, x_pt = list(cache.values())
    with pytest.raises(IndexError):
        x.eval({x_pt: np.zeros((10, 2)), i_pt: 8, j_pt: 3})


def test_sliced_indexbase_1d():
    cache = {}
    x = sp.IndexedBase("x", shape=(10,))
    x = as_tensor(x[7], cache=cache)
    x_pt = list(cache.values())[0]

    assert x.type.shape == ()
    assert x.owner.inputs[0].type.shape == (10,)
    assert len(cache) == 1
    assert x.eval({x_pt: np.arange(10)}) == 7.0


def test_sliced_indexbase_2d():
    cache = {}
    x = sp.IndexedBase("x", shape=(10, 10))
    x1 = as_tensor(x[0, 1], cache=cache)
    x2 = as_tensor(x[5, 4], cache=cache)
    x_pt = list(cache.values())[0]

    assert len(cache) == 1
    assert x1.type.shape == ()
    assert x1.owner.inputs[0].ndim == 2
    assert x1.owner.inputs[0].type.shape == (10, 10)
    assert x1.eval({x_pt: np.arange(100).reshape(10, 10)}) == 1.0
    assert x2.eval({x_pt: np.arange(100).reshape(10, 10)}) == 54.0
