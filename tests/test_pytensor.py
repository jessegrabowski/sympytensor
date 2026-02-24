import re

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.basic import equal_computations
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.variable import TensorVariable
from scipy import sparse

import sympy as sp
from sympy.abc import t, x, y, z
from sympy.core.singleton import S

from sympytensor.pytensor import (
    as_tensor,
    dim_handling,
    dod_to_csr,
    pytensor_function,
    PytensorPrinter,
)


xt, yt, zt = (pt.scalar(name, dtype="floatX") for name in "xyz")
Xt, Yt, Zt = (pt.tensor(n, dtype="floatX", shape=(None, None)) for n in "XYZ")


def get_pt_vars(cache, names):
    if not isinstance(names, list):
        names = [names]

    pt_vars = list(cache.values())
    var_names = [v.name for v in pt_vars]
    out = []
    for name in names:
        var = pt_vars[var_names.index(name)]
        out.append(var)

    return out if len(out) > 1 else out[0]


# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = (sp.MatrixSymbol(n, 4, 4) for n in "XYZ")

# For testing AppliedUndef
f_t = sp.Function("f")(t)


def fgraph_of(*exprs):
    """Transform SymPy expressions into Pytensor Computation.

    Parameters
    ----------
    exprs
        SymPy expressions

    Returns
    -------
    pytensor.graph.fg.FunctionGraph
    """

    outs = list(map(as_tensor, exprs))
    ins = list(pytensor.graph.basic.graph_inputs(outs))
    ins, outs = pytensor.graph.basic.clone(ins, outs)
    return pytensor.graph.fg.FunctionGraph(ins, outs)


def pytensor_simplify(fgraph):
    """Simplify a Pytensor Computation.

    Parameters
    ----------
    fgraph : pytensor.graph.fg.FunctionGraph

    Returns
    -------
    pytensor.graph.fg.FunctionGraph
    """
    mode = pytensor.compile.get_default_mode().excluding("fusion")
    fgraph = fgraph.clone()
    mode.optimizer.rewrite(fgraph)
    return fgraph


def assert_graph_equal(actual, expected, in_actual=None, in_expected=None):
    """Assert two PyTensor graphs represent the same computation."""
    xs = [actual] if not isinstance(actual, list) else actual
    ys = [expected] if not isinstance(expected, list) else expected
    assert equal_computations(xs, ys, in_xs=in_actual, in_ys=in_expected), (
        f"Graphs are not equal.\n"
        f"  Actual:   {pytensor.printing.debugprint(actual, file='str')}\n"
        f"  Expected: {pytensor.printing.debugprint(expected, file='str')}"
    )


def test_numeric_constant_conversion():
    float_one = sp.Float(1.0)
    int_one = sp.Integer(1)
    assert as_tensor(int_one) == 1
    assert as_tensor(float_one) == 1.0


@pytest.mark.parametrize(
    "sp_obj, expected_name, expected_ndim",
    [
        (x, "x", 0),
        (y, "y", 0),
        (z, "z", 0),
        (X, "X", 2),
        (Y, "Y", 2),
        (Z, "Z", 2),
    ],
    ids=["x", "y", "z", "X", "Y", "Z"],
)
def test_symbol_roundtrip_to_pytensor(sp_obj, expected_name, expected_ndim):
    cache = {}
    result = as_tensor(sp_obj, cache=cache)
    assert result.name == expected_name
    assert result.type.ndim == expected_ndim


def test_Symbol():
    xx = as_tensor(x, broadcastables={x: ()})
    assert xx.broadcastable == ()
    assert xx.name == x.name


def test_MatrixSymbol():
    XX = as_tensor(X)
    assert isinstance(XX, TensorVariable)
    assert XX.type.broadcastable == (False, False)


def test_AppliedUndef():
    ftt = as_tensor(f_t)
    assert isinstance(ftt, TensorVariable)
    assert ftt.broadcastable == ()
    assert ftt.name == "f_t"


def test_add():
    expr = x + y
    comp = as_tensor(expr)
    assert comp.owner.op == pytensor.tensor.add


@pytest.mark.parametrize(
    "f_sp, f_pt",
    [
        (sp.Abs, pt.abs),
        (sp.sign, pt.sgn),
        (sp.ceiling, pt.ceil),
        (sp.floor, pt.floor),
        (sp.cos, pt.cos),
        (sp.acos, pt.arccos),
        (sp.sin, pt.sin),
        (sp.asin, pt.arcsin),
        (sp.tan, pt.tan),
        (sp.atan, pt.arctan),
        (sp.cosh, pt.cosh),
        (sp.acosh, pt.arccosh),
        (sp.sinh, pt.sinh),
        (sp.asinh, pt.arcsinh),
        (sp.tanh, pt.tanh),
        (sp.atanh, pt.arctanh),
        (sp.erf, pt.erf),
        (sp.gamma, pt.gamma),
        (sp.loggamma, pt.gammaln),
        (sp.log, pt.log),
        (sp.exp, pt.exp),
    ],
    ids=lambda f: getattr(f, "__name__", str(f)),
)
def test_unary_mapping(f_sp, f_pt):
    cache = {}
    result = as_tensor(f_sp(x), cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert_graph_equal(result, f_pt(x_pt))


@pytest.mark.parametrize(
    "f_sp, f_pt",
    [
        (sp.Max, pt.maximum),
        (sp.Min, pt.minimum),
        (sp.atan2, pt.arctan2),
    ],
    ids=["Max", "Min", "atan2"],
)
def test_binary_mapping(f_sp, f_pt):
    cache = {}
    result = as_tensor(f_sp(x, y), cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    assert_graph_equal(result, f_pt(x_pt, y_pt))


@pytest.mark.parametrize(
    "f_sp, f_pt",
    [
        (sp.re, pt.real),
        (sp.im, pt.imag),
        (sp.arg, pt.angle),
    ],
    ids=["re", "im", "arg"],
)
def test_complex_unary_mapping(f_sp, f_pt):
    # Complex dtype prevents SymPy from simplifying re(x) -> x, etc.
    cache = {}
    result = as_tensor(f_sp(x), cache=cache, dtypes={x: "complex128"})
    x_pt = get_pt_vars(cache, "x")
    assert_graph_equal(result, f_pt(x_pt))


def test_logical_not():
    # Boolean dtype prevents SymPy from simplifying Not(p)
    p = sp.Symbol("p")
    cache = {}
    result = as_tensor(sp.Not(p), cache=cache, dtypes={p: "bool"})
    p_pt = get_pt_vars(cache, "p")
    assert_graph_equal(result, pt.invert(p_pt))


def test_logical_xor():
    p, q = sp.symbols("p q")
    cache = {}
    result = as_tensor(sp.Xor(p, q), cache=cache, dtypes={p: "bool", q: "bool"})
    p_pt, q_pt = get_pt_vars(cache, ["p", "q"])
    assert_graph_equal(result, pt.bitwise_xor(p_pt, q_pt))


def test_Trace():
    A = sp.MatrixSymbol("A", 3, 3)
    cache = {}
    result = as_tensor(sp.Trace(A), cache=cache)
    A_pt = get_pt_vars(cache, "A")
    assert_graph_equal(result, pt.linalg.trace(A_pt))


def test_Determinant():
    A = sp.MatrixSymbol("A", 3, 3)
    cache = {}
    result = as_tensor(sp.Determinant(A), cache=cache)
    A_pt = get_pt_vars(cache, "A")
    assert_graph_equal(result, pt.linalg.det(A_pt))


def test_HadamardProduct():
    A = sp.MatrixSymbol("A", 3, 3)
    B = sp.MatrixSymbol("B", 3, 3)
    cache = {}
    result = as_tensor(sp.HadamardProduct(A, B), cache=cache)
    A_pt, B_pt = get_pt_vars(cache, ["A", "B"])
    assert_graph_equal(result, A_pt * B_pt)


def test_complex_expression():
    expr = sp.exp(x**2 + sp.cos(y)) * sp.log(2 * z)
    cache = {}
    comp = as_tensor(expr, cache=cache)
    x_pt, y_pt, z_pt = get_pt_vars(cache, ["x", "y", "z"])
    expected = pt.exp(x_pt**2 + pt.cos(y_pt)) * pt.log(2 * z_pt)
    assert_graph_equal(comp, expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "int16", "int32", "int64"])
def test_dtype(dtype):
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
    expr = x + y
    comp = as_tensor(expr, broadcastables={x: bc1, y: bc2})
    assert comp.broadcastable == bc3


def test_MatMul():
    expr = X * Y * Z
    cache = {}
    expr_t = as_tensor(expr, cache=cache)
    Xt, Yt, Zt = get_pt_vars(cache, ["X", "Y", "Z"])
    expected = pt.dot(pt.dot(Xt, Yt), Zt)
    assert_graph_equal(expr_t, expected)


def test_Transpose():
    assert isinstance(as_tensor(X.T).owner.op, DimShuffle)


def test_MatAdd():
    expr = X + Y + Z
    assert isinstance(as_tensor(expr).owner.op, Elemwise)


def test_Rationals():
    assert as_tensor(sp.Integer(2) / 3) == 2 / 3
    assert as_tensor(S.Half) == 0.5


def test_Integers():
    assert as_tensor(sp.Integer(3)) == 3


def test_factorial():
    n = sp.Symbol("n")
    sp_fact = as_tensor(sp.factorial(n))
    assert sp_fact.eval({"n": 3}) == 6


@pytest.mark.filterwarnings("ignore: A Supervisor feature is missing")
def test_Derivative():
    def simp(expr):
        return pytensor_simplify(fgraph_of(expr))

    fg_actual = simp(as_tensor(sp.Derivative(sp.sin(x), x, evaluate=False)))
    fg_expected = simp(pytensor.grad(pt.sin(xt), xt))
    assert equal_computations(
        fg_actual.outputs,
        fg_expected.outputs,
        in_xs=list(fg_actual.inputs),
        in_ys=list(fg_expected.inputs),
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
def test_pytensor_matrix_function_matches_numpy(n_out, scalar):
    m = sp.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])

    f = pytensor_function([x, y, z], [m] * n_out, scalar=scalar)
    output = f(1.0, 2.0, 3.0)
    if n_out == 1:
        output = np.expand_dims(output, 0)
    for out in output:
        assert_allclose(out, expected)


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
    from pytensor.compile.function.types import Function

    f = pytensor_function(inputs, outputs, dims=in_dims, scalar=scalar)

    assert isinstance(f.pytensor_function, Function)

    in_values = [
        np.ones([1 if bc else 5 for bc in i.type.broadcastable])
        for i in f.pytensor_function.input_storage
    ]
    out_values = f(*in_values)
    if not isinstance(out_values, list):
        out_values = [out_values]

    assert len(out_dims) == len(out_values)
    for d, value in zip(out_dims, out_values):
        if scalar and d == 0:
            assert isinstance(value, np.number)
        else:
            assert isinstance(value, np.ndarray)
            assert value.ndim == d


def test_pytensor_function_raises_on_bad_kwarg():
    with pytest.raises(TypeError, match=re.escape("function() got an unexpected keyword argument")):
        pytensor_function([x], [x + 1], foobar=3)


def test_slice():
    assert as_tensor(slice(1, 2, 3)) == slice(1, 2, 3)

    def assert_slice_equal(s1, s2):
        for attr in ["start", "stop", "step"]:
            a1 = getattr(s1, attr)
            a2 = getattr(s2, attr)
            if a1 is None or a2 is None:
                assert a1 is None and a2 is None, f"slice.{attr} mismatch: {a1} vs {a2}"
            elif isinstance(a1, TensorVariable) and isinstance(a2, TensorVariable):
                assert_graph_equal(a1, a2)
            else:
                assert a1 == a2, f"slice.{attr} mismatch: {a1} vs {a2}"

    dtypes = {x: "int32", y: "int32"}
    cache = {}
    actual_slice = as_tensor(slice(x, y), dtypes=dtypes, cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    assert_slice_equal(actual_slice, slice(x_pt, y_pt))

    cache = {}
    actual_slice = as_tensor(slice(1, x, 3), dtypes=dtypes, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert_slice_equal(actual_slice, slice(1, x_pt, 3))


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
    cache = {}
    Block = sp.BlockMatrix([[A, B], [C, D]])
    Blockt = as_tensor(Block, cache=cache)
    At, Bt, Ct, Dt = get_pt_vars(cache, ["A", "B", "C", "D"])
    solutions = [
        pt.join(0, pt.join(1, At, Bt), pt.join(1, Ct, Dt)),
        pt.join(1, pt.join(0, At, Ct), pt.join(0, Bt, Dt)),
    ]
    assert any(equal_computations([Blockt], [sol]) for sol in solutions)


def test_DenseMatrix():
    from pytensor.tensor.subtensor import AdvancedIncSubtensor

    t = sp.Symbol("theta")
    for MatrixType in [sp.Matrix, sp.ImmutableMatrix]:
        X = MatrixType([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
        cache = {}
        tX = as_tensor(X, cache=cache)
        assert isinstance(tX, TensorVariable)
        assert isinstance(tX.owner.op, AdvancedIncSubtensor)

        t_pt = get_pt_vars(cache, ["theta"])
        theta_val = np.pi / 4
        result = tX.eval({t_pt: theta_val})
        expected = np.array(
            [
                [np.cos(theta_val), -np.sin(theta_val)],
                [np.sin(theta_val), np.cos(theta_val)],
            ]
        )
        assert_allclose(result, expected)


def test_empty_matrix():
    X = sp.Matrix([[0 for _ in range(20)] for _ in range(20)])
    tX = as_tensor(X)
    assert np.allclose(tX.eval(), np.zeros((20, 20)))


def test_large_dense_matrix():
    from pytensor.tensor.subtensor import AdvancedIncSubtensor

    vars = [sp.Symbol(f"x_{i}") for i in range(100)]

    eqs = sp.Matrix([x**2 for x in vars])
    jac = eqs.jacobian(vars)

    jac_pt = as_tensor(jac)

    assert isinstance(jac_pt.owner.op, AdvancedIncSubtensor)

    small_eqs = sp.Matrix([x**2 for x in vars[:3]])
    small_jac = small_eqs.jacobian(vars[:3])
    small_jac_pt = as_tensor(small_jac)

    assert isinstance(small_jac_pt.owner.op, AdvancedIncSubtensor)

    const_matrix = sp.ones(50, 50)
    const_pt = as_tensor(const_matrix)
    assert const_pt.owner is None
    assert_allclose(const_pt.eval(), np.ones((50, 50)))


def test_dense_matrix_mixed_symbolic_numeric():
    a, b = sp.symbols("a b")
    M = sp.Matrix(
        [
            [1, a, 0],
            [sp.Rational(1, 2), -3, b],
            [sp.pi, 0, a + b],
        ]
    )

    cache = {}
    M_pt = as_tensor(M, cache=cache)
    a_pt, b_pt = get_pt_vars(cache, ["a", "b"])

    result = M_pt.eval({a_pt: 2.0, b_pt: 5.0})
    expected = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.5, -3.0, 5.0],
            [np.pi, 0.0, 7.0],
        ]
    )
    assert_allclose(result, expected)


def test_dense_matrix_all_numeric_varied():
    M = sp.Matrix(
        [
            [-sp.Rational(7, 3), sp.sqrt(2), 0],
            [sp.pi, -sp.exp(1), sp.Rational(1, 7)],
            [100, 0, -sp.Rational(1, 1000)],
        ]
    )
    M_pt = as_tensor(M)

    result = M_pt.eval()
    expected = np.array(
        [
            [-7 / 3, np.sqrt(2), 0.0],
            [np.pi, -np.e, 1 / 7],
            [100.0, 0.0, -0.001],
        ]
    )
    assert_allclose(result, expected, rtol=1e-7)


pairs = [
    (x, sp.Symbol("x")),
    (X, sp.MatrixSymbol("X", *X.shape)),
    (f_t, sp.Function("f")(sp.Symbol("t"))),
]


@pytest.mark.parametrize("s1, s2", pairs)
def test_cache_basic(s1, s2):
    cache = {}
    st = as_tensor(s1, cache=cache)

    assert as_tensor(s1, cache=cache) is st
    assert as_tensor(s1, cache={}) is not st
    assert as_tensor(s2, cache=cache) is st


def test_global_cache():
    from sympytensor.pytensor import global_cache

    backup = dict(global_cache)
    try:
        global_cache.clear()

        for s in [x, X, f_t]:
            st = as_tensor(s)
            assert as_tensor(s) is st

    finally:
        global_cache.update(backup)


def test_cache_types_distinct():
    symbols = [sp.Symbol("f_t"), sp.MatrixSymbol("f_t", 4, 4), f_t]

    cache = {}
    printed = {}

    for s in symbols:
        st = as_tensor(s, cache=cache)
        assert st not in printed.values()
        printed[s] = st

    assert len(set(map(id, printed.values()))) == len(symbols)

    for s, st in printed.items():
        assert as_tensor(s, cache=cache) is st


def test_symbols_are_created_once():
    expr = sp.Add(x, x, evaluate=False)
    cache = {}
    comp = as_tensor(expr, cache=cache)
    x_pt = get_pt_vars(cache, "x")

    assert_graph_equal(comp, x_pt + x_pt)

    # A separately-created variable should NOT match (different identity)
    x_other = pt.scalar("x", dtype="floatX")
    assert not equal_computations([comp], [x_pt + x_other])


def test_cache_complex():
    expr = x**2 + (y - sp.exp(x)) * sp.sin(z - x * y)
    symbol_names = {s.name for s in expr.free_symbols}
    expr_t = as_tensor(expr)

    seen = set()
    for v in pytensor.graph.basic.ancestors([expr_t]):
        if v.owner is None and not isinstance(v, pytensor.graph.basic.Constant):
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    assert seen == symbol_names


def test_Piecewise():
    # A piecewise linear
    expr = sp.Piecewise((0, x < 0), (x, x < 2), (1, True))  # ___/III
    cache = {}
    result = as_tensor(expr, cache=cache)
    assert result.owner.op == pt.switch
    x_pt = get_pt_vars(cache, "x")
    expected = pt.switch(x_pt < 0, 0, pt.switch(x_pt < 2, x_pt, 1))
    assert_graph_equal(result, expected)

    cache = {}
    expr = sp.Piecewise((x, x < 0))
    result = as_tensor(expr, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    expected = pt.switch(x_pt < 0, x_pt, np.nan)
    assert_graph_equal(result, expected)

    cache = {}
    expr = sp.Piecewise((0, sp.And(x > 0, x < 2)), (x, sp.Or(x > 2, x < 0)))
    result = as_tensor(expr, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    expected = pt.switch(
        pt.and_(x_pt > 0, x_pt < 2), 0, pt.switch(pt.or_(x_pt > 2, x_pt < 0), x_pt, np.nan)
    )
    assert_graph_equal(result, expected)


@pytest.mark.parametrize(
    "sp_rel, pt_rel_fn",
    [
        (lambda x, y: sp.Eq(x, y), pt.eq),
        (lambda x, y: sp.Ne(x, y), pt.neq),
        (lambda x, y: x > y, pt.gt),
        (lambda x, y: x < y, pt.lt),
        (lambda x, y: x >= y, pt.ge),
        (lambda x, y: x <= y, pt.le),
    ],
    ids=["Eq", "Ne", "Gt", "Lt", "Ge", "Le"],
)
def test_relational(sp_rel, pt_rel_fn):
    cache = {}
    result = as_tensor(sp_rel(x, y), cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    assert_graph_equal(result, pt_rel_fn(x_pt, y_pt))


def test_complex_number_operations():
    from sympy.functions.elementary.complexes import conjugate

    dtypes = {x: "complex128", y: "complex128"}

    cache = {}
    result = as_tensor(y * conjugate(x), dtypes=dtypes, cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    assert_graph_equal(result, y_pt * x_pt.conj())

    cache = {}
    result = as_tensor((1 + 2j) * x, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    expected = x_pt * (pt.as_tensor_variable(1.0) + pt.as_tensor_variable(2.0) * pt.complex(0, 1))
    assert_graph_equal(result, expected)


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

    i_pt, j_pt, x_pt = get_pt_vars(cache, ["i", "j", "x"])
    assert x.eval({x_pt: np.arange(20).reshape((10, 2)), i_pt: 5, j_pt: 1}) == 11.0

    with pytest.raises(IndexError):
        x.eval({x_pt: np.zeros((10, 2)), i_pt: 8, j_pt: 3})


def test_indexedbase_with_index_and_no_range():
    i = sp.Idx("i")
    j = sp.Idx("j")

    cache = {}
    x = as_tensor(sp.IndexedBase("x")[i, j], cache=cache)
    assert x.type.shape == ()
    assert x.owner.inputs[0].ndim == 2
    assert len(cache) == 3

    i_pt, j_pt, x_pt = get_pt_vars(cache, ["i", "j", "x"])

    assert x.eval({x_pt: np.arange(20).reshape((10, 2)), i_pt: 5, j_pt: 1}) == 11.0


def test_sliced_indexbase_1d():
    cache = {}
    x = sp.IndexedBase("x", shape=(10,))
    x = as_tensor(x[7], cache=cache)
    x_pt = get_pt_vars(cache, ["x"])

    assert x.type.shape == ()
    assert x.owner.inputs[0].type.shape == (10,)
    assert len(cache) == 1
    assert x.eval({x_pt: np.arange(10)}) == 7.0


def test_sliced_indexbase_2d():
    cache = {}
    x = sp.IndexedBase("x", shape=(10, 10))
    x1 = as_tensor(x[0, 1], cache=cache)
    x2 = as_tensor(x[5, 4], cache=cache)
    x_pt = get_pt_vars(cache, ["x"])

    assert len(cache) == 1
    assert x1.type.shape == ()
    assert x1.owner.inputs[0].ndim == 2
    assert x1.owner.inputs[0].type.shape == (10, 10)
    assert x1.eval({x_pt: np.arange(100).reshape(10, 10)}) == 1.0
    assert x2.eval({x_pt: np.arange(100).reshape(10, 10)}) == 54.0


@pytest.mark.parametrize("i_range", [(0, 10), (5, 7)])
@pytest.mark.parametrize("reduce_op", [sp.Sum, sp.Product])
def test_print_reduce_1d(i_range: tuple, reduce_op):
    cache = {}
    i = sp.Idx("i")

    low, high = i_range
    x = sp.IndexedBase(
        "x",
    )[i]
    z = reduce_op(x, (i, low, high))
    z = as_tensor(z, cache=cache)

    x_pt = get_pt_vars(cache, ["x"])

    x_val = np.arange(1, 11)
    expected = x_val[low : high + 1]
    expected = expected.sum() if reduce_op == sp.Sum else np.prod(expected)
    assert z.eval({x_pt: x_val}) == expected


@pytest.mark.parametrize("i_range", [(0, 10), (5, 7)])
@pytest.mark.parametrize("reduce_op", [sp.Sum, sp.Product])
def test_print_reduce_2d(i_range: tuple, reduce_op):
    cache = {}
    i = sp.Idx("i")
    j = sp.Idx("j")

    low, high = i_range
    x = sp.IndexedBase(
        "x",
    )[i, j]
    z = reduce_op(x, (i, low, high))
    z = as_tensor(z, cache=cache)

    x_pt, j_pt = get_pt_vars(cache, ["x", "j"])
    x_val = np.arange(1, 21).reshape(10, 2)
    expected = x_val[low : high + 1, 0]
    expected = expected.sum(axis=0) if reduce_op == sp.Sum else np.prod(expected, axis=0)
    assert z.eval({x_pt: x_val, j_pt: 0}) == expected


@pytest.mark.parametrize("reduce_op", [sp.Sum, sp.Product])
def test_print_reduce_many_d(reduce_op):
    cache = {}
    i, j, k, l = sp.symbols("i j k l", cls=sp.Idx)  # noqa: E741

    x = sp.IndexedBase(
        "x",
    )[i, j, k, l]
    z = reduce_op(x, (i, 0, 1), (j, 0, 1), (k, 0, 1))
    z = as_tensor(z, cache=cache)

    x_pt, l_pt = get_pt_vars(cache, ["x", "l"])
    x_val = np.linspace(1, 2, 16).reshape(2, 2, 2, 2)
    expected = x_val[:2, :2, :2, 0]
    expected = (
        expected.sum(axis=(0, 1, 2)) if reduce_op == sp.Sum else np.prod(expected, axis=(0, 1, 2))
    )

    assert z.eval({x_pt: x_val, l_pt: 0}) == expected


def sparse_allclose(A, B, atol=1e-8):
    if np.array_equal(A.shape, B.shape) == 0:
        return False

    r1, c1, v1 = sparse.find(A)
    r2, c2, v2 = sparse.find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)

    if index_match == 0:
        return False
    else:
        return np.allclose(v1, v2, atol=atol)


def test_sparse_matrix():
    a, b = sp.symbols("a b")
    X = sp.SparseMatrix(2, 2, {(0, 1): 2, (1, 0): 3})
    y = sp.SparseMatrix(2, 1, {(0, 0): a, (1, 0): b})
    z = X @ y

    X_pt = as_tensor(X)

    assert X_pt.owner.op == pytensor.sparse.CSR
    assert sparse_allclose(X_pt.eval(), sparse.csr_matrix([[0, 2], [3, 0]]))

    cache = {}
    z_pt = as_tensor(z, cache=cache)
    a_pt, b_pt = get_pt_vars(cache, ["a", "b"])

    assert z_pt.owner.op == pytensor.sparse.CSR
    assert sparse_allclose(z_pt.eval({a_pt: 1, b_pt: 2}), sparse.csr_matrix([[4], [3]]))


def test_MatPow_positive_integer():
    A = sp.MatrixSymbol("A", 3, 3)
    cache = {}
    result = as_tensor(A**2, cache=cache)
    A_pt = get_pt_vars(cache, "A")
    expected = pt.dot(A_pt, A_pt)
    assert_graph_equal(result, expected)


def test_MatPow_negative_exponent_raises():
    A = sp.MatrixSymbol("A", 3, 3)
    with pytest.raises(NotImplementedError, match="positive integer matrix powers"):
        as_tensor(sp.MatPow(A, sp.Integer(-2)), cache={})


def test_unknown_sympy_type_raises():
    class UnknownFunc(sp.Function):
        pass

    with pytest.raises(NotImplementedError, match="has no PyTensor mapping"):
        as_tensor(UnknownFunc(x), cache={})


def test_dod_to_csr_empty():
    data, idxs, pointers, shape = dod_to_csr({}, shape=(3, 4))
    assert data == []
    assert idxs == []
    assert pointers == [0, 0, 0, 0]
    assert shape == (3, 4)


def test_reduction_unsupported_op_raises():
    i = sp.Idx("i")
    expr = sp.Sum(sp.IndexedBase("x")[i], (i, 0, 5))
    printer = PytensorPrinter(cache={}, settings={})
    with pytest.raises(NotImplementedError, match="Unsupported reduction operation"):
        printer._print_reduction(expr, op="mean")


def test_emptyPrinter_passthrough():
    printer = PytensorPrinter(cache={}, settings={})
    sentinel = object()
    assert printer.emptyPrinter(sentinel) is sentinel


def test_1x1_matrix():
    M = sp.Matrix([[x]])
    cache = {}
    M_pt = as_tensor(M, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert M_pt.type.ndim == 2
    assert M_pt.type.shape == (1, 1)
    assert_allclose(M_pt.eval({x_pt: 7.0}), [[7.0]])


def test_large_integer():
    big = sp.Integer(10**100)
    result = as_tensor(big)
    assert result == 10**100


def test_nested_piecewise():
    inner = sp.Piecewise((x, x > 0), (0, True))
    outer = sp.Piecewise((inner, y > 0), (-1, True))
    cache = {}
    result = as_tensor(outer, cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    expected = pt.switch(y_pt > 0, pt.switch(x_pt > 0, x_pt, 0), -1)
    assert_graph_equal(result, expected)


def test_piecewise_single_true():
    expr = sp.Piecewise((x**2, True))
    cache = {}
    result = as_tensor(expr, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    # True is printed as 1 (an int), so switch(1, x**2, nan) — evaluates to x**2
    assert_allclose(result.eval({x_pt: 3.0}), 9.0)


def test_identity_matrix():
    M_pt = as_tensor(sp.eye(4), cache={})
    assert M_pt.type.ndim == 2
    assert_allclose(M_pt.eval(), np.eye(4))


def test_negative_literal_index():
    xb = sp.IndexedBase("x", shape=(10,))
    cache = {}
    result = as_tensor(xb[-1], cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert_allclose(result.eval({x_pt: np.arange(10, dtype="float64")}), 9.0)


def test_complex_dtype_propagation():
    cache = {}
    result = as_tensor(sp.sin(x), cache=cache, dtypes={x: "complex64"})
    assert result.type.dtype == "complex64"


def test_sparse_matrix_with_empty_rows():
    a, b = sp.symbols("a b")
    S = sp.SparseMatrix(3, 3, {(0, 1): a, (2, 0): b})
    cache = {}
    S_pt = as_tensor(S, cache=cache)
    a_pt, b_pt = get_pt_vars(cache, ["a", "b"])
    result = S_pt.eval({a_pt: 5.0, b_pt: 7.0})
    expected = np.array([[0, 5, 0], [0, 0, 0], [7, 0, 0]], dtype="float64")
    assert_allclose(result.toarray(), expected)


def test_zero_matrix():
    Z = sp.ZeroMatrix(3, 4)
    cache = {}
    Z_pt = as_tensor(Z, cache=cache)
    assert Z_pt.type.ndim == 2
    assert Z_pt.type.shape == (3, 4)
    assert_allclose(Z_pt.eval(), np.zeros((3, 4)))
