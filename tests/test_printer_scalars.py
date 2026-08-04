import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.basic import equal_computations
from pytensor.tensor.variable import TensorVariable

import sympy as sp
from sympy.abc import x, y, z
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate

from sympytensor.pytensor import PytensorPrinter, as_tensor

from tests.helpers import X, assert_graph_equal, f_t, fgraph_of, get_pt_vars, pytensor_simplify, xt


def test_numeric_constant_conversion():
    float_one = sp.Float(1.0)
    int_one = sp.Integer(1)
    assert as_tensor(int_one) == 1
    assert as_tensor(float_one) == 1.0


@pytest.mark.parametrize(
    "sp_obj, expected_name, expected_ndim",
    [(x, "x", 0), (X, "X", 2)],
    ids=["Symbol", "MatrixSymbol"],
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
    assert XX.type.shape == (4, 4)


def test_MatrixSymbol_symbolic_shape():
    n = sp.Symbol("n", integer=True, positive=True)
    A = sp.MatrixSymbol("A_sym", n, 5)
    AA = as_tensor(A)
    assert AA.type.shape == (None, 5)


def test_MatrixSymbol_column_vector():
    v = sp.MatrixSymbol("v_col", 7, 1)
    vv = as_tensor(v)
    assert vv.type.shape == (7, 1)
    assert vv.type.broadcastable == (False, True)


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
        (sp.sign, pt.sign),
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
    "f_sp, expected",
    [(sp.Max, 5.0), (sp.Min, -1.0)],
    ids=["Max", "Min"],
)
def test_Max_Min_variadic(f_sp, expected):
    cache = {}
    result = as_tensor(f_sp(x, y, z), cache=cache)
    x_pt, y_pt, z_pt = get_pt_vars(cache, ["x", "y", "z"])
    assert_allclose(result.eval({x_pt: 2.0, y_pt: 5.0, z_pt: -1.0}), expected)


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


def test_complex_expression():
    expr = sp.exp(x**2 + sp.cos(y)) * sp.log(2 * z)
    cache = {}
    comp = as_tensor(expr, cache=cache)
    x_pt, y_pt, z_pt = get_pt_vars(cache, ["x", "y", "z"])
    expected = pt.exp(x_pt**2 + pt.cos(y_pt)) * pt.log(2 * z_pt)
    assert_graph_equal(comp, expected)


@pytest.mark.parametrize("dtype", ["float32", "int8"])
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


@pytest.mark.parametrize("bc, shape", [((False,), (None,)), ((True,), (1,)), ((True, False), (1, None))])
def test_broadcastables_set_static_shape(bc, shape):
    assert as_tensor(x, broadcastables={x: bc}, cache={}).type.shape == shape


def test_no_broadcastable_deprecation_warning(recwarn):
    as_tensor(x, broadcastables={x: (False, True)}, cache={})
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


broadcasting_cases = [
    [(), (), ()],
    [(False,), (False,), (False,)],
    [(True,), (False,), (False,)],
    [(False, True), (False, False), (False, False)],
    [(True, False), (False, False), (False, False)],
]


@pytest.mark.parametrize("bc1, bc2, bc3", broadcasting_cases)
def test_broadcasting(bc1, bc2, bc3):
    expr = x + y
    comp = as_tensor(expr, broadcastables={x: bc1, y: bc2})
    assert comp.broadcastable == bc3


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
    expected = pt.switch(pt.and_(x_pt > 0, x_pt < 2), 0, pt.switch(pt.or_(x_pt > 2, x_pt < 0), x_pt, np.nan))
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


def test_unknown_sympy_type_raises():
    class UnknownFunc(sp.Function):
        pass

    with pytest.raises(NotImplementedError, match="has no PyTensor mapping"):
        as_tensor(UnknownFunc(x), cache={})


def test_emptyPrinter_passthrough():
    printer = PytensorPrinter(cache={}, settings={})
    sentinel = object()
    assert printer.emptyPrinter(sentinel) is sentinel


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


def test_complex_dtype_propagation():
    cache = {}
    result = as_tensor(sp.sin(x), cache=cache, dtypes={x: "complex64"})
    assert result.type.dtype == "complex64"
