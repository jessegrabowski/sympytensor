import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.basic import equal_computations
from pytensor.scalar.basic import ScalarType
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.subtensor import AdvancedIncSubtensor
from pytensor.tensor.variable import TensorVariable

import sympy as sp
from sympy.abc import x, y

from sympytensor.pytensor import as_tensor, pytensor_function

from tests.helpers import X, Y, Z, assert_graph_equal, get_pt_vars


def test_Trace():
    A = sp.MatrixSymbol("A", 3, 3)
    cache = {}
    result = as_tensor(sp.Trace(A), cache=cache)
    A_pt = get_pt_vars(cache, "A")
    assert_graph_equal(result, pt.trace(A_pt))


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

    assert tuple(Yt.owner.op.idx_list) == (slice(0, 1, 2), slice(3, 4, 5))
    assert Yt.owner.inputs[0] == as_tensor(X, cache=cache)
    assert all(Yt.owner.inputs[i].data == i for i in range(1, 7))

    k = sp.Symbol("k")
    start, stop, step = 4, k, 2
    Y = X[start:stop:step]
    Yt = as_tensor(Y, dtypes={n: "int32", k: "int32"})
    stop_pos = Yt.owner.op.idx_list[0].stop
    assert Yt.owner.inputs[1 + stop_pos].type == ScalarType("int32")


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
    theta = sp.Symbol("theta")
    for MatrixType in [sp.Matrix, sp.ImmutableMatrix]:
        X = MatrixType([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])
        cache = {}
        tX = as_tensor(X, cache=cache)
        assert isinstance(tX, TensorVariable)
        assert isinstance(tX.owner.op, AdvancedIncSubtensor)

        theta_pt = get_pt_vars(cache, ["theta"])
        theta_val = np.pi / 4
        result = tX.eval({theta_pt: theta_val})
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


def test_MatPow_positive_integer():
    A = sp.MatrixSymbol("A", 3, 3)
    f = pytensor_function([A], [A**3], dims={A: 2})
    A_val = np.random.default_rng(0).standard_normal((3, 3))
    assert_allclose(f(A_val), A_val @ A_val @ A_val)


def test_MatPow_zero_returns_identity():
    A = sp.MatrixSymbol("A", 3, 3)
    f = pytensor_function([A], [A**0], dims={A: 2}, on_unused_input="ignore")
    A_val = np.random.default_rng(0).standard_normal((3, 3))
    assert_allclose(f(A_val), np.eye(3))


def test_Inverse():
    A = sp.MatrixSymbol("A", 3, 3)
    f = pytensor_function([A], [A.inv()])
    A_val = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    assert np.allclose(f(A_val), np.linalg.inv(A_val))


def test_Inverse_times_vector():
    A = sp.MatrixSymbol("A", 3, 3)
    b = sp.MatrixSymbol("b", 3, 1)
    f = pytensor_function([A, b], [A.inv() * b])
    A_val = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b_val = np.array([[1.0], [2.0], [3.0]])
    assert np.allclose(f(A_val, b_val), np.linalg.solve(A_val, b_val))


def test_MatPow_large_exponent_uses_matrix_power():
    A = sp.MatrixSymbol("A", 3, 3)
    f = pytensor_function([A], [A**8], dims={A: 2})
    dot_nodes = [n for n in f.maker.fgraph.toposort() if "dot" in type(n.op).__name__.lower()]
    assert len(dot_nodes) <= 4


def test_MatPow_non_integer_exponent_raises():
    A = sp.MatrixSymbol("A", 3, 3)
    n = sp.Symbol("n")
    with pytest.raises(NotImplementedError, match="must be an integer"):
        as_tensor(sp.MatPow(A, n), cache={})


def test_1x1_matrix():
    M = sp.Matrix([[x]])
    cache = {}
    M_pt = as_tensor(M, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert M_pt.type.ndim == 2
    assert M_pt.type.shape == (1, 1)
    assert_allclose(M_pt.eval({x_pt: 7.0}), [[7.0]])


def test_identity_matrix():
    M_pt = as_tensor(sp.eye(4), cache={})
    assert M_pt.type.ndim == 2
    assert_allclose(M_pt.eval(), np.eye(4))


def test_zero_matrix():
    Z = sp.ZeroMatrix(3, 4)
    cache = {}
    Z_pt = as_tensor(Z, cache=cache)
    assert Z_pt.type.ndim == 2
    assert Z_pt.type.shape == (3, 4)
    assert_allclose(Z_pt.eval(), np.zeros((3, 4)))
