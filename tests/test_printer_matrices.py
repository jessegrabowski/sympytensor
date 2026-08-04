import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pytensor import config
from pytensor.graph.basic import equal_computations

import sympy as sp
from sympy.abc import x, y

from sympytensor.pytensor import _matrix_dtype, as_tensor, pytensor_function

from tests.helpers import X, Y, Z, assert_graph_equal, assert_slice_equal, get_pt_vars


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
    cache = {}
    expr_pt = as_tensor(X * Y * Z, cache=cache)
    X_pt, Y_pt, Z_pt = get_pt_vars(cache, ["X", "Y", "Z"])
    assert_graph_equal(expr_pt, pt.dot(pt.dot(X_pt, Y_pt), Z_pt))


def test_Transpose():
    cache = {}
    expr_pt = as_tensor(X.T, cache=cache)
    X_pt = get_pt_vars(cache, "X")
    X_val = np.arange(16.0).reshape(4, 4)
    assert_allclose(expr_pt.eval({X_pt: X_val}), X_val.T)


def test_MatAdd():
    cache = {}
    expr_pt = as_tensor(X + Y + Z, cache=cache)
    X_pt, Y_pt, Z_pt = get_pt_vars(cache, ["X", "Y", "Z"])
    values = [np.arange(16.0).reshape(4, 4) * scale for scale in (1, 2, 3)]
    result = expr_pt.eval(dict(zip([X_pt, Y_pt, Z_pt], values, strict=True)))
    assert_allclose(result, sum(values))


def test_slice():
    assert as_tensor(slice(1, 2, 3)) == slice(1, 2, 3)

    dtypes = {x: "int32", y: "int32"}
    cache = {}
    actual_slice = as_tensor(slice(x, y), dtypes=dtypes, cache=cache)
    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    assert_slice_equal(actual_slice, slice(x_pt, y_pt))

    cache = {}
    actual_slice = as_tensor(slice(1, x, 3), dtypes=dtypes, cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert_slice_equal(actual_slice, slice(1, x_pt, 3))


def test_MatrixSlice_constant_bounds():
    n = sp.Symbol("n", integer=True)
    X = sp.MatrixSymbol("X", n, n)

    cache = {}
    Y_pt = as_tensor(X[1:2:3, 4:5:6], cache=cache)
    X_pt = get_pt_vars(cache, "X")

    X_val = np.arange(100.0).reshape(10, 10)
    assert_allclose(Y_pt.eval({X_pt: X_val}), X_val[1:2:3, 4:5:6])


def test_MatrixSlice_symbolic_bound():
    n = sp.Symbol("n", integer=True)
    k = sp.Symbol("k", integer=True)
    X = sp.MatrixSymbol("X", n, n)

    cache = {}
    Y_pt = as_tensor(X[4:k:2], dtypes={n: "int32", k: "int32"}, cache=cache)
    X_pt, k_pt, n_pt = get_pt_vars(cache, ["X", "k", "n"])

    assert (k_pt.type.dtype, n_pt.type.dtype) == ("int32", "int32")

    X_val = np.arange(100.0).reshape(10, 10)
    result = Y_pt.eval({X_pt: X_val, k_pt: np.int32(9), n_pt: np.int32(10)})
    assert_allclose(result, X_val[4:9:2, :])


def test_BlockMatrix():
    n = sp.Symbol("n", integer=True)
    A, B, C, D = (sp.MatrixSymbol(name, n, n) for name in "ABCD")
    cache = {}
    block_pt = as_tensor(sp.BlockMatrix([[A, B], [C, D]]), cache=cache)
    A_pt, B_pt, C_pt, D_pt = get_pt_vars(cache, ["A", "B", "C", "D"])
    accepted_graphs = [
        pt.join(0, pt.join(1, A_pt, B_pt), pt.join(1, C_pt, D_pt)),
        pt.join(1, pt.join(0, A_pt, C_pt), pt.join(0, B_pt, D_pt)),
    ]
    assert any(equal_computations([block_pt], [graph]) for graph in accepted_graphs)


def jacobian_of_squares(symbols):
    """Jacobian of ``[x**2 for x in symbols]`` -- a diagonal matrix holding ``2 * x``."""
    return sp.Matrix([symbol**2 for symbol in symbols]).jacobian(symbols)


@pytest.mark.parametrize("MatrixType", [sp.Matrix, sp.ImmutableMatrix], ids=["Matrix", "ImmutableMatrix"])
def test_DenseMatrix(MatrixType):
    theta = sp.Symbol("theta")
    rotation = MatrixType([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]])

    cache = {}
    X_pt = as_tensor(rotation, cache=cache)
    theta_pt = get_pt_vars(cache, "theta")

    theta_val = np.pi / 4
    expected = np.array(
        [
            [np.cos(theta_val), -np.sin(theta_val)],
            [np.sin(theta_val), np.cos(theta_val)],
        ]
    )
    assert_allclose(X_pt.eval({theta_pt: theta_val}), expected)


def test_dense_matrix_fills_numeric_base_then_sets_symbolic_entries():
    """A symbolic dense matrix becomes a constant base plus one set-subtensor.

    Stacking every cell would evaluate identically, so the structure is what
    distinguishes the two strategies: the numeric entries have to be folded into the
    constant base, leaving only the symbolic ones in the graph.
    """
    symbols = [sp.Symbol(f"x_{i}") for i in range(3)]

    cache = {}
    jacobian_pt = as_tensor(jacobian_of_squares(symbols), cache=cache)
    x_pt = get_pt_vars(cache, [symbol.name for symbol in symbols])

    diagonal = pt.as_tensor([0, 1, 2])
    base = pt.as_tensor_variable(np.zeros((3, 3), dtype=config.floatX))
    expected = base[diagonal, diagonal].set([2 * symbol_pt for symbol_pt in x_pt])

    assert_graph_equal(jacobian_pt, expected)


def test_empty_matrix():
    X = sp.Matrix([[0 for _ in range(20)] for _ in range(20)])
    X_pt = as_tensor(X)
    assert np.allclose(X_pt.eval(), np.zeros((20, 20)))


def test_large_dense_matrix():
    symbols = [sp.Symbol(f"x_{i}") for i in range(100)]

    cache = {}
    jacobian_pt = as_tensor(jacobian_of_squares(symbols), cache=cache)
    values = np.arange(1.0, 1.0 + len(symbols))
    symbol_values = dict(zip(get_pt_vars(cache, [symbol.name for symbol in symbols]), values, strict=True))

    assert_allclose(jacobian_pt.eval(symbol_values), np.diag(2 * values))


def test_large_constant_matrix_is_folded():
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


def test_dense_matrix_with_complex_entries():
    M_pt = as_tensor(sp.Matrix([[1, 2 + 3 * sp.I], [sp.I, 0]]), cache={})

    assert M_pt.type.dtype == "complex128"
    assert_allclose(M_pt.eval(), np.array([[1, 2 + 3j], [1j, 0]]))


def test_dense_matrix_complex_symbolic_entry_keeps_imaginary_part():
    """A complex symbolic entry must widen the constant base, not be cast into a real one."""
    a = sp.Symbol("a")

    cache = {}
    M_pt = as_tensor(sp.Matrix([[a, sp.I]]), cache=cache, dtypes={a: "complex128"})
    a_pt = get_pt_vars(cache, "a")

    assert M_pt.type.dtype == "complex128"
    assert_allclose(M_pt.eval({a_pt: 4 + 1j}), np.array([[4 + 1j, 1j]]))


def test_dense_matrix_real_symbol_with_complex_entry():
    """A complex constant widens the matrix even when the symbolic entries are real."""
    a = sp.Symbol("a")

    cache = {}
    M_pt = as_tensor(sp.Matrix([[a, sp.I]]), cache=cache)
    a_pt = get_pt_vars(cache, "a")

    assert M_pt.type.dtype == "complex128"
    assert_allclose(M_pt.eval({a_pt: 4.0}), np.array([[4, 1j]]))


@pytest.mark.parametrize(
    "floatX, expected_complex", [("float64", "complex128"), ("float32", "complex64")], ids=["float64", "float32"]
)
def test_matrix_dtype_complex_follows_floatX(monkeypatch, floatX, expected_complex):
    """The complex dtype tracks floatX, so a float32 configuration is not silently widened to complex128."""
    monkeypatch.setattr(config, "floatX", floatX)

    assert _matrix_dtype([1.0]) == floatX
    assert _matrix_dtype([1j]) == expected_complex


def test_dense_matrix_non_finite_entries_stay_real():
    """``sympy.oo`` reports ``is_real=False`` and ``sympy.nan`` reports ``None``, but both are floats."""
    M_pt = as_tensor(sp.Matrix([[sp.oo, sp.nan]]), cache={})

    assert M_pt.type.dtype == config.floatX
    assert_allclose(M_pt.eval(), np.array([[np.inf, np.nan]]))


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
    dot_nodes = [node for node in f.maker.fgraph.toposort() if "dot" in type(node.op).__name__.lower()]
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
