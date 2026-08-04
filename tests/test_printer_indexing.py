import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.traversal import ancestors
from pytensor.raise_op import CheckAndRaise

import sympy as sp

from sympytensor.pytensor import PytensorPrinter, as_tensor

from tests.helpers import get_pt_vars


def count_range_checks(x):
    return sum(var.owner is not None and isinstance(var.owner.op, CheckAndRaise) for var in ancestors([x]))


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


def test_Idx_non_concrete_bounds_unguarded():
    k = sp.Idx("k", (1, sp.oo))

    cache = {}
    x = as_tensor(sp.IndexedBase("x")[k], cache=cache)
    assert count_range_checks(x) == 0

    k_pt, x_pt = get_pt_vars(cache, ["k", "x"])
    assert x.eval({x_pt: np.arange(10.0), k_pt: 3}) == 3.0


@pytest.mark.xfail(
    strict=True,
    reason="_print_Indexed coerces base dimensions with a bare int(); fixed by the _static_dim read in commit 1.5",
)
def test_indexed_symbolic_shape():
    n = sp.Symbol("n", integer=True)
    i = sp.Idx("i")

    cache = {}
    x = as_tensor(sp.IndexedBase("A", shape=(n,))[i], cache=cache)
    assert x.type.shape == ()


def test_Idx_guard_emitted_once():
    i = sp.Idx("i", range=10)

    cache = {}
    x = as_tensor(sp.IndexedBase("A")[i] * sp.IndexedBase("B")[i], cache=cache)
    assert count_range_checks(x) == 1

    i_pt, a_pt, b_pt = get_pt_vars(cache, ["i", "A", "B"])
    with pytest.raises(IndexError):
        x.eval({a_pt: np.zeros(10), b_pt: np.zeros(10), i_pt: 10})


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


@pytest.mark.parametrize("i_range", [(0, 9), (5, 7)])
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


@pytest.mark.parametrize("i_range", [(0, 9), (5, 7)])
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
    i, j, k, m = sp.symbols("i j k m", cls=sp.Idx)

    x = sp.IndexedBase(
        "x",
    )[i, j, k, m]
    z = reduce_op(x, (i, 0, 1), (j, 0, 1), (k, 0, 1))
    z = as_tensor(z, cache=cache)

    x_pt, m_pt = get_pt_vars(cache, ["x", "m"])
    x_val = np.linspace(1, 2, 16).reshape(2, 2, 2, 2)
    expected = x_val[:2, :2, :2, 0]
    expected = expected.sum(axis=(0, 1, 2)) if reduce_op == sp.Sum else np.prod(expected, axis=(0, 1, 2))

    assert np.isclose(z.eval({x_pt: x_val, m_pt: 0}), expected)


def test_sum_with_mul_summand():
    cache = {}
    i = sp.Idx("i")
    x = sp.IndexedBase("x")[i]
    a = sp.Symbol("a")
    z = as_tensor(sp.Sum(a * x, (i, 0, 5)), cache=cache)

    x_pt, a_pt = get_pt_vars(cache, ["x", "a"])
    x_val = np.arange(1.0, 7.0)
    assert np.isclose(z.eval({x_pt: x_val, a_pt: 2.0}), 2.0 * x_val.sum())


def test_sum_with_add_summand():
    cache = {}
    i = sp.Idx("i")
    x = sp.IndexedBase("x")[i]
    y = sp.IndexedBase("y")[i]
    z = as_tensor(sp.Sum(x + y, (i, 0, 5)), cache=cache)

    x_pt, y_pt = get_pt_vars(cache, ["x", "y"])
    x_val = np.arange(1.0, 7.0)
    y_val = np.arange(7.0, 13.0)
    assert np.isclose(z.eval({x_pt: x_val, y_pt: y_val}), (x_val + y_val).sum())


def test_sum_with_pow_summand():
    cache = {}
    i = sp.Idx("i")
    x = sp.IndexedBase("x")[i]
    z = as_tensor(sp.Sum(x**2, (i, 0, 5)), cache=cache)

    x_pt = get_pt_vars(cache, ["x"])
    x_val = np.arange(1.0, 7.0)
    assert np.isclose(z.eval({x_pt: x_val}), (x_val**2).sum())


def test_product_with_mul_summand():
    cache = {}
    i = sp.Idx("i")
    x = sp.IndexedBase("x")[i]
    a = sp.Symbol("a")
    z = as_tensor(sp.Product(a * x, (i, 0, 5)), cache=cache)

    x_pt, a_pt = get_pt_vars(cache, ["x", "a"])
    x_val = np.arange(1.0, 7.0)
    assert np.isclose(z.eval({x_pt: x_val, a_pt: 2.0}), (2.0**6) * x_val.prod())


def test_reduction_unsupported_op_raises():
    i = sp.Idx("i")
    expr = sp.Sum(sp.IndexedBase("x")[i], (i, 0, 5))
    printer = PytensorPrinter(cache={}, settings={})
    with pytest.raises(NotImplementedError, match="Unsupported reduction operation"):
        printer._print_reduction(expr, op="mean")


def test_negative_literal_index():
    xb = sp.IndexedBase("x", shape=(10,))
    cache = {}
    result = as_tensor(xb[-1], cache=cache)
    x_pt = get_pt_vars(cache, "x")
    assert_allclose(result.eval({x_pt: np.arange(10, dtype="float64")}), 9.0)
