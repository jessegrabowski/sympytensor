import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import ancestors

import sympy as sp
from sympy.abc import x, y, z

from sympytensor.pytensor import PytensorPrinter, as_tensor, global_cache

from tests.helpers import X, assert_graph_equal, f_t, get_pt_vars


equivalent_symbol_pairs = [
    (x, sp.Symbol("x")),
    (X, sp.MatrixSymbol("X", *X.shape)),
    (f_t, sp.Function("f")(sp.Symbol("t"))),
]


@pytest.mark.parametrize("s1, s2", equivalent_symbol_pairs)
def test_cache_basic(s1, s2):
    cache = {}
    st = as_tensor(s1, cache=cache)

    assert as_tensor(s1, cache=cache) is st
    assert as_tensor(s1, cache={}) is not st
    assert as_tensor(s2, cache=cache) is st


def test_global_cache():
    # The _isolate_global_cache fixture in conftest.py restores the cache afterwards.
    global_cache.clear()

    for s in [x, X, f_t]:
        st = as_tensor(s)
        assert as_tensor(s) is st

    assert len(global_cache) == 3


def test_printer_cache_none_uses_global():
    """An explicit ``None`` must resolve to the global cache, not be stored as ``None`` and fail on first lookup."""
    assert PytensorPrinter(cache=None, settings={}).cache is global_cache
    assert PytensorPrinter(settings={}).cache is global_cache


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
    for v in ancestors([expr_t]):
        if v.owner is None and not isinstance(v, pytensor.graph.basic.Constant):
            assert v.name in symbol_names
            assert v.name not in seen
            seen.add(v.name)

    assert seen == symbol_names
