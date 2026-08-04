"""Shared helpers and symbol fixtures for the printer test suite."""

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import graph_inputs
from scipy import sparse
import sympy as sp
from sympy.abc import t
from sympytensor.pytensor import as_tensor


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
    ins = list(graph_inputs(outs))
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
