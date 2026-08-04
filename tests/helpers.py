import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import graph_inputs
from pytensor.tensor.variable import TensorVariable
from scipy import sparse

import sympy as sp
from sympy.abc import t

from sympytensor.pytensor import as_tensor


xt = pt.scalar("x", dtype="floatX")

# Default set of matrix symbols for testing - make square so we can both
# multiply and perform elementwise operations between them.
X, Y, Z = (sp.MatrixSymbol(n, 4, 4) for n in "XYZ")

# For testing AppliedUndef
f_t = sp.Function("f")(t)


def get_pt_vars(cache, names):
    """Look up printed PyTensor variables in a printer cache by name.

    Parameters
    ----------
    cache : dict
        Printer cache populated by :func:`~sympytensor.pytensor.as_tensor`.
    names : str or list of str
        Variable name, or list of names, to retrieve.

    Returns
    -------
    vars : TensorVariable or list of TensorVariable
        The matching variable when a single name resolves, otherwise one variable per
        requested name in the order given.
    """
    if not isinstance(names, list):
        names = [names]

    pt_vars = list(cache.values())
    var_names = [v.name for v in pt_vars]
    out = []
    for name in names:
        var = pt_vars[var_names.index(name)]
        out.append(var)

    return out if len(out) > 1 else out[0]


def fgraph_of(*exprs):
    """Convert SymPy expressions into a cloned PyTensor function graph.

    Parameters
    ----------
    exprs
        SymPy expressions.

    Returns
    -------
    fgraph : pytensor.graph.fg.FunctionGraph
        Graph over clones of the printed inputs and outputs.
    """
    outs = list(map(as_tensor, exprs))
    ins = list(graph_inputs(outs))
    ins, outs = pytensor.graph.basic.clone(ins, outs)
    return pytensor.graph.fg.FunctionGraph(ins, outs)


def pytensor_simplify(fgraph):
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


def assert_slice_equal(actual, expected):
    """Assert two slices agree attribute by attribute, comparing symbolic bounds as graphs."""
    for attr in ("start", "stop", "step"):
        a, e = getattr(actual, attr), getattr(expected, attr)
        assert (a is None) == (e is None), f"slice.{attr} mismatch: {a} vs {e}"

        if a is None:
            continue

        if isinstance(a, TensorVariable):
            assert_graph_equal(a, e)
        else:
            assert a == e, f"slice.{attr} mismatch: {a} vs {e}"


def sparse_allclose(A, B, atol=1e-8):
    if not np.array_equal(A.shape, B.shape):
        return False

    rows_A, cols_A, values_A = sparse.find(A)
    rows_B, cols_B, values_B = sparse.find(B)

    if not (np.array_equal(rows_A, rows_B) and np.array_equal(cols_A, cols_B)):
        return False

    return np.allclose(values_A, values_B, atol=atol)
