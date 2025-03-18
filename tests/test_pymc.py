import pymc as pm
import pytensor.tensor as pt
import pytest
import sympy as sp
from numpy.testing import assert_allclose

from sympytensor import SympyDeterministic, as_tensor
from sympytensor.pymc import _match_cache_to_rvs


def test_match_rvs_to_symbols_simple():
    x_sp = sp.Symbol("x")
    y_sp = x_sp + 1

    with pm.Model():
        x_pm = pm.Normal("x")
        cache = {}
        as_tensor(y_sp, cache=cache)
        sub_dict = _match_cache_to_rvs(cache)

    pymc_vars = [x_pm]
    for var_pt, var_pm in zip(cache.values(), pymc_vars):
        assert sub_dict[var_pt] == var_pm


def test_make_sympy_deterministic_simple():
    x_sp = sp.Symbol("x")
    y_sp = x_sp + 1

    with pm.Model():
        x_pm = pm.Normal("x")
        y_pm = SympyDeterministic("y", y_sp)

    assert_allclose(*pm.draw([y_pm, x_pm + 1], 10))


def test_make_sympy_deterministic_raises_if_missing_inputs():
    x_sp = sp.Symbol("x")
    z_sp = sp.Symbol("z")
    y_sp = x_sp + z_sp

    with pm.Model():
        pm.Normal("x")
        with pytest.raises(
            ValueError, match="The following symbols were found in the provided sympy expression"
        ):
            SympyDeterministic("y", y_sp)


def test_make_sympy_deterministic_complex():
    # Very simple linear model
    variables = x_s, x_d, P, P_e, M_d, M_s = sp.symbols("x_s x_d P P_e M_d M_s")
    params = a, b, c, d, P_e_bar, tau = sp.symbols(r"a b c d P_e_bar tau")

    equations = [
        P - a * x_s - b,
        P + c * x_d - d,
        M_d - x_d + x_s,
        P_e - P_e_bar,
        M_d - M_s,
        P - (1 + tau) * P_e,
    ]

    # Solve by putting it into reduced row-echelon form
    system, bias = sp.linear_eq_to_matrix(equations, variables)
    Ab = sp.Matrix([[system, bias]])
    A_rref, pivots = Ab.rref()

    # Solutions are in the last column. Sympy can't squeeze, so extract the elements into a list
    model = [elem for elem in A_rref[:, -1]]
    coords = {"variable": ["x_s", "x_d", "P", "P_e", "M_d", "M_s"]}

    with pm.Model(coords=coords):
        a = pm.Normal("a")
        b = pm.Normal("b")
        c = pm.Normal("c")
        d = pm.Normal("d")
        P_e_bar = pm.Normal("P_e_bar")
        tau = pm.Normal("tau")

        y = SympyDeterministic("y", model, dims=["variable"])
        pm.sample_prior_predictive()

    *param_draw, y_draw = pm.draw([a, b, c, d, P_e_bar, tau, y])
    f_sympy = sp.lambdify(params, model)

    assert_allclose(f_sympy(*param_draw), y_draw)


def test_sympy_deterministic_linalg():
    from sympy.abc import a, b, c, d

    A = sp.Matrix([[a, b], [c, d]])
    A_inv = sp.matrices.Inverse(A).doit()
    with pm.Model():
        a_pm = pm.Normal("a")
        b_pm = pm.Normal("b")
        c_pm = pm.Normal("c")
        d_pm = pm.Normal("d")
        A_pt = pt.stack([pt.stack([a_pm, b_pm]), pt.stack([c_pm, d_pm])])
        A_inv_pm = SympyDeterministic("A_inv", A_inv)
        A_inv_pt = pt.linalg.inv(A_pt)

    assert_allclose(*pm.draw([A_inv_pm, A_inv_pt]))


def test_sympy_deterministic_constant():
    x = sp.Float(5.0)
    with pm.Model():
        x_pm = SympyDeterministic("x", x)

    assert_allclose(pm.draw(x_pm, draws=100), 5.0)
