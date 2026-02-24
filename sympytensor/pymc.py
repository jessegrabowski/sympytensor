import pymc as pm
import pytensor
import sympy as sp
from pytensor.tensor import TensorVariable

from sympytensor.pytensor import as_tensor


def _match_cache_to_rvs(cache: dict, model=None) -> dict:
    """Match cached PyTensor variables to random variables in a PyMC model.

    Iterates over the printer `cache` and looks up each symbol name as an attribute on `model`.  If all symbols are
    found the function returns a substitution dict suitable for :func:`pytensor.graph_replace`.

    Parameters
    ----------
    cache : dict
        Printer cache produced by :class:`~sympytensor.pytensor.PytensorPrinter`.
    model : :class:`pymc.Model`, optional
        PyMC model to search.  Defaults to the current model context.

    Returns
    -------
    sub_dict : dict of TensorVariable to TensorVariable
        Mapping from the cached printer variables to the corresponding model random variables.

    Raises
    ------
    ValueError
        If any symbol name in `cache` does not correspond to a variable in `model`.
    """
    pymc_model = pm.modelcontext(model)
    found_params = []
    var_names = [info[0] for info in cache.keys()]
    sub_dict = {}

    with pymc_model:
        for info, pytensor_var in cache.items():
            param_name, constructor, broadcast, dtype, shape = info
            param = getattr(pymc_model, param_name, None)
            if param is not None:
                found_params.append(param.name)
                sub_dict[pytensor_var] = param

    missing_params = list(set(var_names) - set(found_params))
    if len(missing_params) > 0:
        raise ValueError(
            "The following symbols were found in the provided sympy expression, but are not found among model "
            "variables or deterministics: " + ", ".join(missing_params)
        )

    return sub_dict


def SympyDeterministic(
    name: str, expr: sp.Expr | list[sp.Expr], model: pm.Model | None = None, dims=None
) -> TensorVariable:
    """Create a :class:`pymc.Deterministic` variable from a SymPy expression.

    The expression is converted to a PyTensor graph via :func:`~sympytensor.pytensor.as_tensor`, and any SymPy symbols
    whose names match random variables in the active :class:`pymc.Model` are automatically substituted.

    Parameters
    ----------
    name : str
        Name for the deterministic variable in the PyMC model.
    expr : sympy.Expr or list of sympy.Expr
        SymPy expression (or list of expressions to stack) to convert.
    model : pymc.Model, optional
        PyMC model to add the variable to.  Defaults to the current model context.
    dims : str or tuple of str, optional
        Dimension names, forwarded to :class:`pymc.Deterministic`.

    Returns
    -------
    expr_pm : TensorVariable
        The new deterministic variable registered in the model.
    """
    model = pm.modelcontext(model)
    cache = {}

    if isinstance(expr, list):
        pytensor_expr = pytensor.tensor.stack([as_tensor(e, cache=cache) for e in expr])
    else:
        pytensor_expr = as_tensor(expr, cache=cache)

    # This catches corner cases where the input is a constant variable
    pytensor_expr = pytensor.tensor.as_tensor_variable(pytensor_expr)

    replace_dict = _match_cache_to_rvs(cache, model)
    pymc_expr = pytensor.graph_replace(pytensor_expr, replace_dict, strict=True)
    expr_pm = pm.Deterministic(name=name, var=pymc_expr, model=model, dims=dims)

    return expr_pm
