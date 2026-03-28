import pytensor
import sympy as sp
from pytensor.tensor import TensorVariable

from sympytensor.pytensor import as_tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymc.model import Model

pm = None


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
    """
    global pm
    if pm is None:
        import pymc as pm

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


def _cache_name_lookup(cache: dict) -> dict[str, TensorVariable]:
    """Build a name to cached PyTensor variable lookup from a printer cache."""
    return {info[0]: pt_var for info, pt_var in cache.items()}


def _resolve_sympy_key(expr_key: str | sp.Expr) -> str:
    """Extract the symbol name from a replacement key."""
    if isinstance(expr_key, str):
        return expr_key
    if isinstance(expr_key, sp.Expr) and hasattr(expr_key, "name"):
        return expr_key.name
    raise TypeError(f"Replacement keys must be named sympy expressions or strings, got {type(expr_key)}")


def _resolve_model_value(model_value: str | TensorVariable, model: Model) -> TensorVariable:
    """Resolve a replacement value to a concrete :class:`~pytensor.tensor.TensorVariable`."""
    if isinstance(model_value, TensorVariable):
        return model_value
    if isinstance(model_value, str):
        result = getattr(model, model_value, None)
        if result is None:
            raise AttributeError(
                f"Variable '{model_value}' not found in the PyMC model. "
                f"Available variables: {[v.name for v in model.value_vars]}"
            )
        return result
    raise TypeError(f"Replacement values must be TensorVariables or strings, got {type(model_value)}")


def _resolve_replacements(
    replacements: dict[str | sp.Expr, str | TensorVariable],
    cache: dict,
    model: Model,
) -> dict[TensorVariable, TensorVariable]:
    """Resolve user-supplied replacements into a PyTensor substitution dict.

    Each key in `replacements` identifies a SymPy symbol (by reference or by name) that was
    printed into the PyTensor graph, and each value identifies the :class:`~pytensor.tensor.TensorVariable`
    (or a model variable looked up by name) that should replace it.

    Parameters
    ----------
    replacements : dict
        User-supplied mapping.  Keys may be :class:`sympy.Symbol` instances or ``str`` names
        corresponding to symbols in `cache`.  Values may be :class:`~pytensor.tensor.TensorVariable`
        instances or ``str`` names corresponding to variables in `model`.
    cache : dict
        Printer cache produced by :class:`~sympytensor.pytensor.PytensorPrinter`.
    model : :class:`pymc.Model`
        PyMC model used to look up variables when the value side is a string.

    Returns
    -------
    sub_dict : dict of TensorVariable to TensorVariable
    """
    name_to_cache = _cache_name_lookup(cache)

    sub_dict: dict[TensorVariable, TensorVariable] = {}
    for expr_key, model_value in replacements.items():
        symbol_name = _resolve_sympy_key(expr_key)

        if symbol_name not in name_to_cache:
            raise KeyError(
                f"Symbol '{symbol_name}' not found in the printed expression cache. "
                f"Available symbols: {list(name_to_cache)}"
            )

        sub_dict[name_to_cache[symbol_name]] = _resolve_model_value(model_value, model)

    return sub_dict


def _exclude_replaced_from_cache(cache: dict, replaced_vars: set[TensorVariable]) -> dict:
    """Return a copy of `cache` without entries whose variables were explicitly replaced."""
    return {k: v for k, v in cache.items() if v not in replaced_vars}


def SympyDeterministic(
    name: str,
    expr: sp.Expr | list[sp.Expr],
    model: Model | None = None,
    dims=None,
    replacements: dict[str | sp.Expr, str | TensorVariable] | None = None,
) -> TensorVariable:
    """Create a :class:`pymc.Deterministic` variable from a SymPy expression.

    The expression is converted to a PyTensor graph via :func:`~sympytensor.pytensor.as_tensor`, and any SymPy symbols
    whose names match random variables in the active :class:`pymc.Model` are automatically substituted.

    An optional `replacements` mapping can be provided to explicitly bind SymPy symbols to model
    variables (or arbitrary :class:`~pytensor.tensor.TensorVariable` objects), overriding name-based matching.
    Symbols not covered by `replacements` are still auto-matched by name.

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
    replacements : dict, optional
        Explicit mapping from SymPy symbols (or their string names) to PyTensor variables (or
        string names of model variables).  These take priority over automatic name matching.

    Returns
    -------
    expr_pm : TensorVariable
        The new deterministic variable registered in the model.
    """
    global pm
    if pm is None:
        import pymc as pm

    model = pm.modelcontext(model)
    cache = {}

    if isinstance(expr, list):
        pytensor_expr = pytensor.tensor.stack([as_tensor(e, cache=cache) for e in expr])
    else:
        pytensor_expr = as_tensor(expr, cache=cache)

    pytensor_expr = pytensor.tensor.as_tensor_variable(pytensor_expr)

    if replacements is not None:
        explicit_dict = _resolve_replacements(replacements, cache, model)
        remaining_cache = _exclude_replaced_from_cache(cache, set(explicit_dict.keys()))
    else:
        explicit_dict = {}
        remaining_cache = cache

    auto_dict = _match_cache_to_rvs(remaining_cache, model) if remaining_cache else {}
    replace_dict = {**auto_dict, **explicit_dict}

    pymc_expr = pytensor.graph_replace(pytensor_expr, replace_dict, strict=True)
    expr_pm = pm.Deterministic(name=name, var=pymc_expr, model=model, dims=dims)

    return expr_pm
