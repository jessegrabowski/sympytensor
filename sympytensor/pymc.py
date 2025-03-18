import pymc as pm
import pytensor
import sympy as sp

from sympytensor.pytensor import as_tensor


def _match_cache_to_rvs(cache, model=None):
    pymc_model = pm.modelcontext(model)
    found_params = []
    var_names = [info[0] for info in cache.keys()]
    sub_dict = {}

    with pymc_model:
        for info, pytensor_var in cache.items():
            param_name, constructor, broadcast, dtype, shape = info
            param = getattr(pymc_model, param_name, None)
            if param:
                found_params.append(param.name)
                sub_dict[pytensor_var] = param

    missing_params = list(set(var_names) - set(found_params))
    if len(missing_params) > 0:
        raise ValueError(
            "The following symbols were found in the provided sympy expression, but are not found among model "
            "variables or deterministics: " + ", ".join(missing_params)
        )

    return sub_dict


def SympyDeterministic(name: str, expr: sp.Expr | list[sp.Expr], model=None, dims=None):
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
