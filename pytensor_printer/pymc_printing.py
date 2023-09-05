import pymc as pm
from pytensor_printer.printing import pytensor_code
import pytensor


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
            "variables or deterministics: "
            + ", ".join(missing_params)
        )


    return sub_dict


def SympyDeterministic(name, expr, model=None, dims=None):
    model = pm.modelcontext(model)
    cache = {}

    pytensor_expr = pytensor_code(expr, cache=cache)
    replace_dict = _match_cache_to_rvs(cache, model)

    pymc_expr = pytensor.graph_replace(pytensor_expr, replace_dict, strict=True)
    expr_pm = pm.Deterministic(name=name, var=pymc_expr, model=model, dims=dims)

    return expr_pm
