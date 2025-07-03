from importlib.metadata import version
from sympytensor.pymc import SympyDeterministic
from sympytensor.pytensor import as_tensor, pytensor_function

__version__ = version("sympytensor")
__all__ = ["as_tensor", "pytensor_function", "SympyDeterministic"]
