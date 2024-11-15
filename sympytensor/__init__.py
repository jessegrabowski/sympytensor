from sympytensor._version import get_versions
from sympytensor.pymc import SympyDeterministic
from sympytensor.pytensor import as_tensor, pytensor_function

__version__ = get_versions()["version"]
__all__ = ["as_tensor", "pytensor_function", "SympyDeterministic"]
