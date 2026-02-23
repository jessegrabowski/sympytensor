import pytest

from sympytensor.pytensor import global_cache


@pytest.fixture(autouse=True)
def _isolate_global_cache():
    """Backup and restore the global printer cache around each test.

    Prevents tests that call ``as_tensor()`` without an explicit ``cache={}``from leaking state into other tests via
    the module-level ``global_cache``.
    """
    backup = dict(global_cache)
    yield
    global_cache.clear()
    global_cache.update(backup)
