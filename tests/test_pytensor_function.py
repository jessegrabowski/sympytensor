import re

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytensor.compile import Function

import sympy as sp
from sympy.abc import x, y, z

from sympytensor.pytensor import dim_handling, pytensor_function

from tests.helpers import X, Y


def test_pytensor_function_single_output():
    f = pytensor_function([x, y], [x + y])
    assert f(2, 3) == 5


def test_pytensor_function_multiple_outputs():
    f = pytensor_function([x, y], [x + y, x - y])
    o1, o2 = f(2, 3)
    assert o1 == 5
    assert o2 == -1


def test_pytensor_function_matches_numpy():
    f = pytensor_function([x, y], [x + y], dim=1, dtypes={x: "float64", y: "float64"})
    assert np.linalg.norm(f([1, 2], [3, 4]) - np.asarray([4, 6])) < 1e-9

    f = pytensor_function([x, y], [x + y], dtypes={x: "float64", y: "float64"}, dim=1)
    xx = np.arange(3).astype("float64")
    yy = 2 * np.arange(3).astype("float64")
    assert np.linalg.norm(f(xx, yy) - 3 * np.arange(3)) < 1e-9


@pytest.mark.parametrize("n_out", [1, 2])
def test_pytensor_matrix_function_matches_numpy(n_out):
    m = sp.Matrix([[x, y], [z, x + y + z]])
    expected = np.array([[1.0, 2.0], [3.0, 1.0 + 2.0 + 3.0]])

    f = pytensor_function([x, y, z], [m] * n_out)
    output = f(1.0, 2.0, 3.0)
    if n_out == 1:
        output = np.expand_dims(output, 0)
    for out in output:
        assert_allclose(out, expected)


def test_dim_handling():
    assert dim_handling([x], dim=2) == {x: (False, False)}
    assert dim_handling([x, y], dims={x: 1, y: 2}) == {x: (False, True), y: (False, False)}
    assert dim_handling([x], broadcastables={x: (False,)}) == {x: (False,)}


def test_dim_handling_rejects_unknown_symbols():
    with pytest.raises(ValueError, match=re.escape("`dims` contains symbols not in `inputs`: ['y']")):
        dim_handling([x], dims={y: 1})


def test_dim_handling_empty_dims():
    assert dim_handling([x], dims={}) == {}


@pytest.mark.parametrize(
    "kwargs, test_inputs, expected_result",
    [
        (
            dict(dim=1, on_unused_input="ignore", dtypes={x: "float64", y: "float64", z: "float64"}),
            ([1, 2], [3, 4], [0, 0]),
            (np.asarray([4, 6])),
        ),
        (
            dict(dtypes={x: "float64", y: "float64", z: "float64"}, dim=1, on_unused_input="ignore"),
            ([np.arange(3), 2 * np.arange(3), 2 * np.arange(3)]),
            (3 * np.arange(3)),
        ),
    ],
)
def test_addition_pytensor_kwargs_in_function_printer(kwargs, test_inputs, expected_result):
    f = pytensor_function([x, y, z], [x + y], **kwargs)
    assert np.linalg.norm(f(*test_inputs) - expected_result) < 1e-9


function_dim_cases = [
    ([x, y], [x + y], None, [0]),  # Single 0d output
    ([X, Y], [X + Y], None, [2]),  # Single 2d output
    ([x, y], [x + y], {x: 0, y: 1}, [1]),  # Single 1d output
    ([x, y], [x + y, x - y], None, [0, 0]),  # Two 0d outputs
    ([x, y, X, Y], [x + y, X + Y], None, [0, 2]),  # One 0d output, one 2d
]


UNKNOWN_DIM_LENGTH = 5


def call_with_ones(f):
    """Call a compiled function on all-ones inputs, returning its outputs as a list."""
    in_values = [
        np.ones(tuple(UNKNOWN_DIM_LENGTH if dim is None else dim for dim in var.type.shape)) for var in f.input_storage
    ]
    out_values = f(*in_values)

    return out_values if isinstance(out_values, list) else [out_values]


@pytest.mark.parametrize(
    "inputs, outputs, in_dims, out_dims",
    function_dim_cases,
    ids=["single 0d", "single 2d", "single 1d", "two 0d", "mixed"],
)
def test_printing_scalar_function(inputs, outputs, in_dims, out_dims):
    f = pytensor_function(inputs, outputs, dims=in_dims)
    assert isinstance(f, Function)

    out_values = call_with_ones(f)

    assert [value.ndim for value in out_values] == out_dims


def test_pytensor_function_raises_on_bad_kwarg():
    with pytest.raises(TypeError, match=re.escape("function() got an unexpected keyword argument")):
        pytensor_function([x], [x + 1], foobar=3)


def test_constant_functions():
    tf = pytensor_function([], [1 + 1j])
    assert tf() == 1 + 1j
