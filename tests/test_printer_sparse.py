import numpy as np
import pytensor
from numpy.testing import assert_allclose
from scipy import sparse

import sympy as sp

from sympytensor.pytensor import as_tensor, dod_to_csr

from tests.helpers import get_pt_vars, sparse_allclose


def test_sparse_matrix():
    a, b = sp.symbols("a b")
    X = sp.SparseMatrix(2, 2, {(0, 1): 2, (1, 0): 3})
    y = sp.SparseMatrix(2, 1, {(0, 0): a, (1, 0): b})
    z = X @ y

    X_pt = as_tensor(X)

    assert X_pt.owner.op == pytensor.sparse.CSR
    assert sparse_allclose(X_pt.eval(), sparse.csr_matrix([[0, 2], [3, 0]]))

    cache = {}
    z_pt = as_tensor(z, cache=cache)
    a_pt, b_pt = get_pt_vars(cache, ["a", "b"])

    assert z_pt.owner.op == pytensor.sparse.CSR
    assert sparse_allclose(z_pt.eval({a_pt: 1, b_pt: 2}), sparse.csr_matrix([[4], [3]]))


def test_dod_to_csr_empty():
    data, idxs, pointers, shape = dod_to_csr({}, shape=(3, 4))
    assert data == []
    assert idxs == []
    assert pointers == [0, 0, 0, 0]
    assert shape == (3, 4)


def test_sparse_matrix_with_empty_rows():
    a, b = sp.symbols("a b")
    S = sp.SparseMatrix(3, 3, {(0, 1): a, (2, 0): b})
    cache = {}
    S_pt = as_tensor(S, cache=cache)
    a_pt, b_pt = get_pt_vars(cache, ["a", "b"])
    result = S_pt.eval({a_pt: 5.0, b_pt: 7.0})
    expected = np.array([[0, 5, 0], [0, 0, 0], [7, 0, 0]], dtype="float64")
    assert_allclose(result.toarray(), expected)
