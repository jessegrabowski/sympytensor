# Sympytensor

A tool for converting Sympy expressions to a Pytensor graph, with support for working with PyMC models.

# Installation

```text
pip install sympytensor
```

# Examples

## Writing expressions to pytensor
Two functions are provided to convert sympy expressions:

- `as_tensor` converts a sympy expression to a `pytensor` symbolic graph
- `pytensor_function` returns a compiled `pytensor.function` that computes the expression. Keyword arguments to
`pytensor.function` can be provided as `**kwargs`

Use sympy to compute 1d splines, then convert the splines to a symbolic pytensor variable:

```python
import pytensor
import sympy as sp
from sympytensor import as_tensor
from sympy.abc import x

x_data = [0, 1, 2, 3, 4, 5]
y_data = [3, 6, 5, 7, 9, 1]

s = sp.interpolating_spline(d=3, x=x, X=x_data, Y=y_data)
s_pt = as_tensor(s)
```

This generates the following function graph:
```python
pytensor.dprint(s_pt)

>>>Out: Elemwise{switch,no_inplace} [id A]
>>>      |Elemwise{and_,no_inplace} [id B]
>>>      | |Elemwise{ge,no_inplace} [id C]
>>>      | | |x [id D]
>>>      | | |TensorConstant{0} [id E]
>>>      | |Elemwise{le,no_inplace} [id F]
>>>      |   |x [id D]
>>>      |   |TensorConstant{2} [id G]
>>>      |Elemwise{add,no_inplace} [id H]
>>>      | |TensorConstant{3} [id I]
>>>      | |Elemwise{mul,no_inplace} [id J]
>>>      | | |Elemwise{true_div,no_inplace} [id K]
>>>      | | | |TensorConstant{-33} [id L]
>>>      | | | |TensorConstant{5} [id M]
>>>      | | |Elemwise{pow,no_inplace} [id N]
>>>      | |   |x [id D]
>>>      | |   |TensorConstant{2} [id O]
>>>      | |Elemwise{mul,no_inplace} [id P]
>>>      | | |Elemwise{true_div,no_inplace} [id Q]
>>>      | | | |TensorConstant{23} [id R]
>>>      | | | |TensorConstant{15} [id S]
>>>      | | |Elemwise{pow,no_inplace} [id T]
>>>      | |   |x [id D]
>>>      | |   |TensorConstant{3} [id U]
>>>      | |Elemwise{mul,no_inplace} [id V]
>>>      |   |Elemwise{true_div,no_inplace} [id W]
>>>      |   | |TensorConstant{121} [id X]
>>>      |   | |TensorConstant{15} [id Y]
>>>      |   |x [id D]
>>>      |Elemwise{switch,no_inplace} [id Z]
>>>        |Elemwise{and_,no_inplace} [id BA]
>>>        | |Elemwise{ge,no_inplace} [id BB]
>>>        | | |x [id D]
>>>        | | |TensorConstant{2} [id BC]
>>>        | |Elemwise{le,no_inplace} [id BD]
>>>        |   |x [id D]
>>>        |   |TensorConstant{3} [id BE]
>>>        |Elemwise{add,no_inplace} [id BF]
>>>        | |Elemwise{true_div,no_inplace} [id BG]
>>>        | | |TensorConstant{103} [id BH]
>>>        | | |TensorConstant{5} [id BI]
>>>        | |Elemwise{mul,no_inplace} [id BJ]
>>>        | | |Elemwise{true_div,no_inplace} [id BK]
>>>        | | | |TensorConstant{-55} [id BL]
>>>        | | | |TensorConstant{3} [id BM]
>>>        | | |x [id D]
>>>        | |Elemwise{mul,no_inplace} [id BN]
>>>        | | |Elemwise{true_div,no_inplace} [id BO]
>>>        | | | |TensorConstant{-2} [id BP]
>>>        | | | |TensorConstant{3} [id BQ]
>>>        | | |Elemwise{pow,no_inplace} [id BR]
>>>        | |   |x [id D]
>>>        | |   |TensorConstant{3} [id BS]
>>>        | |Elemwise{mul,no_inplace} [id BT]
>>>        |   |Elemwise{true_div,no_inplace} [id BU]
>>>        |   | |TensorConstant{33} [id BV]
>>>        |   | |TensorConstant{5} [id BW]
>>>        |   |Elemwise{pow,no_inplace} [id BX]
>>>        |     |x [id D]
>>>        |     |TensorConstant{2} [id BY]
>>>        |Elemwise{switch,no_inplace} [id BZ]
>>>          |Elemwise{and_,no_inplace} [id CA]
>>>          | |Elemwise{ge,no_inplace} [id CB]
>>>          | | |x [id D]
>>>          | | |TensorConstant{3} [id CC]
>>>          | |Elemwise{le,no_inplace} [id CD]
>>>          |   |x [id D]
>>>          |   |TensorConstant{5} [id CE]
>>>          |Elemwise{add,no_inplace} [id CF]
>>>          | |TensorConstant{53} [id CG]
>>>          | |Elemwise{mul,no_inplace} [id CH]
>>>          | | |Elemwise{true_div,no_inplace} [id CI]
>>>          | | | |TensorConstant{-761} [id CJ]
>>>          | | | |TensorConstant{15} [id CK]
>>>          | | |x [id D]
>>>          | |Elemwise{mul,no_inplace} [id CL]
>>>          | | |Elemwise{true_div,no_inplace} [id CM]
>>>          | | | |TensorConstant{-28} [id CN]
>>>          | | | |TensorConstant{15} [id CO]
>>>          | | |Elemwise{pow,no_inplace} [id CP]
>>>          | |   |x [id D]
>>>          | |   |TensorConstant{3} [id CQ]
>>>          | |Elemwise{mul,no_inplace} [id CR]
>>>          |   |Elemwise{true_div,no_inplace} [id CS]
>>>          |   | |TensorConstant{87} [id CT]
>>>          |   | |TensorConstant{5} [id CU]
>>>          |   |Elemwise{pow,no_inplace} [id CV]
>>>          |     |x [id D]
>>>          |     |TensorConstant{2} [id CW]
>>>          |TensorConstant{nan} [id CX]
```

## Inserting PyMC random variables into an expression

The `SympyDeterministic` function works as a drop-in replacement for pm.Deterministic, except a `sympy` expression is
expected. It will automatically search the active model context for random variables corresponding to symbols in the
expression and make substitutions.

Here is an example using sympy to symbolically compute the inverse of a matrix, which is then used in a model:

```python
from sympytensor import SympyDeterministic
import pymc as pm
import sympy as sp
from sympy.abc import a, b, c, d

A = sp.Matrix([[a, b],
               [c, d]])
A_inv = sp.matrices.Inverse(A).doit()

with pm.Model() as m:
    a_pm = pm.Normal('a')
    b_pm = pm.Normal('b')
    c_pm = pm.Normal('c')
    c_pm = pm.Normal('d')
    A_inv_pm = SympyDeterministic('A_inv', A_inv)
```
