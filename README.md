# Sympytensor

Sympytensor converts sympy expressions to Pytensor graphs, with support for working with PyMC models. This allows you
to take advantage of sympy's symbolic computation capabilities together with pytensor's powerful symbolic graph
machinery.

# Installation

`sympytensor` is available on conda-forge, and can be installed with conda as follows:

```bash
conda install -c conda-forge sympytensor
```

Alternatively, it can be installed with pip:

```bash
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
import sympy as sp
from sympytensor import as_tensor
from sympy.abc import x

x_data = [0, 1, 2, 3, 4, 5]
y_data = [3, 6, 5, 7, 9, 1]

s = sp.interpolating_spline(d=3, x=x, X=x_data, Y=y_data)
s_pt = as_tensor(s)
s_pt.dprint()
```

This generates the following function graph:
```text
Switch [id A]
 ├─ And [id B]
 │  ├─ Ge [id C]
 │  │  ├─ x [id D]
 │  │  └─ 0 [id E]
 │  └─ Le [id F]
 │     ├─ x [id D]
 │     └─ 2 [id G]
 ├─ Add [id H]
 │  ├─ 3 [id I]
 │  ├─ Mul [id J]
 │  │  ├─ True_div [id K]
 │  │  │  ├─ -33 [id L]
 │  │  │  └─ 5 [id M]
 │  │  └─ Pow [id N]
 │  │     ├─ x [id D]
 │  │     └─ 2 [id O]
 │  ├─ Mul [id P]
 │  │  ├─ True_div [id Q]
 │  │  │  ├─ 23 [id R]
 │  │  │  └─ 15 [id S]
 │  │  └─ Pow [id T]
 │  │     ├─ x [id D]
 │  │     └─ 3 [id U]
 │  └─ Mul [id V]
 │     ├─ True_div [id W]
 │     │  ├─ 121 [id X]
 │     │  └─ 15 [id Y]
 │     └─ x [id D]
 └─ Switch [id Z]
    ├─ And [id BA]
    │  ├─ Ge [id BB]
    │  │  ├─ x [id D]
    │  │  └─ 2 [id BC]
    │  └─ Le [id BD]
    │     ├─ x [id D]
    │     └─ 3 [id BE]
    ├─ Add [id BF]
    │  ├─ True_div [id BG]
    │  │  ├─ 103 [id BH]
    │  │  └─ 5 [id BI]
    │  ├─ Mul [id BJ]
    │  │  ├─ True_div [id BK]
    │  │  │  ├─ -55 [id BL]
    │  │  │  └─ 3 [id BM]
    │  │  └─ x [id D]
    │  ├─ Mul [id BN]
    │  │  ├─ True_div [id BO]
    │  │  │  ├─ -2 [id BP]
    │  │  │  └─ 3 [id BQ]
    │  │  └─ Pow [id BR]
    │  │     ├─ x [id D]
    │  │     └─ 3 [id BS]
    │  └─ Mul [id BT]
    │     ├─ True_div [id BU]
    │     │  ├─ 33 [id BV]
    │     │  └─ 5 [id BW]
    │     └─ Pow [id BX]
    │        ├─ x [id D]
    │        └─ 2 [id BY]
    └─ Switch [id BZ]
       ├─ And [id CA]
       │  ├─ Ge [id CB]
       │  │  ├─ x [id D]
       │  │  └─ 3 [id CC]
       │  └─ Le [id CD]
       │     ├─ x [id D]
       │     └─ 5 [id CE]
       ├─ Add [id CF]
       │  ├─ 53 [id CG]
       │  ├─ Mul [id CH]
       │  │  ├─ True_div [id CI]
       │  │  │  ├─ -761 [id CJ]
       │  │  │  └─ 15 [id CK]
       │  │  └─ x [id D]
       │  ├─ Mul [id CL]
       │  │  ├─ True_div [id CM]
       │  │  │  ├─ -28 [id CN]
       │  │  │  └─ 15 [id CO]
       │  │  └─ Pow [id CP]
       │  │     ├─ x [id D]
       │  │     └─ 3 [id CQ]
       │  └─ Mul [id CR]
       │     ├─ True_div [id CS]
       │     │  ├─ 87 [id CT]
       │     │  └─ 5 [id CU]
       │     └─ Pow [id CV]
       │        ├─ x [id D]
       │        └─ 2 [id CW]
       └─ nan [id CX]
```

Since we now have a pytensor graph, we can manipulate it like any other pytensor variable. For example, we can apply
graph simplifications using `rewrite_graph`:

```python
from pytensor.graph.rewriting import rewrite_graph
rewrite_graph(s_pt)
s_pt.dprint()
```

The next code block shows the simplified graph. Note that constant folding has been applied, and powers of `x` that
are used in multiple places are now longer computed multiple times (notice how `Pow [id L]` and `Pow [id O]` appear in
several places)

```text
Switch [id A]
 ├─ And [id B]
 │  ├─ Ge [id C]
 │  │  ├─ x [id D]
 │  │  └─ 0 [id E]
 │  └─ Le [id F]
 │     ├─ x [id D]
 │     └─ 2 [id G]
 ├─ Add [id H]
 │  ├─ 3.0 [id I]
 │  ├─ Mul [id J]
 │  │  ├─ -6.6 [id K]
 │  │  └─ Pow [id L]
 │  │     ├─ x [id D]
 │  │     └─ 2 [id G]
 │  ├─ Mul [id M]
 │  │  ├─ 1.5333333333333334 [id N]
 │  │  └─ Pow [id O]
 │  │     ├─ x [id D]
 │  │     └─ 3 [id P]
 │  └─ Mul [id Q]
 │     ├─ 8.066666666666666 [id R]
 │     └─ x [id D]
 └─ Switch [id S]
    ├─ And [id T]
    │  ├─ Ge [id U]
    │  │  ├─ x [id D]
    │  │  └─ 2 [id G]
    │  └─ Le [id V]
    │     ├─ x [id D]
    │     └─ 3 [id P]
    ├─ Add [id W]
    │  ├─ 20.6 [id X]
    │  ├─ Mul [id Y]
    │  │  ├─ -18.333333333333332 [id Z]
    │  │  └─ x [id D]
    │  ├─ Mul [id BA]
    │  │  ├─ -0.6666666666666666 [id BB]
    │  │  └─ Pow [id O]
    │  │     └─ ···
    │  └─ Mul [id BC]
    │     ├─ 6.6 [id BD]
    │     └─ Pow [id L]
    │        └─ ···
    └─ Switch [id BE]
       ├─ And [id BF]
       │  ├─ Ge [id BG]
       │  │  ├─ x [id D]
       │  │  └─ 3 [id P]
       │  └─ Le [id BH]
       │     ├─ x [id D]
       │     └─ 5 [id BI]
       ├─ Add [id BJ]
       │  ├─ 53.0 [id BK]
       │  ├─ Mul [id BL]
       │  │  ├─ -50.733333333333334 [id BM]
       │  │  └─ x [id D]
       │  ├─ Mul [id BN]
       │  │  ├─ -1.8666666666666667 [id BO]
       │  │  └─ Pow [id O]
       │  │     └─ ···
       │  └─ Mul [id BP]
       │     ├─ 17.4 [id BQ]
       │     └─ Pow [id L]
       │        └─ ···
       └─ nan [id BR]
```

## Inserting PyMC random variables into an expression

The `SympyDeterministic` function works as a drop-in replacement for pm.Deterministic, except a `sympy` expression is
expected. It will automatically search the active model context for random variables corresponding to symbols in the
expression and make substitutions.

Here is an example using sympy to symbolically compute the inverse of a matrix, which is then used in a model:

```python
import sympy as sp
from sympy.abc import a, b, c, d

A = sp.Matrix([[a, b],
               [c, d]])
A_inv = sp.matrices.Inverse(A).doit()
```

`A_inv` is a sympy expression that computes each element in of the inverse of `A` symbolically:


$$
A^{-1} = \begin{pmatrix} \frac{d}{ad - bc} & -\frac{b}{ad - bc} \\ -\frac{c}{ad - bc} & \frac{a}{ad - bc} \end{pmatrix}
$$

We can now use this expression in a PyMC model:

```python
from sympytensor import SympyDeterministic
import pymc as pm

with pm.Model() as m:
    # We have to be careful to match the *names* of the pymc variables to the names of the sympy symbols!
    a_pm = pm.Normal('a')
    b_pm = pm.Normal('b')
    c_pm = pm.Normal('c')
    c_pm = pm.Normal('d')

    # Transform the sympy expression to a pytensor graph, then insert the random variables into it
    A_inv_pm = SympyDeterministic('A_inv', A_inv)

A_inv_pm.dprint()
```

This results in the following graph:

```text
Identity [id A] 'A_inv'
 └─ Join [id B]
    ├─ 0 [id C]
    ├─ ExpandDims{axis=0} [id D]
    │  └─ MakeVector{dtype='float64'} [id E]
    │     ├─ Mul [id F]
    │     │  ├─ normal_rv{"(),()->()"}.1 [id G] 'd'
    │     │  │  ├─ RNG(<Generator(PCG64) at 0x1691B60A0>) [id H]
    │     │  │  ├─ NoneConst{None} [id I]
    │     │  │  ├─ 0 [id J]
    │     │  │  └─ 1.0 [id K]
    │     │  └─ Pow [id L]
    │     │     ├─ Add [id M]
    │     │     │  ├─ Mul [id N]
    │     │     │  │  ├─ normal_rv{"(),()->()"}.1 [id O] 'a'
    │     │     │  │  │  ├─ RNG(<Generator(PCG64) at 0x1691B4040>) [id P]
    │     │     │  │  │  ├─ NoneConst{None} [id I]
    │     │     │  │  │  ├─ 0 [id Q]
    │     │     │  │  │  └─ 1.0 [id R]
    │     │     │  │  └─ normal_rv{"(),()->()"}.1 [id G] 'd'
    │     │     │  │     └─ ···
    │     │     │  └─ Mul [id S]
    │     │     │     ├─ -1 [id T]
    │     │     │     ├─ normal_rv{"(),()->()"}.1 [id U] 'b'
    │     │     │     │  ├─ RNG(<Generator(PCG64) at 0x1691B5D20>) [id V]
    │     │     │     │  ├─ NoneConst{None} [id I]
    │     │     │     │  ├─ 0 [id W]
    │     │     │     │  └─ 1.0 [id X]
    │     │     │     └─ normal_rv{"(),()->()"}.1 [id Y] 'c'
    │     │     │        ├─ RNG(<Generator(PCG64) at 0x1691B5EE0>) [id Z]
    │     │     │        ├─ NoneConst{None} [id I]
    │     │     │        ├─ 0 [id BA]
    │     │     │        └─ 1.0 [id BB]
    │     │     └─ -1 [id BC]
    │     └─ Mul [id BD]
    │        ├─ -1 [id BE]
    │        ├─ normal_rv{"(),()->()"}.1 [id U] 'b'
    │        │  └─ ···
    │        └─ Pow [id BF]
    │           ├─ Add [id BG]
    │           │  ├─ Mul [id BH]
    │           │  │  ├─ normal_rv{"(),()->()"}.1 [id O] 'a'
    │           │  │  │  └─ ···
    │           │  │  └─ normal_rv{"(),()->()"}.1 [id G] 'd'
    │           │  │     └─ ···
    │           │  └─ Mul [id BI]
    │           │     ├─ -1 [id BJ]
    │           │     ├─ normal_rv{"(),()->()"}.1 [id U] 'b'
    │           │     │  └─ ···
    │           │     └─ normal_rv{"(),()->()"}.1 [id Y] 'c'
    │           │        └─ ···
    │           └─ -1 [id BK]
    └─ ExpandDims{axis=0} [id BL]
       └─ MakeVector{dtype='float64'} [id BM]
          ├─ Mul [id BN]
          │  ├─ -1 [id BO]
          │  ├─ normal_rv{"(),()->()"}.1 [id Y] 'c'
          │  │  └─ ···
          │  └─ Pow [id BP]
          │     ├─ Add [id BQ]
          │     │  ├─ Mul [id BR]
          │     │  │  ├─ normal_rv{"(),()->()"}.1 [id O] 'a'
          │     │  │  │  └─ ···
          │     │  │  └─ normal_rv{"(),()->()"}.1 [id G] 'd'
          │     │  │     └─ ···
          │     │  └─ Mul [id BS]
          │     │     ├─ -1 [id BT]
          │     │     ├─ normal_rv{"(),()->()"}.1 [id U] 'b'
          │     │     │  └─ ···
          │     │     └─ normal_rv{"(),()->()"}.1 [id Y] 'c'
          │     │        └─ ···
          │     └─ -1 [id BU]
          └─ Mul [id BV]
             ├─ normal_rv{"(),()->()"}.1 [id O] 'a'
             │  └─ ···
             └─ Pow [id BW]
                ├─ Add [id BX]
                │  ├─ Mul [id BY]
                │  │  ├─ normal_rv{"(),()->()"}.1 [id O] 'a'
                │  │  │  └─ ···
                │  │  └─ normal_rv{"(),()->()"}.1 [id G] 'd'
                │  │     └─ ···
                │  └─ Mul [id BZ]
                │     ├─ -1 [id CA]
                │     ├─ normal_rv{"(),()->()"}.1 [id U] 'b'
                │     │  └─ ···
                │     └─ normal_rv{"(),()->()"}.1 [id Y] 'c'
                │        └─ ···
                └─ -1 [id CB]
```

## Explicit replacements

By default, `SympyDeterministic` matches sympy symbols to PyMC variables by name. If you prefer to be explicit about
which symbols map to which variables, pass a `replacements` dict. This is especially useful when the sympy expression
comes from an external source, or when the symbol names don't match the model variable names:

```python
import sympy as sp
from sympytensor import SympyDeterministic
import pymc as pm

# Suppose the expression uses greek letters
alpha, beta, gamma, delta = sp.symbols("alpha beta gamma delta")
A = sp.Matrix([[alpha, beta],
               [gamma, delta]])
A_inv = sp.matrices.Inverse(A).doit()

with pm.Model() as m:
    a = pm.Normal('a')
    b = pm.Normal('b')
    c = pm.Normal('c')
    d = pm.Normal('d')

    # Explicitly map sympy symbols to model variables
    A_inv_pm = SympyDeterministic('A_inv', A_inv, replacements={
        alpha: a,
        beta: b,
        gamma: c,
        delta: d,
    })
```

Keys can be sympy symbols or strings, and values can be PyTensor variables or strings naming model variables. You can
also mix explicit replacements with automatic matching — any symbols not covered by `replacements` are still matched
by name:

```python
# Expression mixes named model variables with a symbol that has a different name
offset = sp.Symbol('offset')
expr = alpha + offset

with pm.Model() as m:
    a = pm.Normal('a')
    bias = pm.Normal('bias')

    # 'alpha' is mapped to 'a' explicitly; 'offset' doesn't match any model variable,
    # so we map it to 'bias' by name
    y = SympyDeterministic('y', expr, replacements={alpha: a, "offset": bias})
```
