# trexdiff

_A minimal autodiff package written in C._

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/traikodinev/trexdiff/actions/workflows/test.yml/badge.svg?branch=dev-scalar)](https://github.com/traikodinev/trexdiff/actions/workflows/test.yml)

You are on branch `dev-blas`, which has a BLAS-based implementation of autodiff.
This is the second _dev_ branch of the package, built from `dev-scalar`.

This branch is still under development, some functionality is still using `dev-scalar` code!

## Current Progress

- [x] tensor2d - basic math w/ BLAS calls
- [x] tensor2d as backend for node
- [ ] python tests
    - [x] framework and tensor matmul test
    - [ ] API correctness test

`Node` and `Tensor2D` should be unified, because they are not very usable:

```py
from trexdiff import tensor2d, Node

x = tensor2d.ones(1, 1)
y = Node(x)
# node now owns `x`

del y
# also deletes x
```

Instead, current WIP is to use an `array` class:

```py
from trexdiff import array
x = array([1,2])

# array acts as both node and tensor2d
```

Other issues:

- [ ] functionals and transpose copy nodes and waste memory
- [ ] Node count is overly pessimistic
- [ ] no cycle detection in graph
- [ ] graph compilation

## Examples

_NOTE: This is still using dev-scalar code_

- 2D-regression,see [examples/01_regression_gd.ipynb](https://github.com/traikodinev/trexdiff/blob/dev-scalar/examples/01_regression_gd.ipynb)
- Neural Net classification [examples/02_nonlinear_classification.ipynb](https://github.com/traikodinev/trexdiff/blob/dev-scalar/examples/02_nonlinear_classification.ipynb)

```py
N_iter = 500

# input
x = Node(0.0)
a = Node(0.0)
b = Node(0.0)

y = Node(0.0)
y_p = a * x + b

# squared error loss
l = (y - y_p) * (y - y_p)

# learning rate
mu = 2.5e-1

for iter in range(N_iter):
    # zero grad every iteration, goes through entire graph
    l.zerograd()

    for i in range(N):
        x.val = xs[i]
        y.val = ys[i]

        l.forward()
        l.backward()

    # mean squared error (/N)
    a.val -= mu * a.grad / N
    b.val -= mu * b.grad / N
```

## Build

```sh
make example
make python

# Both
make
```

python:

```sh
make python
pip install -e .
```

## Test

```sh
make test
```

## Citation

```bibtex
@software{dinev2026trexdiff,
  author  = {Dinev, Traiko},
  title   = {trexdiff: A minimal automatic differentiation library in C},
  year    = {2026},
  url     = {https://github.com/traikodinev/trexdiff},
}
```
