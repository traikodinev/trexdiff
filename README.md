# trexdiff

_A minimal autodiff package written in C._

You are on branch `dev-scalar`, which has a scalar implementation of autodiff.
This is the earliest _dev_ branch of the package.

## Example

2D-regression,see `examples/01_regression_gd.ipynb`

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
