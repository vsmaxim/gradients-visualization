# Gradients visualization

## Run with poetry

To run with poetry, install dependencies with `poetry install`, then run `poetry run -m gradients_visualization`.
You will be redirected to user input prompt with syntax described below.

## Visualization syntax

All commands for prompt has following syntax:
```text
optimize (<function_def>) with <method_name> for x = <x_lbound>..<x_rbound>, y = <y_lbound>..<y_rbound>
```

Where `<function_def>` is function definition, some examples are:
- `sin(x*2) + cos(x*2)`
- `x^5-y^3*exp(y)`
- `x^5-ctg(x - 1)`
- `x / y % 10`

Where `<method_name>` is one of following:
- `vanilla` for vanilla gradient descent
- `rmsprop` for rmsprop gradient descent
- `adam` for adam gradient descent
- `momentum` for momentum gradient descent
- `adagrad` for adagrad gradient descent

Where `<x_lbound>`, `<x_rbound>`, `<y_lbound>`, `<y_rbound>` are floating point literals.

Examples of optimisation strings are:
- `optimize (sin(x)*cos(y)) with momentum for x = -2..2, y = -2..2`
- `optimize (x^2-y^2) with adagrad for x = -1.5..3.6, y = -2..2`
- `optimize (sin(x)*cos(y)) with momentum for x = -2..2, y = -2..2`
