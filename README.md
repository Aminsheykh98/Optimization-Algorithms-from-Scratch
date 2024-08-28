# Optimization-Algorithms-from-Scratch

# JAX-Based Optimization Algorithms

This repository contains a set of optimization algorithms implemented from scratch using JAX. The code is organized into three main files:

1. **Optimization_algorithms.py**: This file contains the implementations of various optimization algorithms.
2. **Solver_class.py**: This file defines a `Solver` class that applies the optimization algorithms to specific functions.
3. **main.py**: This file demonstrates the usage of the optimization algorithms and the `Solver` class on different objective functions.

## Optimization Algorithms

The following optimization algorithms are implemented in `Optimization_algorithms.py`:

- **Analytical Gradient Descent** (`analyt_GD`): Computes the optimal step size analytically for quadratic functions.
- **Gradient Descent** (`GD`): Implements standard gradient descent with a fixed learning rate.
- **Golden Search Gradient Descent** (`Gsearch_GD`): Uses golden section search to find the optimal step size in gradient descent.
- **Newton's Method** (`Newton`): Utilizes the Hessian matrix for optimization, ensuring positive definiteness.
- **Quasi-Newton Method** (`Quasi_Newton`): A modified version of Newton's method that ensures the Hessian matrix is positive definite.

### Additional Utility Functions

- **alpha_analytical**: Computes the optimal step size for gradient descent analytically.
- **golden_search**: Implements the golden section search algorithm.
- **num_min_check**: Analyzes the quadratic function to determine the number of minima.

## Solver Class

The `Solver` class, defined in `Solver_class.py`, is a general-purpose class for applying different optimization algorithms to arbitrary functions. It supports the following modes:

- `Analytical_stepsize`: Uses the Analytical Gradient Descent algorithm.
- `Gradient_descent`: Uses the standard Gradient Descent algorithm.
- `Golden_search_gradient_descent`: Uses the Golden Search Gradient Descent algorithm.
- `Newton`: Uses Newton's Method.
- `Mixed`: Starts with Gradient Descent and switches to Quasi-Newton Method if early stopping is triggered.

### Methods

- **iter_loop(x_init, epochs, patience=5)**: Iteratively applies the chosen optimization algorithm until convergence or early stopping.
- **plot()**: Plots the contour of the objective function and the optimization path (for functions with two variables).

## Usage

To use the optimization algorithms, you can define your objective function and pass it to the `Solver` class along with the desired optimization mode. Here is an example usage with a quadratic function:

```python
from Optimization_algorithms import *
from Solver_class import *

# Define your quadratic function
def func1(x, **kwargs):
    Q, q, p = kwargs["Q"], kwargs["q"], kwargs["p"]
    return jnp.squeeze(0.5 * x.T @ Q @ x + q.T @ x + p)

# Set the hyperparameters
Q = jnp.array([[48.0, 12.0], [8.0, 8.0]])
q = jnp.array([[13.0], [23.0]])
p = 4.0
x_init_f1 = jnp.array([[23.0], [37.0]])

# Initialize the solver
solution = Solver(func1, 'Analytical_stepsize', Q=Q, q=q, p=p)

# Run the optimization loop
solution.iter_loop(x_init_f1, 100)

# Plot the results
solution.plot()
```

### Examples
The `main.py` file contains additional examples demonstrating the use of the Solver class with different functions and optimization modes. You can run this file to see the optimization process in action.

## Dependencies
### Dependencies

- JAX
- Matplotlib

To install the dependencies, run:

```bash
pip install jax matplotlib
```
