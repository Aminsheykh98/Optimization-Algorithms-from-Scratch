import jax
import jax.numpy as jnp
from jax import grad
from Optimization_algorithms import *
from Solver_class import *
import matplotlib.pyplot as plt




def func1(x, **kwargs):
    Q, q, p = kwargs['Q'], kwargs['q'], kwargs['p']
    return jnp.squeeze(0.5 * x.T @ Q @ x + q.T @ x + p)

def func2(x, **kwargs):
    size_x = len(x)
    b, a = kwargs['b'], kwargs['a']
    values = jnp.array([(b*(x[i+1]**2 - x[i]**2) + (x[i] - a)**2) for i in range(size_x - 1)])
    val = jnp.sum(values, axis=0)
    return jnp.squeeze(val)

def func3(x, **kwargs):
    return jnp.squeeze((x[0] - 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4)


## Hyper Parameters of the Quadratic function
Q = jnp.array([[48.0, 12.0], [8.0, 8.0]])
q = jnp.array([[13.0], [23.0]])
p = 4.0
x_init_f1 = jnp.array([[23.0], [37.0]])

## Hyper Parameters of the Second function
b = 150
a = -2
x_init_f2 = jnp.array([[1.0], [2.0]])

## Hyper Parameters of the third function
x_init_f3 = jnp.array([1.0, 2.0, 2.0, 2.0]).reshape(4,1)

## Sample to initialize Solver for function 1
solution = Solver(func1, 'mixed', Q=Q, q=q, p=p, h=1, learning_rate=0.5, patience=5) # Initializing solver class
solution.iter_loop(x_init_f1, 100) # Finding minimum
solution.plot(x_final_lin=40, y_final_lin=40) # Plotting the contour


## Sample to initialize Solver for function 2
# solution = Solver(func2, 'mixed', a=a, b=b, h=1, learning_rate=0.5, patience=5) # Initializing solver class
# solution.iter_loop(x_init_f2, 100) # Finding minimum
# solution.plot(x_final_lin=40, y_final_lin=40) # Plotting the contour


## Sample to initialize Solver for function 3
# solution = Solver(func3, 'mixed', a=a, b=b, h=1, learning_rate=0.5, patience=5) # Initializing solver class
# solution.iter_loop(x_init_f3, 100) # Finding minimum
# # solution.plot(x_final_lin=40, y_final_lin=40) # Returns error.


