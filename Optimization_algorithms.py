import jax
import jax.numpy as jnp


def alpha_analytical(x, Q, q):
    g = Q @ x + q
    return (g.T @ g) / (g.T @ Q @ g)


def analyt_GD(func, x_k, **kwargs):
    Q = kwargs['Q']
    q = kwargs['q']
    p = kwargs['p']
    alpha = alpha_analytical(x_k, Q, q)
    g_k = Q @ x_k + q
    x_k = x_k - alpha * g_k
    value = func(x_k, **kwargs)
    return x_k, value, g_k, alpha


def GD(func, x_k, **kwargs):
    g = jax.grad(func)(x_k, **kwargs)
    learning_rate = kwargs['learning_rate']
    d_k = -g/jnp.linalg.norm(g)
    x_k = x_k + learning_rate * d_k
    value = func(x_k, **kwargs)
    return x_k, value, g, learning_rate

def Gsearch_GD(func, x_k, **kwargs):
    h = kwargs['h']
    f_prime_x = jax.grad(func)(x_k, **kwargs)
    g = lambda alpha: func(x_k - alpha * f_prime_x, **kwargs)

    learning_rate = golden_search(g, 0, h)

    x_k = x_k - learning_rate * f_prime_x
    value = func(x_k, **kwargs)

    return x_k, value, g, learning_rate
    
def golden_search(g, a, b, tol=1e-7):
    ratio = 2 / ((5)**(0.5) + 1)
    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)

    f1 = g(x1)
    f2 = g(x2)

    while (jnp.abs(b - a) > tol):
        if f2 > f1:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - ratio * (b - a)
            f1 = g(x1)

        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + ratio * (b - a)
            f2 = g(x2)

    return (a + b)/2


def Newton(func, x_k, **kwargs):
    hessian_func = jnp.squeeze(jax.hessian(func)(x_k, **kwargs))
    eigen_values = jnp.linalg.eigvals(hessian_func)
    if (eigen_values > 0).all():
        pass
    else:
        raise ZeroDivisionError('Hessian is not positive definite.')

    grad_func = jax.grad(func)(x_k, **kwargs)

    alpha_d_k = -jnp.linalg.inv(hessian_func) @ grad_func
    x_k = x_k + alpha_d_k
    value = func(x_k, **kwargs)

    return x_k, value, grad_func, alpha_d_k

def Quasi_Newton(func, x_k, **kwargs):
    hessian_func = jnp.squeeze(jax.hessian(func)(x_k, **kwargs))
    eigen_values = jnp.linalg.eigvals(hessian_func)
    lr = kwargs['learning_rate']
    if (eigen_values > 0).all():
        pass
    else:
        hessian_func = lr*jnp.eye(len(hessian_func)) + hessian_func

    grad_func = jax.grad(func)(x_k, **kwargs)

    alpha_d_k = -jnp.linalg.inv(hessian_func) @ grad_func
    x_k = x_k + alpha_d_k
    value = func(x_k, **kwargs)

    return x_k, value, grad_func, alpha_d_k
    

def num_min_check(kwargs):
    Q = kwargs['Q']
    q = kwargs['q']
    print(f'The objective function is quadratic...')
    eigen_values = jnp.linalg.eigvals(kwargs['Q'])
    if (eigen_values > 0).all():
        print(f'Quadratic function has exactly one minimum.')

    elif (eigen_values >= 0).all():
        Q_pinv = jnp.linalg.pinv(Q)
        proj_Q = jnp.dot(Q, jnp.dot(Q_pinv, q)) 

        if jnp.allclose(q, proj_Q):
            print("q is in the range of Q")
            print('Quadratic function has multiple minima')
        else:
            print("q is not in the range of Q")
            print(f'Quadratic function has no minimum.')

    elif (eigen_values < 0).all():
        print(f'Quadratic function has exactly one maximum.')

    elif (eigen_values <= 0).all():
        print('Quadratic function has no minimum.')

    else:
        print('Quadratic function Does not have maxima or minima.')

