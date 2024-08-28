from Optimization_algorithms import *
import matplotlib.pyplot as plt
import time

class Solver:
    '''
    A general purpose class, defined to take in arbitrary functions to implement different optimization algorithms.
    Args:
        :func: Function to minimize
        :mode: Minimization algorithms. currently 
               'Analytical_stepsize', 'Gradient_descent',
               'Golden_search_gradient_descent',
               'Newton', 'Mixed' are available modes.
        :**kwargs: Please input the hyperparameters 
                   of your function as key word arguments.

    Methods:
        :iter_loop: implemented method to minimize the objective function
        :plot: implemented method to plot contour of the objective function.
    '''
    def __init__(self, func, mode, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.x_memory = list()
        self.value_memory = list()
        self.mode_initial = None
        mode = mode.lower()
        if mode == 'analytical_stepsize':
            self.mode = analyt_GD
            assert 'Q' in kwargs, "Parameter 'Q' is missing. This mode is only implemented for quadratic function!"
            assert 'q' in kwargs, "Parameter 'q' is missing. This mode is only implemented for quadratic function!"
            assert 'p' in kwargs, "Parameter 'p' is missing. This mode is only implemented for quadratic function!"

        elif mode == 'gradient_descent':
            self.mode = GD
            assert 'learning_rate' in kwargs, "Parameter 'learning_rate' is missing."
            if 'Q' in self.kwargs:
                num_min_check(self.kwargs)

        elif mode == 'golden_search_gradient_descent':
            self.mode = Gsearch_GD
            assert 'h' in kwargs, "Parameter 'h' is missing."
        
        elif mode == 'mixed':
            self.mode_initial = 'Mixed'
            self.mode = GD
            assert 'learning_rate' in kwargs, "Parameter 'learning_rate' is missing."

        elif mode == 'newton':
            self.mode = Newton
            if 'Q' in self.kwargs:
                num_min_check(self.kwargs)

        else:
            raise NotImplementedError(f'{mode} is not implemented please choose from: "Analytical_stepsize", "Gradient_descent", "Golden_search_gradient_descent", "Mixed", "Newton"')

    def iter_loop(self, x_init, epochs, patience=5):
        x_k = x_init
        self.x_memory.append(x_k)
        try:
            self.value_memory.append(self.func(x_init, **self.kwargs))
        except:
            raise TypeError('Please add **kwargs as your function input!')
        lowest_val = 10e5
        epoch = 0
        counter = 0
        flag_mixed = False
        start_time = time.time()
        while epoch < epochs:
            x_k, value, g_k, alpha = self.mode(self.func, x_k, **self.kwargs)

            if jnp.isnan(x_k).all():
                break

            self.x_memory.append(x_k)
            self.value_memory.append(value)
            # print(f'Iter: {epoch+1}, value: {value}, x_k: {x_k}') # Uncomment this if you want to see each iteration details.
            epoch += 1
            if value < lowest_val:
                lowest_val = value
                counter = 0
            else:
                counter += 1

            
            if counter >= patience:
                if self.mode_initial != 'Mixed' or flag_mixed:
                    print('Early stopping activated...')
                    break
                elif self.mode_initial == 'Mixed':
                    self.mode = Quasi_Newton
                    print(f"Minimum value of the gradient descent: {lowest_val} in iteration {epoch}")
                    print("Method switched to Quasi-newton algorithm")
                    flag_mixed = True

        end_time = time.time()
        self.x_memory = jnp.array(self.x_memory)
        self.value_memory = jnp.array(self.value_memory)

        arg_lowest_val = jnp.argmin(self.value_memory)
        lowest_value = self.value_memory[arg_lowest_val]
        x_lowest_value = self.x_memory[arg_lowest_val,:]

        print(f'Converged to: {x_lowest_value}')
        print(f'Total time: {end_time - start_time}')
        print(f'Duration of each iteration: {(end_time - start_time) / len(self.value_memory)}')
        print(f'Value of the function in that point: {lowest_value}')
        print(f'Number of Iterations to convergence: {arg_lowest_val}')

            

    def plot(self, **kwargs):
        assert len(self.x_memory[0]) <= 2, 'Make sure your function has two variables if you want to plot contours!'
        x_final_lin = 3.0*(self.x_memory[0,0] + self.x_memory[-1,0]).item()
        y_final_lin = 3.0*(self.x_memory[0,1] + self.x_memory[-1,1]).item()
        if 'x_final_lin' in kwargs:
            x_final_lin = kwargs['x_final_lin']

        if 'y_final_lin' in kwargs:
            y_final_lin = kwargs['y_final_lin']

        x1 = jnp.linspace(-x_final_lin, x_final_lin, 100)
        x2 = jnp.linspace(-y_final_lin, y_final_lin, 100)
        X1, X2 = jnp.meshgrid(x1, x2)

        x = jnp.vstack([X1.ravel(), X2.ravel()])
        Z = self.func(x, **self.kwargs)

        if len(Z.shape) == 2:
            Z = jnp.diag(Z)


        Z = jnp.array(Z).reshape(X1.shape)

        fig, ax = plt.subplots()
        cp = ax.contour(X1, X2, Z, levels=20)
        ax.clabel(cp, inline=True, fontsize=10)
        ax.set_title('Function Contour')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        ax.scatter(self.x_memory[:,0], self.x_memory[:, 1])
        ax.plot(self.x_memory[:,0], self.x_memory[:, 1])

        plt.show()
