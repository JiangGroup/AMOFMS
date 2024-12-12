.. _BayesianOptimizer:

:class:`BayesianOptimizer`
===========================
Implements Bayesian Optimization using Gaussian Processes (GP) to find the minimum of an objective function.

Attributes
----------
- ``objective_function`` (callable): The objective function to be minimized.
- ``bounds`` (list of tuples): The bounds for each dimension of the input parameters.
- ``n_initial_points`` (int): The number of initial random samples.
- ``n_iter`` (int): The number of iterations to perform after initial sampling.
- ``opt_folder`` (str): The directory where optimization artifacts will be stored.
- ``recorder`` (LossRecorder): An instance of the LossRecorder class to track losses and optimization parameters.
- ``max_no_improvement_iters`` (int): Maximum number of iterations without improvement for convergence.
- ``converged_threshold`` (float): Convergence threshold for optimization termination.
- ``gp`` (GaussianProcessRegressor): The Gaussian Process model used for function approximation.
- ``X_samples`` (list): List of sampled input parameters.
- ``Y_samples`` (list): List of objective function values corresponding to input samples.

Methods
-------
- ``__init__(self, objective_function, bounds, opt_folder, n_initial_points=5, max_iter=25, max_no_improvement_iters=8, converged_threshold=1e-6)``: Initializes the BayesianOptimizer.
- ``sample_initial_points(self)``: Randomly samples initial points within the defined bounds.
- ``sample_single_point(self, idx)``: Samples a single initial point within the bounds (used in parallel sampling).
- ``sample_initial_points_mpi(self, max_processes=1)``: Samples initial points in parallel using MPI.
- ``expected_improvement(self, x)``: Calculates the Expected Improvement at a given point.
- ``optimize(self, initial_sample_mpi=False, initial_sample_max_processes=1)``: Executes Bayesian Optimization.

The ``BayesianOptimizer`` class implements Bayesian Optimization, a technique for optimizing expensive-to-evaluate functions by building a probabilistic surrogate model (GP) of the objective function and iteratively selecting points to evaluate based on an acquisition function (expected improvement).

The attributes of this class include the objective function, search space bounds, initial sample size, number of iterations, optimization folder, convergence parameters, GP model, and lists to store sampled points and function values.

The ``sample_initial_points`` method randomly samples initial points within the bounds, while ``sample_initial_points_mpi`` allows parallel sampling using MPI.

``expected_improvement`` computes the Expected Improvement acquisition function at a given point, guiding the selection of the next point to evaluate.

The main optimization process is performed by the ``optimize`` method, which iteratively fits the GP model, selects points to evaluate, updates the GP with new observations, and records optimization progress.

The ``recorder`` attribute enables tracking of optimization progress, including iteration details and convergence metrics, facilitating analysis and troubleshooting.

``sample_single_point`` method is used internally by ``sample_initial_points_mpi`` for parallel sampling of initial points.




