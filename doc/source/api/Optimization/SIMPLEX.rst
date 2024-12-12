.. _SimplexOptimizer:

:class:`SimplexOptimizer`
==========================

Implements the Nelder-Mead simplex algorithm for optimization.

Attributes
----------
- ``objective_function`` (callable): The objective function to be minimized.
- ``bounds`` (list of tuples): The lower and upper bounds for each dimension of the search space.
- ``n_initial_points`` (int): The number of initial points to sample.
- ``n_iter`` (int): The maximum number of iterations for the optimization process.
- ``opt_folder`` (str): The folder to store optimization results.
- ``max_no_improvement_iters`` (int): The maximum number of iterations without improvement before stopping.
- ``converged_threshold`` (float): The threshold for convergence.
- ``recorder`` (LossRecorder): Records and retrieves various metrics throughout the optimization process.

Methods
-------
- ``optimize(initial_sample_mpi=False, initial_sample_max_processes=1, shrink_mpi=False, shrink_max_process=1)``:
  Executes the simplex algorithm to optimize the objective function.

The :class:`SimplexOptimizer` class provides an interface to perform optimization using the Nelder-Mead simplex algorithm. It initializes with the objective function, search space bounds, and optimization parameters. The `optimize` method executes the optimization process, optionally sampling initial points and parallelizing operations using MPI.
