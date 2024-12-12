:class:`Particle`
===================
Represents a single particle in the Particle Swarm Optimization (PSO) algorithm.

Attributes
----------
- ``position`` (numpy.ndarray): The current position of the particle in the search space.
- ``velocity`` (numpy.ndarray): The current velocity of the particle.
- ``best_position`` (numpy.ndarray): The best position discovered by the particle.
- ``best_score`` (float): The objective function value corresponding to the best position.
- ``idx`` (int): An identifier for the particle.
- ``iter`` (int): The current iteration number for the particle.

Methods
-------
- ``__init__(self, bounds, init_position='random')``: Initializes a new particle with a random or specified initial position and zero initial velocity.


.. _ParticleSwarmOptimizer:

:class:`ParticleSwarmOptimizer`
================================
Implements the Particle Swarm Optimization (PSO) algorithm for finding the minimum of an objective function in a multi-dimensional space.

Attributes
----------
- ``objective_function`` (callable): The objective function to be minimized.
- ``bounds`` (list of tuples): Defines the search space through lower and upper bounds for each parameter.
- ``num_particles`` (int): The number of particles in the swarm.
- ``max_iter`` (int): The maximum number of iterations for optimization.
- ``converged_threshold`` (float): The convergence threshold for optimization termination.
- ``recorder`` (LossRecorder): An instance of the LossRecorder class to track losses and optimization parameters.
- ``iter`` (int): The current iteration number.
- ``options`` (dict): Configuration options for PSO.

Methods
-------
- ``__init__(self, objective_function, bounds, update_boundary_frequency=5, begin_update_boundary_frequency_iter=0, converged_threshold=1e-6, max_no_improvement_iters=8, num_particles=30, max_iter=100, options=None)``: Initializes the ParticleSwarmOptimizer with objective function, search space bounds, and optimization parameters.
- ``update_inertia_weight(self, iter)``: Adjusts the inertia weight based on the current iteration for exploration and exploitation balance.
- ``adaptive_bounds(self, current_iter)``: Dynamically adjusts the search space bounds based on the global best position for exploration.
- ``optimize(self, opt_folder=None)``: Executes the PSO algorithm to minimize the objective function within defined bounds.
- ``optimize_mpi(self, max_processes, opt_folder=None)``: Executes PSO in parallel using multiple processes for faster optimization.

``optimize_mpi`` is a parallel version of ``optimize`` that utilizes multiprocessing to speed up the computation.

``optimize`` executes PSO sequentially, while ``optimize_mpi`` executes PSO in parallel, leveraging multiple processes for faster computation.
``optimize_mpi`` requires specifying the maximum number of worker processes for parallel computation.
``optimize_mpi`` also provides an optional parameter to specify an output folder for storing optimization artifacts.

``update_inertia_weight``, ``adaptive_bounds``, and ``optimize`` methods handle different aspects of the PSO algorithm, such as adjusting inertia weight, dynamically updating search space bounds, and executing the optimization process.

The ``options`` parameter in the ``ParticleSwarmOptimizer`` class allows users to customize PSO parameters such as inertia weight, acceleration coefficients, etc.
``options`` provide flexibility for fine-tuning the PSO algorithm according to specific optimization tasks.

The ``recorder`` attribute of the ``ParticleSwarmOptimizer`` class is an instance of the ``LossRecorder`` class, used for recording losses and optimization parameters over iterations.
``recorder`` facilitates monitoring and analyzing optimization progress, aiding in result interpretation and troubleshooting.
``recorder`` records losses, global best parameters, and iteration details for comprehensive optimization tracking and analysis.



