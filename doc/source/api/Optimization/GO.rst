:class:`Individual`
====================

Represents an individual solution within an optimization algorithm.

Attributes
----------
- ``iter`` (int): The iteration number or generation in which the individual exists.
- ``idx`` (int): The index or identifier of the individual within its population.
- ``position`` (array-like): The parameter values or position of the individual in the search space.

Methods
-------
- ``__init__(self, individual_array)``:
  Initializes an individual with the given parameter values.

The ``Individual`` class encapsulates the properties of an individual solution in an optimization algorithm. It includes attributes such as the iteration number (`iter`), index (`idx`), and position (`position`) in the search space.

.. _GeneticOptimizer:

:class:`GeneticOptimizer`
==========================
Implements a Genetic Algorithm (GA) for optimization. The GA mimics natural selection by evolving a population of solutions over generations through selection, crossover, and mutation.

Attributes
----------
- ``objective_function`` (callable): The objective function to minimize.
- ``bounds`` (list of tuples): Search space bounds for each parameter.
- ``method`` (str, optional): Method for diversity maintenance or specific optimization strategies.
- ``population_size`` (int): Number of individuals in the population.
- ``mutation_rate`` (float): Probability of mutation for each gene.
- ``crossover_rate`` (float): Probability of crossover between two parents.
- ``max_generations`` (int): Maximum number of generations to evolve.
- ``recorder`` (LossRecorder): Records and retrieves optimization metrics.

Methods
-------
- ``__init__(self, objective_function, bounds, method=None, population_size=50, mutation_rate=0.1, crossover_rate=0.8, max_generations=100)``: Initializes the GeneticOptimizer.
- ``initialize_population()``: Creates an initial population of random solutions within the search space bounds.
- ``calculate_fitness(population)``: Evaluates the fitness (objective function value) of each individual in the population.
- ``select(population, fitness_scores)``: Selects individuals for the next generation based on fitness scores.
- ``crossover(parent1, parent2, strategy='single_point')``: Performs crossover between two parents to produce offspring.
- ``mutate(individual)``: Applies mutation to an individual's genes.
- ``optimize()``: Executes the genetic algorithm to find the best solution.
- ``optimize_mpi(max_processes=1, opt_folder=None)``: Executes genetic algorithm in parallel using MPI.

The ``GeneticOptimizer`` class implements a Genetic Algorithm for optimization, where a population of solutions evolves over generations through selection, crossover, and mutation. Attributes include the objective function, search space bounds, population size, mutation and crossover rates, and maximum generations.

Methods such as ``initialize_population``, ``calculate_fitness``, ``select``, ``crossover``, and ``mutate`` handle various aspects of the GA operations. The ``optimize`` method executes the GA, iterating over generations to find the best solution, while ``optimize_mpi`` parallelizes the process using MPI.

The ``recorder`` attribute facilitates tracking of optimization metrics throughout the process.
