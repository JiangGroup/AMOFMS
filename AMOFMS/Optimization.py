import os
import copy
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from tools.properties import *
from tools.utilies import mkdir


class LossRecorder:
    """
    A class to record and retrieve losses for various terms across iterations during a training process.
    This class allows tracking of the total loss and individual term losses per iteration,
    providing insights into the training dynamics and performance.

    Attributes:
    -----------
    iteration_losses : dict
        Stores the total and individual term losses for each iteration. The key is the iteration number,
        and the value is another dictionary containing 'total_loss' and each term's loss.
    iteration_opt_parameters : dict
        Keeps track of the optimization parameters used in each iteration. The key is the iteration number,
        and the value is the parameters used in that iteration.
    header_written : bool
        Indicates if the header has been written to the output file. This ensures the header is only written once.

    Methods:
    --------
    record_loss(iteration, opt_parameters, total_loss, each_term_loss, write_iteration_loss_filepath=None):
        Records the losses for a given iteration and optionally writes them to a specified file.

    get_iteration_loss(iteration):
        Retrieves the loss record for a specific iteration, including total loss and individual term losses.

    get_losses():
        Returns a dictionary of all recorded losses for every iteration.

    get_term_loss(term):
        Retrieves a list of losses for a specified term across all recorded iterations.

    get_min_loss_and_iteration(term):
        Identifies the minimum loss value for a specified term across all iterations and the corresponding iteration number.

    write_losses_to_file(filepath):
        Writes the recorded losses to a file, formatting the output for readability.
    """

    def __init__(self):
        """
        Initializes the LossRecorder class, setting up the necessary attributes for storing loss records
        and tracking whether the header has been written to a file.
        """

        self.iteration_losses = {}  # Initializes to store losses per iteration
        self.iteration_opt_parameters = {}  # Initializes to store optimization parameters
        self.header_written = False  # Initially set to False, indicating that the header has not been written to a file

    def record_loss(self, iteration, opt_parameters, total_loss, each_term_loss, write_iteration_loss_filepath=None, note=None):
        """
        Records the total loss and individual term losses for a specific iteration. Optionally writes this information
        to a specified file path.

        Parameters:
        -----------
        iteration : int
            The current iteration number.
        opt_parameters : dict or list
            The optimization parameters used in the current iteration.
        total_loss : float
            The total loss value for the current iteration.
        each_term_loss : dict
            A dictionary containing the individual term losses for the current iteration.
        write_iteration_loss_filepath : str, optional
            The file path where to write the iteration loss data. If None, data is not written to a file.
        note : str, optional
            Additional note for the loss.
        """

        # Create a record for this iteration.
        loss_record = {'total_loss': total_loss}

        # Add each term loss to the record.
        for term, value in each_term_loss.items():
            loss_record[term] = value

        # Store this record in the iteration_losses dictionary.
        self.iteration_losses[iteration] = loss_record
        self.iteration_opt_parameters[iteration] = opt_parameters

        # If a file path is provided, formats and writes the iteration's loss data to the file
        if write_iteration_loss_filepath is not None:
            # Format and write this iteration's record to the file.
            with open(write_iteration_loss_filepath, 'a+') as file:

                # Writes the header line only once
                if not self.header_written:
                    title_line = '{:<12}'.format('Iteration')
                    column_widths = {'total_loss': 10}
                    for key in self.iteration_losses[0].keys():
                        column_width = max(len(key), 10)
                        column_widths[key] = column_width
                        title_line += '{:<{width}}'.format(key, width=column_width + 2)

                    file.write(title_line + '\n')
                    self.header_written = True  # Update flag to indicate the header has been written

                # Write the iteration number.
                line = '{:<12}'.format(iteration)

                # Write the total loss and each term loss.
                for key, val in loss_record.items():
                    column_width = max(len(key), 10)
                    line += '{:<{width}}'.format(round(val, 6), width=column_width + 2)

                if note is not None:
                    line = line + f'\t {note}'

                # Append the formatted line to the file.
                file.write(line + '\n')


    def get_iteration_loss(self, iteration):
        """
        Retrieves the loss record for a specific iteration.

        Parameters:
        -----------
        iteration : int
            The iteration number for which to retrieve the loss record.

        Returns:
        --------
        dict
            The loss record for the specified iteration. If the iteration does not exist, returns None.
        """

        # Returns the loss record for the specified iteration, or None if not found
        return self.iteration_losses[iteration]

    def get_losses(self):
        """
        Retrieves the loss records for all iterations.

        Returns:
        --------
        dict
            The complete dictionary of iteration losses.
        """

        # Returns all recorded iteration losses
        return self.iteration_losses

    def get_term_loss(self, term):
        """
        Retrieves a list of losses for a specified term across all iterations.

        Parameters:
        -----------
        term : str
            The term for which to retrieve losses.

        Returns:
        --------
        list
            A list of losses for the specified term across all iterations. If a term is not found in an iteration,
            None is included in the list for that iteration.
        """

        # Iterates over all iterations and compiles a list of losses for the specified term
        term_losses = []
        # Iterate over all stored iterations.
        for iteration in self.iteration_losses:
            # Append the loss for the specified term, or None if the term is not found.
            term_losses.append(self.iteration_losses[iteration].get(term, None))
        return term_losses

    def get_min_loss_and_iteration(self, term):
        """
        Finds the minimum loss value for a specified term and its corresponding iteration number.

        Parameters:
        -----------
        term : str
            The term for which to find the minimum loss.

        Returns:
        --------
        tuple
            A tuple containing the minimum loss and its iteration number (float, int).
            If the term is not found, returns (None, None).
        """

        # Searches for the minimum loss and its iteration for the specified term
        min_loss = None
        min_iteration = None
        # Iterate over all stored iterations.
        for iteration, losses in self.iteration_losses.items():
            # Check if the current iteration's loss for the term is less than the current minimum.
            if term in losses and (min_loss is None or losses[term] < min_loss):
                min_loss = losses[term]
                min_iteration = iteration
        return min_loss, min_iteration

    def write_losses_to_file(self, filepath):
        """
        Writes all recorded losses to a specified file, formatting the output to align columns for better readability.

        Parameters:
        -----------
        filepath : str
            The path to the file where the losses should be written.
        """

        # Writes the loss records to a file with formatted, aligned columns
        with open(filepath, 'w') as file:
            # Constructs and writes the title line with column names
            title_line = '{:<12}'.format('Iteration')
            column_widths = {}
            for key in self.iteration_losses[0].keys():
                column_width = max(len(key), 10)  # set the minimum length to 10
                column_widths[key] = column_width
                title_line += '{:<{width}}'.format(key, width=column_width + 2)
            file.write(title_line + '\n')

            # Writes each iteration's losses
            for iteration, values in self.iteration_losses.items():
                line = '{:<12}'.format(iteration)
                for key, val in values.items():
                    column_width = column_widths[key]
                    line += '{:<{width}}'.format(round(val, 6), width=column_width + 2)
                file.write(line + '\n')


class OptParametersProcess:
    """
    A class to manage the optimization of parameters within molecular dynamics simulations. It processes
    optimization parameters, including handling the conversion between simulation-readable formats and
    optimization algorithm formats.

    Attributes:
    -----------
    system_topology : dict
        Stores the system's molecular topology information.
    opt_term_parse : dict
        Contains details on which parameters are to be optimized.
    equivalent_term_id_list : list
        Maps parameters to their equivalent terms for optimization purposes.
    equivalent_term_boundary_value_dict : dict
        Specifies boundary values for equivalent optimization terms.
    opt_parameter_boundary_array : numpy.ndarray
        An array containing the lower and upper boundaries for each optimization parameter.

    Methods:
    --------
    __init__(self, system_topology, opt_term_parse, equivalent_term_id_list=None, equivalent_term_boundary_value_dict=None):
        Initializes the OptParametersProcess class with the system topology and optimization parameters. This setup includes the handling of equivalent terms for grouped optimization.

    pack_opt_parameters_to_boundary_array(self):
        Converts the optimization parameters defined in the system topology into a structured array that delineates their lower and upper boundaries. This array is essential for informing optimization algorithms about the permissible range of each parameter.

    unpack_updated_parameters_to_top(self, updated_parameters_array):
        Updates the system topology with the optimized parameter values obtained from the optimization process. This method ensures that the optimized parameters are correctly integrated back into the system's topology, ready for further simulation or analysis.

    adjust_for_equivalent_terms(self, boundary_array):
        (Private Method) Adjusts the parameter boundary array to account for equivalent terms as defined by `equivalent_term_id_list`. This ensures that parameters identified as equivalent are optimized within the same boundary constraints, maintaining their intended relationships.

    get_optimized_parameters(self):
        Retrieves the optimized parameters from the current system topology. This method is useful for extracting the optimized parameters for analysis, reporting, or applying them to a different system setup.

    save_optimized_parameters(self, filepath):
        Saves the optimized parameters to a specified file. This functionality allows for the preservation of optimized parameters for future use, such as initializing different simulations with these parameters or for documentation purposes.

    load_optimized_parameters(self, filepath):
        Loads optimized parameters from a specified file into the system topology. This method enables the use of previously optimized parameters, facilitating reproducibility and the application of optimized conditions to new simulations or analyses.
    """

    def __init__(self, system_topology, opt_term_parse, equivalent_term_id_list=None, equivalent_term_boundary_value_dict=None):
        """
        Initializes the OptParametersProcess class with system topology and optimization terms.

        Ensures that all necessary information for parameter optimization is set up, including handling
        equivalent terms and their boundaries.

        Parameters:
        -----------
        system_topology : dict
            A dictionary containing detailed information about the molecular system, including molecules' properties
            such as mass, charge, and bonding parameters.
        opt_term_parse : dict
            A dictionary specifying which parameters need to be optimized and the range of their optimization.
        equivalent_term_id_list : list, optional
            A list that maps optimization parameters to their equivalent terms, allowing for grouped optimization
            of parameters that should be treated as equivalent.
        equivalent_term_boundary_value_dict : dict, optional
            A dictionary specifying boundary values for the equivalent terms defined in equivalent_term_id_list.
        """
        self.system_topology = system_topology  # Stores the molecular system topology data.
        self.opt_term_parse = opt_term_parse   # Stores optimization terms parsing data that determine the which parameter would be optimzied and its range.

        if equivalent_term_id_list is not None:
            if equivalent_term_boundary_value_dict is not None:
                self.equivalent_term_id_list = equivalent_term_id_list
                self.equivalent_term_boundary_value_dict = equivalent_term_boundary_value_dict
            else:
                print('equivalent_term_boundary_value must be provided')

        # Pack parameters into a boundary array for optimization
        self.opt_parameter_boundary_array = self.pack_opt_parameters_to_boundary_array()

    def pack_opt_parameters_to_boundary_array(self):
        """
        Converts the optimization parameters and their ranges from the system topology into a structured array.
        This array is used by optimization algorithms to understand parameter boundaries.

        Returns:
        --------
        numpy.ndarray
            An array of tuples, each representing the lower and upper boundary for an optimization parameter.
        """

        boundary_array = []  # Initialize an array to store parameter boundaries.

        # Iterate over each molecule's optimization items
        for idx, mol in enumerate(self.opt_term_parse['molecules']):
            # Handle optimization of charge parameters
            if 'charge' in mol:
                for item_idx, i in enumerate(mol['charge']):
                    if i:  # Only process parameters that need optimization (non-zero value).
                        item =self.system_topology['molecules'][idx]['charge'][item_idx]
                        boundary_array.append([item-item*i, item+item*i])

            # Similarly, handle optimization of mass parameters
            if 'mass' in mol:
                for item_idx, i in enumerate(mol['mass']):
                    if i:
                        item =self.system_topology['molecules'][idx]['mass'][item_idx]
                        boundary_array.append([item-item*i, item+item*i])

            # Handle optimization of bond parameters
            if 'bond_parameters' in mol:
                for item_idx, i in enumerate(mol['bond_parameters'].keys()):
                    items = mol['bond_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            item =self.system_topology['molecules'][idx]['bond_parameters'][i][m]
                            boundary_array.append([item-item*n, item+item*n])

            # Handle optimization of angle parameters
            if 'angle_parameters' in mol:
                for item_idx, i in enumerate(mol['angle_parameters'].keys()):
                    items = mol['angle_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            item =self.system_topology['molecules'][idx]['angle_parameters'][i][m]
                            boundary_array.append([item-item*n, item+item*n])

            # Handle optimization of dihedral parameters, if present
            if 'dihedral_parameters' in mol:
                for item_idx, i in enumerate(mol['dihedral_parameters'].keys()):
                    items = mol['dihedral_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            item =self.system_topology['molecules'][idx]['dihedral_parameters'][i][m]
                            boundary_array.append([item-item*n, item+item*n])

        # Handle optimization of Lennard-Jones cross-term parameters
        if 'lj_cross_terms' in self.opt_term_parse:
            for item_idx, i in enumerate(self.opt_term_parse['lj_cross_terms'].keys()):
                items = self.opt_term_parse['lj_cross_terms'][i]
                for m, n in enumerate(items):
                    if n:
                        if i in self.system_topology['lj_cross_terms']:
                            item = self.system_topology['lj_cross_terms'][i][m]
                        elif i[::-1] in self.system_topology['lj_cross_terms']:
                            item = self.system_topology['lj_cross_terms'][i[::-1]][m]
                        else:
                            print("the bond dose not exist.")
                            item = None
                        boundary_array.append([item-item*n, item+item*n])

        self.original_boundary_array = boundary_array

        if hasattr(self, 'equivalent_term_id_list'):
            degenerate_boundart_array = []
            ids = []
            for idx, i in enumerate(self.equivalent_term_id_list):
                if i not in ids:
                    ids.append(i)
                    if i in self.equivalent_term_boundary_value_dict:
                        degenerate_boundart_array.append(self.equivalent_term_boundary_value_dict[i])
                    else:
                        degenerate_boundart_array.append(boundary_array[idx])
                else:
                    continue
            boundary_array = degenerate_boundart_array

        # Return an array containing the boundaries of all optimization parameters.
        return np.array(boundary_array)

    # Similar to the pack method, iterate over optimization items and update system topology data
    def unpack_updated_parameters_to_top(self, updated_parameters_array):
        """
        Updates the system topology with optimized parameter values based on the results from the optimization algorithm.

        Parameters:
        -----------
        updated_parameters_array : numpy.ndarray
            An array containing the optimized values for the parameters.

        Returns:
        --------
        dict
            A dictionary representing the updated system topology with the new optimized parameters.
        """

        if hasattr(self, 'equivalent_term_id_list'):
            full_boundary_array = []
            for i in self.equivalent_term_id_list:
                full_boundary_array.append(updated_parameters_array[i])
            updated_parameters_array = full_boundary_array

        assert len(self.original_boundary_array) == len(updated_parameters_array)

        new_system_topology = copy.deepcopy(self.system_topology)
        # Unpack the optimized parameters back into the system topology
        # Placeholder for actual unpacking logic

        count = 0  # Initialize a parameter index counter.

        # Similar to the pack method, iterate over optimization items and update system topology data
        for idx, mol in enumerate(self.opt_term_parse['molecules']):
            # Update charge parameters if they were marked for optimization
            if 'charge' in mol:
                for item_idx, i in enumerate(mol['charge']):
                    if i:
                        # Update the charge value in the new topology
                        new_system_topology['molecules'][idx]['charge'][item_idx] = updated_parameters_array[count]
                        count += 1

            # Update mass parameters
            if 'mass' in mol:
                for item_idx, i in enumerate(mol['mass']):
                    if i:
                        new_system_topology['molecules'][idx]['mass'][item_idx] = updated_parameters_array[count]
                        count += 1

            # Update bond parameters
            if 'bond_parameters' in mol:
                for item_idx, i in enumerate(mol['bond_parameters'].keys()):
                    items = mol['bond_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            new_system_topology['molecules'][idx]['bond_parameters'][i][m] = updated_parameters_array[count]
                            count += 1

            # Update angle parameters
            if 'angle_parameters' in mol:
                for item_idx, i in enumerate(mol['angle_parameters'].keys()):
                    items = mol['angle_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            new_system_topology['molecules'][idx]['angle_parameters'][i][m] = updated_parameters_array[count]
                            count += 1

            # Update dihedral parameters, if they exist
            if 'dihedral_parameters' in mol:
                for item_idx, i in enumerate(mol['dihedral_parameters'].keys()):
                    items = mol['dihedral_parameters'][i]
                    for m, n in enumerate(items):
                        if n:
                            new_system_topology['molecules'][idx]['dihedral_parameters'][i][m] = updated_parameters_array[count]
                            count += 1

        # Update Lennard-Jones cross-term parameters
        if 'lj_cross_terms' in self.opt_term_parse:
            for item_idx, i in enumerate(self.opt_term_parse['lj_cross_terms'].keys()):
                items = self.opt_term_parse['lj_cross_terms'][i]
                for m, n in enumerate(items):
                    if n:
                        if i in self.system_topology['lj_cross_terms']:
                            new_system_topology['lj_cross_terms'][i][m] = updated_parameters_array[count]
                        elif i[::-1] in self.system_topology['lj_cross_terms']:
                            new_system_topology['lj_cross_terms'][i[::-1]][m] = updated_parameters_array[count]
                        count += 1

        # Return the updated system topology
        return new_system_topology


class Particle:
    """
    Represents a single particle in the Particle Swarm Optimization (PSO) algorithm. A particle is a potential solution in the search space, characterized by its position, velocity, and the best position it has discovered.

    Attributes:
    -----------
    position : numpy.ndarray
        The current position of the particle in the search space, representing a potential solution.
    velocity : numpy.ndarray
        The current velocity of the particle, influencing how its position changes in the next iteration.
    best_position : numpy.ndarray
        The best position (i.e., solution) this particle has discovered based on the objective function value.
    best_score : float
        The objective function value corresponding to the best position.
    idx : int
        An identifier for the particle, useful for tracking and debugging.
    iter : int
        The current iteration number for the particle, used to track its progress over time.
    """

    def __init__(self, bounds, init_position='random'):
        """
        Initializes a new particle with a random or specified initial position and zero initial velocity.

        Parameters:
        -----------
        bounds : list of tuples
            Each tuple contains the lower and upper bounds for a parameter, defining the search space.
        init_position : str, optional
            Specifies how to initialize the particle's position. 'random' for random initialization within the bounds, 'middle' to start at the midpoint of each bound.
        """
        if init_position == 'middle':  # do not set middle with the 0 velocity, otherwise the position would not be updated every iteration
            self.position = np.array([np.round(np.mean([low, high]), 6) for low, high in bounds])
        elif init_position == 'random':
            np.random.seed()
            self.position = np.array([np.round(np.random.uniform(low, high), 6) for low, high in bounds])
        self.velocity = np.zeros(len(bounds))  # Initializes velocity to zero
        self.best_position = np.copy(self.position)
        self.best_score = float('inf')  # Initializes the best score to infinity
        self.idx = 0  # Default index, should be set by the optimizer
        self.iter = 0  # Tracks iterations for this particle


class ParticleSwarmOptimizer:
    """
    Implements the Particle Swarm Optimization (PSO) algorithm for finding the minimum of an objective function in a multi-dimensional space.

    Attributes:
    -----------
    objective_function : callable
        The objective function to be minimized.
    bounds : list of tuples
        Defines the search space through a list of tuples, where each tuple (low, high) represents the bounds for a parameter.
    num_particles : int
        The number of particles in the swarm.
    max_iter : int
        The maximum number of iterations to run the optimization.
    converged_threshold : float
        The convergence threshold; optimization stops if changes in the global best score are below this threshold.
    recorder : LossRecorder
        An instance of the LossRecorder class to track the loss values and optimization parameters over iterations.
    iter : int
        The current iteration number.
    options : dict
        Configuration options for PSO, including inertia weight and acceleration coefficients.

    Methods:
    --------
    update_inertia_weight(self, iter)
        Adjusts the inertia weight based on the current iteration, facilitating control over exploration and exploitation.
    adaptive_bounds(self, current_iter)
        Dynamically adjusts the search space bounds based on the location of the global best position to encourage exploration.
    optimize(self, opt_folder=None)
        Executes the PSO algorithm, updating particles' positions and velocities over iterations to minimize the objective function.
    """

    def __init__(self, objective_function, bounds, update_boundary_frequency=5, begin_update_boundary_frequency_iter=0, converged_threshold=1e-6,
                 max_no_improvement_iters=8, num_particles=30, max_iter=100, options=None):
        """
        Initializes the ParticleSwarmOptimizer with the objective function, search space, and optimization parameters.

        Parameters:
        -----------
        objective_function : callable
            The function to be minimized.
        bounds : list of tuples
            The search space bounds.
        update_boundary_frequency : int
            How frequently (in iterations) to adjust the search space bounds.
        converged_threshold : float
            The threshold for convergence. If changes in the global best score are below this value, optimization stops.
        num_particles : int
            The number of particles in the swarm.
        max_iter : int
            The maximum number of iterations for the optimization process.
        options : dict, optional
            Additional configuration options for PSO.
        """

        self.objective_function = objective_function
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.converged_threshold = converged_threshold
        self.max_no_improvement_iters = max_no_improvement_iters

        self.recorder = LossRecorder()  # Initializes the loss recorder

        self.iter = 0  # Starts from iteration 0
        # Default PSO parameters, can be overridden by the options parameter

        self.linear_shrink_scale = 0.25
        self.begin_update_boundary_frequency_iter = begin_update_boundary_frequency_iter
        self.dynamic_bounds_update_frequency = update_boundary_frequency  # 边界调整的迭代周期
        self.initial_bounds = bounds.copy()  # 保存初始边界

        self.max_bond_scale = 0.5
        self.max_bounds = [(bound[0] - (bound[1] - bound[0]) * self.max_bond_scale,
                           bound[1] + (bound[1] - bound[0]) * self.max_bond_scale) for bound in bounds]

        self.initial_expand_scale = 0.1
        self.bounds = [(bound[0] - (bound[1] - bound[0]) * self.initial_expand_scale,
                        bound[1] + (bound[1] - bound[0]) * self.initial_expand_scale) for bound in bounds]

        self.stable_zone_scale = 0.2
        self.stable_zone = [(bound[0] + (bound[1] - bound[0]) * self.stable_zone_scale,
                             bound[1] - (bound[1] - bound[0]) * self.stable_zone_scale) for bound in self.bounds]

        # Default PSO parameters, can be overridden by the options parameter
        default_options = {
            'initial_inertia_weight': 0.9,
            'final_inertia_weight': 0.5,
            'personal_accel': 0.5,
            'global_accel': 0.5
        }

        # Updates options with any user-specified values
        self.options = default_options if options is None else {**default_options, **options}

        # Placeholder attributes for global best position and score
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.global_best_score_of_each_term = None

        self.particles = [Particle(bounds, init_position='random') for _ in range(num_particles)]
        for i in range(num_particles):
            self.particles[i].idx = i

    def update_inertia_weight(self, iter):
        """
        Dynamically adjusts the inertia weight based on the current iteration to balance global and local exploration.

        Parameters:
        -----------
        iter : int
            The current iteration number.
        """

        weight_range = self.options['initial_inertia_weight'] - self.options['final_inertia_weight']
        self.options['inertia_weight'] = self.options['initial_inertia_weight'] - weight_range * (iter / self.max_iter)

    def adaptive_bounds(self, current_iter):
        """
        Adjusts the search space bounds dynamically based on the global best position to encourage exploration and avoid stagnation.

        Parameters:
        -----------
        current_iter : int
            The current iteration number.
        """

        if current_iter % self.dynamic_bounds_update_frequency == 0 and current_iter > self.begin_update_boundary_frequency_iter:
            for i, (low, high) in enumerate(self.bounds):
                stable_low, stable_high = self.stable_zone[i]
                boundary_shift = (high - low) * self.stable_zone_scale  # 计算基于当前区间长度的20%

                if self.global_best_position[i] < stable_low:  # 全局最优在左侧动态区间
                    new_low = max(self.max_bounds[i][0], low - boundary_shift)  # 向左扩展但不超过初始边界
                    new_high = min(high - boundary_shift, self.max_bounds[i][1])  # 同时向左移动右边界
                    self.bounds[i] = (new_low, new_high)
                elif self.global_best_position[i] > stable_high:  # 全局最优在右侧动态区间
                    new_low = max(low + boundary_shift, self.max_bounds[i][0])  # 同时向右移动左边界
                    new_high = min(self.max_bounds[i][1], high + boundary_shift)  # 向右扩展但不超过初始边界
                    self.bounds[i] = (new_low, new_high)

                decrease_length = (self.initial_bounds[i][1] - self.initial_bounds[i][0]) * \
                                  self.linear_shrink_scale / self.max_iter
                new_low = min(self.bounds[i][0] + decrease_length, self.global_best_position[i])
                new_high = max(self.bounds[i][1] - decrease_length, self.global_best_position[i])
                self.bounds[i] = (new_low, new_high)

    def optimize(self, opt_folder=None):
        """
        Executes the Particle Swarm Optimization (PSO) algorithm. It iteratively updates the particles' positions and velocities,
        aiming to find the global minimum of the objective function within the defined bounds.

        Parameters:
        -----------
        opt_folder : str, optional
            The directory where optimization artifacts, such as iteration logs, can be saved. If not provided, these artifacts are not saved.

        Returns:
        --------
        tuple
            A tuple containing the global best position (numpy.ndarray) found by the optimizer, the objective function value at this position (float),
            and an instance of LossRecorder with recorded losses and optimization parameters (LossRecorder).
        """

        print('\nPSO optimization starting...')

        no_improvement_iters = 0   # Tracks iterations without improvement in the global best score
        previous_best_score = float('inf')  # Initializes the previous best score as infinity for comparison

        # Optional: create a directory for the current iteration if an output folder is specified
        for iter in range(self.max_iter):
            self.iter = iter
            if opt_folder is not None:
                mkdir(os.path.join(opt_folder, f'iter_{iter}'))

            self.update_inertia_weight(iter)  # Dynamically adjust the inertia weight based on the iteration number

            # Update each particle's position and velocity, and evaluate the objective function
            for particle in self.particles:

                particle.iter = iter
                score, score_of_each_term = self.objective_function(particle)
                # Update particle's personal best position and score if an improvement is found
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()

                # Update the global best position and score if this particle's position is an improvement
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
                    self.global_best_score_of_each_term = deepcopy(score_of_each_term)

                # Record the current iteration's losses and parameters
                self.recorder.record_loss(iteration=iter,
                                          opt_parameters=self.global_best_position,
                                          total_loss=self.global_best_score,
                                          each_term_loss=self.global_best_score_of_each_term,
                                          write_iteration_loss_filepath='./loss.log',
                                          note=f'{particle.idx}th-bead')

            # Check for convergence by comparing the change in the global best score to the convergence threshol
            if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            # Stop optimization if there has been no significant improvement for a defined numbe
            if no_improvement_iters >= self.max_no_improvement_iters:
                break

            previous_best_score = self.global_best_score  # Update the previous best score for the next iteration

            # Adaptively adjust the bounds of the search space to encourage exploration
            self.adaptive_bounds(iter)

            # Update particles' velocities and positions for the next iteration
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()  # Random coefficients for stochastic components
                new_velocity = (self.options['inertia_weight'] * particle.velocity +
                                self.options['personal_accel'] * r1 * (particle.best_position - particle.position) +
                                self.options['global_accel'] * r2 * (self.global_best_position - particle.position))
                particle.velocity = new_velocity
                particle.position += new_velocity

                # Apply bounds, ensure the new position respects the bounds of the search space
                for i in range(len(self.bounds)):
                    particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])

        print('\nPSO optimization done!')
        return self.global_best_position, self.global_best_score, self.recorder


    def optimize_mpi(self, max_processes, opt_folder=None):
        """
        Executes the PSO algorithm in parallel using multiple processes. This method aims to optimize the objective function
        by adjusting particles' positions and velocities across the search space, leveraging multiprocessing to expedite
        the computation of the objective function for each particle.

        Parameters:
        -----------
        max_processes : int
            The maximum number of worker processes to use for parallel computation. This should not exceed the number of
            available CPU cores for optimal performance.
        opt_folder : str, optional
            The directory where optimization artifacts (e.g., logs, intermediate results) will be stored. If not specified,
            these artifacts are not saved.

        Returns:
        --------
        tuple
            Returns a tuple containing the global best position found, the corresponding objective function value at this
            position, and an instance of the LossRecorder with recorded losses and optimization parameters.
        """

        print('\nPSO optimization_mpi starting...')
        no_improvement_iters = 0  # Tracks the number of iterations without improvement in the global best score
        previous_best_score = float('inf')  # Initialize the best score for comparison in the first iteration

        for iter in range(self.max_iter):
            self.iter = iter  # Current iteration number
            for particle in self.particles:
                particle.iter = iter

            # Create a directory for the current iteration if an output folder is specified
            if opt_folder is not None:
                mkdir(os.path.join(opt_folder, f'iter_{iter}'))

            self.update_inertia_weight(iter)  # Dynamically adjust the inertia weight for balalancing exploration and exploitation

            # Use ProcessPoolExecutor to parallelize the evaluation of the objective function across particles
            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = {executor.submit(self.objective_function, particle_item): particle_item for particle_item in self.particles}

                # Process completed futures as they finish
                for idx, future in enumerate(as_completed(futures)):
                    particle = futures[future]
                    score, score_of_each_term = future.result()

                    # Update particle's personal best
                    if score < particle.best_score:
                        particle.best_score = score
                        particle.best_position = particle.position.copy()

                    # Update the global best if improvement is found
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = particle.position.copy()
                        self.global_best_score_of_each_term = deepcopy(score_of_each_term)

                    # Record the loss for the current iteration
                    self.recorder.record_loss(iteration=iter,
                                              opt_parameters=self.global_best_position,
                                              total_loss=self.global_best_score,
                                              each_term_loss=self.global_best_score_of_each_term,
                                              write_iteration_loss_filepath='./loss.log',
                                              note=f'{particle.idx}th-bead')

            # Check for convergence based on the improvement threshold
            if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            if no_improvement_iters >= self.max_no_improvement_iters:
                break    # Stop the optimization if no significant improvement is observed

            previous_best_score = self.global_best_score  # Update the best score for the next iteration

            # Adjust search space bounds if necessary
            self.adaptive_bounds(iter)

            # Update velocities and positions for all particles
            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                new_velocity = (self.options['inertia_weight'] * particle.velocity +
                                self.options['personal_accel'] * r1 * (particle.best_position - particle.position) +
                                self.options['global_accel'] * r2 * (self.global_best_position - particle.position))
                particle.velocity = new_velocity
                particle.position += new_velocity

                # Apply bounds, ensure new positions are within bounds
                for i in range(len(self.bounds)):
                    particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])

        print('\nPSO optimization parallel done!')
        return self.global_best_position, self.global_best_score, self.recorder


class BayesianOptimizer:
    """
    Implements Bayesian Optimization using Gaussian Processes (GP) to find the minimum of an objective function.
    Bayesian Optimization is particularly useful for optimizing expensive functions where each function evaluation
    incurs a high cost, such as hyperparameter tuning for machine learning models.

    Attributes:
    -----------
    objective_function : callable
        The objective function to be minimized. It should take a single argument (the parameter vector) and return
        a scalar value representing the function's value at that point.
    bounds : list of tuples
        A list of tuples specifying the bounds for each dimension of the input parameters. Each tuple (low, high)
        represents the inclusive lower and upper bounds for a parameter.
    n_initial_points : int
        The number of initial points to sample within the bounds using random sampling before starting the GP-based optimization.
    n_iter : int
        The number of iterations for which to run the optimization process after initializing.
    gp : GaussianProcessRegressor
        The Gaussian Process (GP) model used to estimate the objective function and its uncertainty.
    X_samples : list
        A list of sampled input parameters.
    Y_samples : list
        A list of objective function values corresponding to each entry in `X_samples`.

    Methods:
    --------
    sample_initial_points()
        Samples initial points within the bounds using random sampling.
    expected_improvement(x)
        Computes the Expected Improvement (EI) at a given point, guiding the selection of the next point to evaluate.
    optimize()
        Executes the Bayesian Optimization process, iteratively selecting and evaluating points based on the GP model.
    optimize_mpi()
        initial point computation are parallelized.
    """

    def __init__(self, objective_function, bounds, opt_folder, n_initial_points=5, max_iter=25, max_no_improvement_iters=8, converged_threshold=1e-6):
        """
        Initializes the BayesianOptimizer with the objective function, bounds, and optimization parameters.

        Parameters:
        -----------
        objective_function : callable
            The function to be minimized.
        bounds : list of tuples
            The bounds for each dimension of the input parameters.
        n_initial_points : int
            The number of initial random samples to take.
        n_iter : int
            The number of iterations to perform after the initial sampling.
        """

        self.objective_function = objective_function
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.n_iter = max_iter
        self.opt_folder = opt_folder
        mkdir(self.opt_folder)
        self.recorder = LossRecorder()

        self.max_no_improvement_iters = max_no_improvement_iters
        self.converged_threshold = converged_threshold
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.global_best_score_of_each_term = None

        # Initialize the GP model with a specified kernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel)

        self.X_samples = []  # List to store input parameters
        self.Y_samples = []  # List to store function values

    def sample_initial_points(self):
        """
        Randomly samples initial points within the defined bounds to bootstrap the optimization process.
        """

        print('Starting initial sampling... \n')

        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        mkdir(initial_points_folder)
        for idx in range(self.n_initial_points):
            np.random.seed()
            idx_folder = os.path.join(initial_points_folder, f'{idx}')
            mkdir(idx_folder)
            # Randomly sample a point within the bounds
            x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            # Evaluate the objective function at this point
            y, y_each_term = self.objective_function(x, idx_folder)
            # Store the sample and its objective function value
            self.X_samples.append(x)
            self.Y_samples.append(y)

    def sample_single_point(self, idx):
        np.random.seed()
        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        idx_folder = os.path.join(initial_points_folder, f'{idx}')
        mkdir(idx_folder)
        # Randomly sample a point within the bounds
        x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        # Evaluate the objective function at this point
        y, y_each_term = self.objective_function(x, idx_folder)
        # Store the sample and its objective function value
        # self.X_samples.append(x)
        # self.Y_samples.append(y)
        return x, y

    def sample_initial_points_mpi(self, max_processes=1):
        """
        Randomly samples initial points within the defined bounds to bootstrap the optimization process.
        """

        print('Starting initial sampling... \n')

        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        mkdir(initial_points_folder)

        # Use ProcessPoolExecutor to parallelize sampling
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = [executor.submit(self.sample_single_point, idx) for idx in range(self.n_initial_points)]
            # Wait for all tasks to complete
            for future in as_completed(futures):
                # Check for any exceptions raised during execution
                if future.exception() is not None:
                    print(f"Exception occurred: {future.exception()}")
                else:
                    x, y = future.result()
                    self.X_samples.append(x)
                    self.Y_samples.append(y)

        print("All parallel tasks have completed.")

    def expected_improvement(self, x):
        """
        Calculates the Expected Improvement (EI) at a given point 'x', which is a key acquisition function in Bayesian Optimization. EI measures the expected amount of improvement over the current best observed value and is used to select the next point to evaluate.

        Parameters:
        -----------
        x : numpy.ndarray
            The point at which to calculate the expected improvement. This should be a 2D array with shape (1, n_features).

        Returns:
        --------
        float
            The expected improvement at point 'x'. A higher value indicates a potentially better point to evaluate next.
        """

        # Reshape 'x' to 2D array for compatibility with GP predict
        x = np.array(x).reshape(-1, len(self.bounds))
        # Predict the mean and standard deviation of the objective function at 'x' using the GP model
        mu, sigma = self.gp.predict(x, return_std=True)
        # Calculate the improvement over the current best observed value
        mu_sample_opt = np.min(self.Y_samples)  # The best objective value observed so fa
        with np.errstate(divide='warn'):
            imp = mu_sample_opt - mu  # Improvement over the current best
            Z = imp / sigma  # Standardize improvement by predicted standard deviation
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)  # Expected improvement formula
            return -ei  # Negate EI because we minimize this acquisition function

    def optimize(self, initial_sample_mpi=False, initial_sample_max_processes=1):
        """
        Executes the Bayesian Optimization process, iteratively selecting the next point to evaluate based on the Expected Improvement acquisition function, and updating the Gaussian Process model with new observations.

        Returns:
        --------
        tuple
            The best input parameters found (as a numpy.ndarray) and their corresponding objective function value (float).
        """

        print('Starting Bayesian optimization...\n')

        # Randomly sample initial points and update the GP model with these observations
        if initial_sample_mpi:
            self.sample_initial_points_mpi(initial_sample_max_processes)
        else:
            self.sample_initial_points()

        no_improvement_iters = 0

        iters_folder = os.path.join(self.opt_folder, "iters")
        mkdir(iters_folder)

        for iter_id in range(self.n_iter):
            iter_folder = os.path.join(iters_folder, f"{iter_id}")
            mkdir(iter_folder)
            print(f'\nIteration {iter_id}...\n')
            # Fit the GP model to the observed data
            self.gp.fit(self.X_samples, self.Y_samples)
            # Find the next point to evaluate by maximizing the Expected Improvement
            x_next = minimize(lambda x: -self.expected_improvement(x),
                              np.random.uniform(*zip(*self.bounds)),
                              bounds=self.bounds,
                              method='L-BFGS-B').x
            # Evaluate the objective function at the chosen point
            score, score_of_each_term = self.objective_function(x_next, iter_folder)
            # Update the lists of samples and objective function values
            self.X_samples.append(x_next)
            self.Y_samples.append(score)

            previous_best_score = self.global_best_score
            # Identify the best observed point and its objective function value

            if self.global_best_score > score:
                self.global_best_score = score
                self.global_best_position = x_next.copy()
                self.global_best_score_of_each_term = deepcopy(score_of_each_term)

            # Optionally, record the iteration's results with LossRecorder or similar mechanism here
            self.recorder.record_loss(iteration=iter_id,
                                      opt_parameters=self.global_best_position,
                                      total_loss=self.global_best_score,
                                      each_term_loss=self.global_best_score_of_each_term,
                                      write_iteration_loss_filepath='./loss.log')

                # Check for convergence by comparing the change in the global best score to the convergence threshol
            if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            # Stop optimization if there has been no significant improvement for a defined numbe
            if no_improvement_iters >= self.max_no_improvement_iters:
                break

        idx_best = np.argmin(self.Y_samples)
        return self.X_samples[idx_best], self.Y_samples[idx_best], self.recorder


class Individual:
    def __init__(self, individual_array):
        self.iter = 0
        self.idx = 0
        self.position = individual_array


class GeneticOptimizer:
    """
    Implements a Genetic Algorithm (GA) for optimization. GA mimics the process of natural selection by creating a population of solutions, applying operators such as selection, crossover, and mutation, and evolving the population over generations.

    Attributes:
    -----------
    objective_function : callable
        The objective function to be minimized.
    bounds : list of tuples
        The bounds for each dimension of the search space, where each tuple contains the lower and upper limits.
    method : str, optional
        Specifies the method for ensuring diversity or handling specific optimization strategies. For example, 'Elitism' ensures the best individuals are carried to the next generation.
    population_size : int
        The number of individuals in the population.
    mutation_rate : float
        The probability of an individual gene mutating.
    crossover_rate : float
        The probability of crossover between two individuals.
    max_generations : int
        The maximum number of generations to evolve the population.
    recorder : LossRecorder
        Records and retrieves various metrics throughout the optimization process.

    Methods:
    --------
    initialize_population()
        Initializes the population with random individuals within the specified bounds.
    calculate_fitness(population)
        Evaluates and returns the fitness (objective function value) of each individual in the population.
    select(population, fitness_scores)
        Selects individuals from the current population to breed a new generation, based on their fitness scores.
    crossover(parent1, parent2)
        Combines two individuals (parents) to produce offspring for the next generation.
    mutate(individual)
        Applies random mutations to an individual's genes.
    optimize()
        Executes the genetic algorithm, returning the best solution found.
    optimize_mpi()

    """

    def __init__(self, objective_function, bounds, converged_threshold=1e-6, max_no_improvement_iters=8,
                 method=None, population_size=50, mutation_rate=0.1, crossover_rate=0.8, max_generations=100):
        """
        Initializes the GeneticOptimizer with the specified objective function, search space bounds, and GA parameters.

        Parameters:
        -----------
        objective_function : callable
            The function to minimize.
        bounds : list of tuples
            Search space bounds for each parameter.
        method : str, optional
            The method used for selection or diversity maintenance.
        population_size : int
            Number of individuals in the population.
        mutation_rate : float
            Probability of mutation for each gene.
        crossover_rate : float
            Probability of crossover between two parents.
        max_generations : int
            Maximum number of generations to evolve.
        """

        self.objective_function = objective_function
        self.converged_threshold = converged_threshold
        self.max_no_improvement_iters = max_no_improvement_iters
        self.bounds = bounds
        self.method = method
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.recorder = LossRecorder()

        # Placeholder attributes for global best position and score
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.global_best_score_of_each_term = None

    def initialize_population(self):
        """
        Creates an initial population of random solutions within the defined search space bounds.

        Returns:
        --------
        list
            A list of individuals, where each individual is represented as a list of parameter values.
        """

        population = []
        for _ in range(self.population_size):
            np.random.seed()
            individual = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            population.append(individual)
        return population

    def calculate_fitness(self, population):
        """
        Calculates the fitness of each individual in the population by evaluating the objective function.

        Parameters:
        -----------
        population : list
            The current population of individuals.

        Returns:
        --------
        list
            A list of fitness scores corresponding to each individual in the population.
        """
        fitness_scores = [self.objective_function(individual) for individual in population]
        return fitness_scores

    def select(self, population, fitness_scores):
        """
        Selects individuals from the current population to be parents for the next generation, based on their fitness scores.

        Parameters:
        -----------
        population : list
            The current population of individuals.
        fitness_scores : list
            The fitness scores for each individual in the population.

        Returns:
        --------
        list
            A list of selected individuals to be parents for the next generation.
        """

        fitness_total = sum(fitness_scores)
        probability = [f / fitness_total for f in fitness_scores]
        chosen_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=probability)
        return [population[i] for i in chosen_indices]

    def crossover(self, parent1, parent2, strategy='single_point'):
        """
        Performs crossover between two parents to produce offspring for the next generation.

        Parameters:
        -----------
        parent1 : list
            The first parent individual.
        parent2 : list
            The second parent individual.

        Returns:
        --------
        tuple
            A tuple containing two offspring individuals resulting from the crossover.
        """

        # Check if crossover will occur based on the crossover rate
        if np.random.rand() >= self.crossover_rate:
            return parent1, parent2

        # Determine the crossover strategy
        if strategy == 'single_point':
            # Single-point crossover: A single crossover point is selected, beyond which genes are swapped between parents
            crossover_point = np.random.randint(1, len(self.bounds))
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        elif strategy == 'two_point':
            # Two-point crossover: Two points are selected, and genes between these points are swapped between the parents
            point1, point2 = sorted(np.random.randint(1, len(self.bounds), size=2))
            child1 = np.concatenate(parent1[:point1], parent2[point1:point2], parent1[point2:])
            child2 = np.concatenate(parent2[:point1], parent1[point1:point2], parent2[point2:])
        elif strategy == 'uniform':
            # Uniform crossover: Each gene is independently considered for swapping with a 50% chance
            child1, child2 = parent1.copy(), parent2.copy()
            for i in range(len(self.bounds)):
                if np.random.rand() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
        else:
            raise ValueError("Unknown crossover strategy: {}".format(strategy))

        # Return the offspring produced by the crossover
        return child1, child2

    def mutate(self, individual):
        """
        Applies mutation to an individual's genes with a specified probability.

        Parameters:
        -----------
        individual : list
            The individual to mutate.

        Returns:
        --------
        list
            The mutated individual.
        """

        for i in range(len(self.bounds)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        return individual

    def optimize(self):
        """
        Executes the genetic algorithm, iterating over generations to find the best solution.

        Returns:
        --------
        tuple
            The best solution found and its objective function value.
        """

        print('Starting genetic optimization...\n')

        population = self.initialize_population()
        no_improvement_iters = 0
        previous_best_score = self.global_best_score

        for generation in range(self.max_generations):

            fitness_scores, fitness_scores_of_each_term = [], []
            Population_list = []
            for individual_id, individual_array in enumerate(population):
                individual_item = Individual(individual_array=individual_array)
                individual_item.idx = individual_id
                individual_item.iter = generation
                Population_list.append(individual_item)
                fitness_scores.append(0)
                fitness_scores_of_each_term.append(0)

            for individual_item in Population_list:
                score, score_of_each_term = self.objective_function(individual_item)
                fitness_scores[individual_item.idx] = score
                fitness_scores_of_each_term[individual_item.idx] = deepcopy(score_of_each_term)

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_score_of_each_term = deepcopy(score_of_each_term)
                    self.global_best_position = individual_item.position.copy()

                # Record the loss for the current iteration
                self.recorder.record_loss(iteration=generation,
                                          opt_parameters=self.global_best_position,
                                          total_loss=self.global_best_score,
                                          each_term_loss=self.global_best_score_of_each_term,
                                          write_iteration_loss_filepath='./loss.log',
                                          note=f'{individual_item.idx}th-individual')

            # Check for convergence based on the improvement threshold
            if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            if no_improvement_iters >= self.max_no_improvement_iters:
                break  # Stop the optimization if no significant improvement is observed

            selected = self.select(population, fitness_scores)
            children = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[min(i + 1, self.population_size - 1)]  # Designed to handle cases where the population size is odd.
                child1, child2 = self.crossover(parent1, parent2)
                children.append(self.mutate(child1))
                if len(children) < self.population_size:  # Designed to handle cases where the population size is odd.
                    children.append(self.mutate(child2))

            # Optional: Implement elitism or other methods to ensure the best individuals a
            if self.method == 'Elitism':
                children[0] = self.global_best_position

            population = children

        print('Genetic optimization done!')

        return self.global_best_position, self.global_best_score, self.recorder

    def optimize_mpi(self, max_processes=1, opt_folder=None):
        """
        Executes the genetic algorithm, iterating over generations to find the best solution.

        Returns:
        --------
        tuple
            The best solution found and its objective function value.
        """

        print('Starting genetic mpi optimization...\n')

        population = self.initialize_population()
        no_improvement_iters = 0
        previous_best_score = self.global_best_score

        for generation in range(self.max_generations):
            fitness_scores, fitness_scores_of_each_term = [], []
            Population_list = []
            for individual_id, individual_array in enumerate(population):
                individual_item = Individual(individual_array=individual_array)
                individual_item.idx = individual_id
                individual_item.iter = generation
                Population_list.append(individual_item)
                fitness_scores.append(0)
                fitness_scores_of_each_term.append(0)

            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                futures = {executor.submit(self.objective_function, individual_item): individual_item for individual_item in Population_list}

                # Process completed futures as they finish
                for idx, future in enumerate(as_completed(futures)):
                    individual_item = futures[future]
                    score, score_of_each_term = future.result()
                    fitness_scores[individual_item.idx] = score
                    fitness_scores_of_each_term[individual_item.idx] = deepcopy(score_of_each_term)

                    # Update global best
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_score_of_each_term = deepcopy(score_of_each_term)
                        self.global_best_position = individual_item.position.copy()

                    # Record the loss for the current iteration
                    self.recorder.record_loss(iteration=generation,
                                              opt_parameters=self.global_best_position,
                                              total_loss=self.global_best_score,
                                              each_term_loss=self.global_best_score_of_each_term,
                                              write_iteration_loss_filepath='./loss.log',
                                              note=f'{individual_item.idx}th-individual')

                # Check for convergence based on the improvement threshold
                if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                    no_improvement_iters += 1
                else:
                    no_improvement_iters = 0

                if no_improvement_iters >= self.max_no_improvement_iters:
                    break  # Stop the optimization if no significant improvement is observed


            selected = self.select(population, fitness_scores)
            children = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected[i], selected[min(i + 1, self.population_size - 1)]  # Designed to handle cases where the population size is odd.
                child1, child2 = self.crossover(parent1, parent2)
                children.append(self.mutate(child1))
                if len(children) < self.population_size:  # Designed to handle cases where the population size is odd.
                    children.append(self.mutate(child2))

            # Optional: Implement elitism or other methods to ensure the best individuals a
            if self.method == 'Elitism':
                children[0] = self.global_best_position

            population = children

        print('Genetic optimization parallel done!')
        return self.global_best_position, self.global_best_score, self.recorder


class SimplexOptimizer:
    """
    Implements the Nelder-Mead simplex algorithm for optimization. This method is a heuristic search algorithm that can be used to find the minimum or maximum of an objective function in a multidimensional space. It is particularly useful for optimization problems where the objective function is non-differentiable, discontinuous, or noisy.

    Attributes:
    -----------
    objective_function : callable
        The objective function to be minimized. It must take a single argument in the form of a numpy array and return a scalar value.
    bounds : list of tuples
        A list where each tuple represents the lower and upper bounds for each dimension of the search space.
    max_iter : int
        The maximum number of iterations to perform.
    tol : float
        The tolerance for convergence. The algorithm stops if the reduction in the function value is below this threshold over one iteration.
    iteration_results : list
        A list to store the results of each iteration for analysis. Each element is a tuple containing the parameters and the corresponding objective function value.

    Methods:
    --------
    optimize(initial_guess=None)
        Executes the simplex algorithm starting from an initial guess. If no initial guess is provided, it starts from a randomly generated point within the bounds.
    """

    def __init__(self, objective_function, bounds,  opt_folder, n_initial_points=5, max_iter=25, max_no_improvement_iters=8, converged_threshold=1e-6):
        """
        Initializes the SimplexOptimizer with the objective function, search space bounds, and optimization parameters.

        Parameters:
        -----------
        objective_function : callable
            The function to minimize.
        bounds : list of tuples
            The bounds for each dimension of the search space.
        max_iter : int
            The maximum number of iterations for the optimization process.
        tol : float
            The convergence tolerance; the optimization stops if the simplex algorithm achieves a smaller reduction in the function value than this threshold.
        """

        self.objective_function = objective_function
        self.bounds = bounds

        self.n_initial_points = n_initial_points
        self.n_iter = max_iter
        self.opt_folder = opt_folder
        mkdir(self.opt_folder)
        self.recorder = LossRecorder()

        self.max_no_improvement_iters = max_no_improvement_iters
        self.converged_threshold = converged_threshold
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.global_best_score_of_each_term = None

        self.X_samples = []  # List to store input parameters
        self.Y_samples = []  # List to store function value
        self.Y_each_term_samples = []

    def sample_initial_points(self):
        """
        Randomly samples initial points within the defined bounds to bootstrap the optimization process.
        """

        print('Starting initial sampling... \n')

        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        mkdir(initial_points_folder)
        for idx in range(self.n_initial_points):
            np.random.seed()
            idx_folder = os.path.join(initial_points_folder, f'{idx}')
            mkdir(idx_folder)
            # Randomly sample a point within the bounds
            x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            # Evaluate the objective function at this point
            y, y_each_term = self.objective_function(x, idx_folder)
            # Store the sample and its objective function value
            self.X_samples.append(x)
            self.Y_samples.append(y)
            self.Y_each_term_samples.append(y_each_term)

    def sample_single_point(self, idx):
        np.random.seed()
        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        idx_folder = os.path.join(initial_points_folder, f'{idx}')
        mkdir(idx_folder)
        # Randomly sample a point within the bounds
        x = np.array([np.random.uniform(low, high) for low, high in self.bounds])
        # Evaluate the objective function at this point
        y, y_each_term = self.objective_function(x, idx_folder)
        # Store the sample and its objective function value
        return x, y, y_each_term

    def sample_initial_points_mpi(self, max_processes=1):
        """
        Randomly samples initial points within the defined bounds to bootstrap the optimization process.
        """

        print('Starting initial sampling... \n')

        initial_points_folder = os.path.join(self.opt_folder, "initial_points")
        mkdir(initial_points_folder)

        # Use ProcessPoolExecutor to parallelize sampling
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = [executor.submit(self.sample_single_point, idx) for idx in range(self.n_initial_points)]
            # Wait for all tasks to complete
            for future in as_completed(futures):
                # Check for any exceptions raised during execution
                if future.exception() is not None:
                    print(f"Exception occurred: {future.exception()}")
                else:
                    x, y, y_each_term = future.result()
                    self.X_samples.append(x)
                    self.Y_samples.append(y)
                    self.Y_each_term_samples.append(y_each_term)

        print("All parallel tasks have completed.")

    def points_mpi(self, folder, max_processes=1):
        # Use ProcessPoolExecutor to parallelize sampling
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # futures = [executor.submit(self.objective_function, x) for x in self.X_samples]
            futures = [executor.submit(self.objective_function, x, os.path.join(folder, str(x_id))) for x_id, x in
             enumerate(self.X_samples)]
            # Wait for all tasks to complete
            for future in as_completed(futures):
                # Check for any exceptions raised during execution
                if future.exception() is not None:
                    print(f"Exception occurred: {future.exception()}")
                else:
                    scores, scores_each_term = future.result()
                    self.Y_samples.append(scores)
                    self.Y_each_term_samples.append(scores_each_term)

    def optimize(self, initial_sample_mpi=False, initial_sample_max_processes=1, shrink_mpi=False, shrink_max_process=1):

        print('Starting simplex optimization...\n')
        # Randomly sample initial points and update the GP model with these observations
        if initial_sample_mpi:
            self.sample_initial_points_mpi(initial_sample_max_processes)
        else:
            self.sample_initial_points()

        no_improvement_iters = 0
        self.X_samples = np.array(self.X_samples)
        self.Y_samples = np.array(self.Y_samples)
        self.Y_each_term_samples = np.array(self.Y_each_term_samples)

        for iter_id in range(self.n_iter):
            iter_folder = os.path.join(self.opt_folder, f"{iter_id}")
            mkdir(iter_folder)
            print(f'\nIteration {iter_id}...\n')

            sorted_idx = np.argsort(self.Y_samples)
            self.Y_samples = self.Y_samples[sorted_idx].copy()
            self.Y_each_term_samples = self.Y_each_term_samples[sorted_idx].copy()
            self.X_samples = self.X_samples[sorted_idx].copy()
            score = self.Y_samples[0]
            score_of_each_term = self.Y_each_term_samples[0]
            best_point = self.X_samples[0]
            worst_point = self.X_samples[-1]
            worst_score = self.Y_samples[-1]
            worst_score_each_term = self.Y_each_term_samples[-1]

            previous_best_score = self.global_best_score
            # Identify the best observed point and its objective function value

            if self.global_best_score > score:
                self.global_best_score = score
                self.global_best_position = best_point.copy()
                self.global_best_score_of_each_term = deepcopy(score_of_each_term)

            # Optionally, record the iteration's results with LossRecorder or similar mechanism here
            self.recorder.record_loss(iteration=iter_id,
                                      opt_parameters=self.global_best_position,
                                      total_loss=self.global_best_score,
                                      each_term_loss=self.global_best_score_of_each_term,
                                      write_iteration_loss_filepath='./loss.log')

            # Check for convergence by comparing the change in the global best score to the convergence threshol
            if abs(self.global_best_score - previous_best_score) < self.converged_threshold:
                no_improvement_iters += 1
            else:
                no_improvement_iters = 0

            # Stop optimization if there has been no significant improvement for a defined numbe
            if no_improvement_iters >= self.max_no_improvement_iters:
                break

            centroid = np.mean(self.X_samples[:-1], axis=0)

            # Reflect
            print('Starting reflection...')
            reflection = centroid + (centroid - worst_point)
            reflect_folder = os.path.join(iter_folder, "reflect")
            mkdir(reflect_folder)
            reflection_value, reflection_value_each_term = self.objective_function(reflection, reflect_folder)
            if self.Y_samples[0] <= reflection_value < self.Y_samples[-2]:
                worst_point = reflection.copy()
                worst_score = reflection_value
                worst_score_each_term = deepcopy(reflection_value_each_term)
            elif reflection_value < self.Y_samples[0]:
                # Expand
                print('Starting expansion...')
                expansion = centroid + 2 * (reflection - centroid)
                expansion_folder = os.path.join(iter_folder, "expand")
                mkdir(expansion_folder)
                expansion_value, expansion_value_each_term = self.objective_function(expansion, expansion_folder)
                if expansion_value < reflection_value:
                    worst_point = expansion.copy()
                    worst_score = expansion_value
                    worst_score_each_term = deepcopy(expansion_value_each_term)
                else:
                    worst_point = reflection.copy()
                    worst_score = reflection_value
                    worst_score_each_term = deepcopy(reflection_value_each_term)
            else:
                # Contract
                print('Starting expansion...')
                contraction = centroid + 0.5 * (worst_point - centroid)
                contraction_folder = os.path.join(iter_folder, "contract")
                mkdir(contraction_folder)
                contraction_value, contraction_value_each_term = self.objective_function(contraction, contraction_folder)
                if contraction_value < self.Y_samples[-1]:
                    worst_point = contraction
                    worst_score =contraction_value
                    worst_score_each_term = deepcopy(contraction_value_each_term)
                else:
                    # Shrink
                    print('Starting shrink...')
                    shrink_folder = os.path.join(iter_folder, "shrink")
                    mkdir(shrink_folder)

                    self.Y_samples = []
                    self.Y_each_term_samples = []
                    for i in range(1, len(self.X_samples)):
                        self.X_samples[i] = 0.5 * (self.X_samples[i] + best_point)
                    if shrink_mpi:
                        self.points_mpi(max_processes=shrink_max_process, folder=shrink_folder)
                    else:
                        for x_id, x in enumerate(self.X_samples):
                            i_folder = os.path.join(shrink_folder, str(x_id))
                            score_sh, score_of_each_term_sh = self.objective_function(x, i_folder)
                            self.Y_samples.append(score_sh)
                            self.Y_each_term_samples.append(score_of_each_term_sh)

                    self.Y_samples = np.array(self.Y_samples)
                    self.Y_each_term_samples = np.array(self.Y_each_term_samples)
                    sorted_idx = np.argsort(self.Y_samples)
                    self.Y_samples = self.Y_samples[sorted_idx].copy()
                    self.Y_each_term_samples = self.Y_each_term_samples[sorted_idx].copy()
                    self.X_samples = self.X_samples[sorted_idx].copy()
                    continue

            self.X_samples[-1] = worst_point
            self.Y_samples[-1] = worst_score
            self.Y_each_term_samples[-1] = deepcopy(worst_score_each_term)

        return self.X_samples[0], self.Y_samples[0], self.recorder


class GradientDescentOptimizer:
    """
    Implements Gradient Descent (GD),Gradient Descent with Momentum (GDM), and Adaptive Moment Estimation (Adam) for optimization.
    This class is useful for optimization problems where the objective function is differentiable.

    Attributes:
    -----------
    objective_function : callable
        The objective function to be minimized. It must take a single argument in the form of a numpy array and return a scalar value.
    gradient_function : callable
        The gradient of the objective function. It must take a single argument in the form of a numpy array and return a numpy array of gradients.
    bounds : list of tuples
        A list where each tuple represents the lower and upper bounds for each dimension of the search space.
    max_iter : int
        The maximum number of iterations to perform.
    tol : float
        The tolerance for convergence. The algorithm stops if the reduction in the function value is below this threshold over one iteration.
    learning_rate : float
        The learning rate for the optimization.
    method : str
        The optimization method to use ('gd', 'gdm', or 'adam').

    Methods:
    --------
    optimize(initial_guess)
        Executes the optimization algorithm starting from an initial guess.
    """

    def __init__(self, objective_function, gradient_function, bounds, initial_guess=None, max_iter=1000, tol=1e-6, learning_rate=0.01, method='gd'):
        """
        Initializes the GradientDescentOptimizer with the objective function, gradient function, search space bounds, and optimization parameters.

        Parameters:
        -----------
        objective_function : callable
            The function to minimize.
        gradient_function : callable
            The gradient of the objective function.
        bounds : list of tuples
            The bounds for each dimension of the search space.
        max_iter : int
            The maximum number of iterations for the optimization process.
        tol : float
            The convergence tolerance.
        learning_rate : float
            The learning rate for the optimization.
        method : str
            The optimization method to use ('gd', 'gdm', or 'adam').
        """

        self.objective_function = objective_function
        self.gradient_function = gradient_function
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.method = method
        self.iteration_results = []

    def optimize(self, initial_guess):
        if self.method == 'gd':
            return self._gradient_descent(initial_guess)
        elif self.method == 'gdm':
            return self._gdm(initial_guess)
        elif self.method == 'adam':
            return self._adam(initial_guess)
        else:
            raise ValueError("Invalid optimization method. Choose 'gd', 'gdm', or 'adam'.")

    def _gradient_descent(self, initial_guess):
        # GD implementation
        pass

    def _gdm(self, initial_guess):
        # SGDM implementation
        pass

    def _adam(self, initial_guess):
        # Adam implementation
        pass

def rastrigin_function(x):
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def x2_function(x):
    return sum([(xi - 2)**2 for xi in x])

def main():

    dimensions = 2
    bounds = [(-1, 1) for _ in range(dimensions)]
    num_particles = 5
    max_iter = 50
    # bounds =[(0, 5)]

    optimizer = SimplexOptimizer(rastrigin_function, bounds)
    best_x, best_y, iteration_results = optimizer.optimize()
    print("Best X:", best_x)
    print("Best Y:", best_y)

    ga = GeneticOptimizer(rastrigin_function, bounds, population_size=1000, max_generations=100)
    best_individual, best_fitness = ga.optimize()
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_fitness)
    optimizer = BayesianOptimizer(x2_function, bounds, n_initial_points=50, n_iter=100)
    best_x, best_y = optimizer.optimize()
    print("Best X:", best_x)
    print("Best Y:", best_y)

    system_top = {'molecules': [
        {'mol_name': '12oh', 'model': 'MARTINI2', 'types': ['P1'], 'id': [0], 'charge': [0.0], 'mass': [57.1146],
         'num_mols': 100, 'lj_parameters': {('P1', 'P1'): [4.5, 0.47]}, 'fg_groups': [
            [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20, 4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28, 8, 9, 10, 11,
             29, 30, 31, 32, 33, 34, 35]], 'bond_parameters': None, 'angle_parameters': None},
        {'mol_name': '16oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'C1', 'P1'], 'id': [0, 1, 2, 3],
         'charge': [1.0, 0.0, 2.0, -1.0], 'mass': [57.1146, 56.1067, 56.1067, 59.0869], 'num_mols': 100,
         'lj_parameters': {('C1', 'P1'): [2.7, 0.47], ('C1', 'C1'): [3.5, 0.47], ('P1', 'P1'): [4.5, 0.47]},
         'fg_groups': [[0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24], [4, 5, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32],
                       [8, 9, 10, 11, 33, 34, 35, 36, 37, 38, 39, 40], [12, 13, 14, 15, 41, 42, 43, 44, 45, 46, 47]],
         'bond_parameters': {(0, 1): [0.4662, 1250.0], (1, 2): [0.4697, 1250.0], (2, 3): [0.4634, 1250.0]},
         'angle_parameters': {(0, 1, 2): [146.7049, 25.0], (1, 2, 3): [146.661, 25.0]}}],
                  'lj_cross_terms': {('C1', 'P1'): [2.7, 0.47], ('C1', 'C1'): [3.5, 0.47], ('P1', 'P1'): [4.5, 0.47]},
                  'cgmodel': 'MARTINI2'}

    opt_term_parse = {
        'molecules': [{'mol_name': '12oh', 'charge': [0.1], 'mass': [0]},
                      {'mol_name': '16oh', 'charge': [0.1, 0, 0.2, 0.3],
                       'angle_parameters': {(0, 1, 2): [0, 0], (1, 2, 3): [0, 0]}}],
        'lj_cross_terms': {('C1', 'P1'): [0, 0], ('C1', 'C1'): [1, 1], ('P1', 'P1'): [0, 0]}}

    opt_para_term = OptParametersProcess(system_topology=system_top, opt_term_parse=opt_term_parse)

    opt_boundary = opt_para_term.pack_parameters()
    opt_array = [(i[0]+i[1])/2 for i in opt_boundary]
    new_top = opt_para_term.unpack_updated_parameters_to_top(opt_array)


    pso = ParticleSwarmOptimizer(objective_function=rastrigin_function, update_boundary_frequency=10,
                                 bounds=bounds, num_particles=num_particles, max_iter=max_iter)



    best_position, best_score = pso.optimize_mpi(max_processes=4)
    #
    # print(f"Best Position: {best_position}")
    # print(f"Best Score: {best_score}")


    system = {'molecules': [{'mol_name': 'A', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'],
                            'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
                            'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
                            'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
                            'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 2},
                           {'mol_name': 'B', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'],
                            'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
                            'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
                            'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
                            'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 3}],
             'lj_cross_terms': {('P1', 'C1'): [2.7, 0.47],  ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]}}

    opt_term = OptParametersProcess(system_topology=system, r_epsilon=0.1, r_sigma=0.1, r_k_bond=0.1, r_k_angle=0.1)

if __name__ == '__main__':
    main()



