PEO
===

Polyethylene oxide (PEO), also known as polyethylene glycol (PEG), is a polymer with a wide range of applications, including in biomaterials, pharmaceuticals, and polymer electrolytes. Understanding its behavior at the molecular level is crucial for optimizing its properties for various applications. (Full scripts used in this example can be found in `here </home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/package_log/repository_test/doc/source/example/PEO>`_.)

.. note::

   To fit the microscopic properties from fine-grained simulaiton, it is needed to prepare the fine-grained trajectory and topology file in advance.


Optimization Objective
-----------------------
The primary objective of this optimization process is to fine-tune the force field parameters of the PEO system such that its behavior in molecular dynamics simulations accurately matches experimental data. Key properties targeted for optimization include density, surface tension, CG force and potential of mean force U(r).

Optimization Workflow
----------------------
#. **System Setup**: The molecular topology of the PEO system is generated based on provided molecular structure information. Bond and angle parameters are adjusted to match the expected behavior of PEO molecules.

#. **Objective Function Definition**: An objective function is defined to quantify the deviation between simulation results and experimental data. This function computes various terms, such as force-match loss, U(r) Boltzmann inversion loss, density loss, and surface tension loss.

#. **Optimization Execution**: The optimization process is executed using the Bayesian optimization technique. The optimizer iteratively adjusts the force field parameters to minimize the total loss function.

#. **Result Analysis**: The optimized parameters and corresponding loss values are recorded. Additionally, the running time of the optimization task is logged for evaluation.

Environment Setup and Module Importing
---------------------------------------

The script sets up the environment by adding the path to the GROMACS executable directory to the ``PATH`` environment variable. It imports necessary modules from the AMOFMS package for optimization, force field parameter handling, and utility functions.

.. code-block:: python

    # Importing necessary modules
    import os
    import copy
    import time

    # Adding GROMACS executable directory to PATH
    os.environ['PATH'] += ':/home/xiaoyedi/data/research/tools/gromacs-2023.3/bin'

    # Importing modules from the AMOFMS package
    from AMOFMS.Optimization import OptParametersProcess, ParticleSwarmOptimizer, Particle, BayesianOptimizer
    from AMOFMS.CGForceFieldParameters import generate_system_top
    from AMOFMS.tools.utilies import mkdir
    from AMOFMS.ObjectiveFunction import BottomUpObjectiveFunction
    from AMOFMS.tools.math_tools import square_diff_of_elements_in_2D_list
    from AMOFMS.tools.properties import compute_density, compute_interface_tension

    from AMOFMS.tools.gromacs import find_gmx_executable, unwrap_trajectory, run_complete_simulation
    import subprocess
    import pexpect

    from opt_init_para import molecule_topology_list

Initialization and Configuration
--------------------------------

Paths, parameters, and options for the optimization task are initialized and configured. Initial settings include paths to input files, output folders, simulation parameters, and optimization parameters.

.. code-block:: python

    # Initializing time and printing start time
    time_start = int(time.time())
    timeArray_start = time.localtime(time_start)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray_start)
    print(f'\nStarting Optimization Task at {start_time} \n')

    # Paths and filenames
    fg_topology = 'aa/prod/prod.tpr'
    fg_trajectory = 'aa/prod/prod_whole.trr'
    gmx = find_gmx_executable()

    # Setting up optimization folder and filenames
    opt_folder = './nb_opt'
    opt_folder = os.path.realpath(opt_folder)
    loss_dat = os.path.join(opt_folder, 'loss.dat')
    mdp_folder = './mdp'
    force_match_mdp = os.path.join(mdp_folder, 'force_match.mdp')
    surface_tension_mdp = './surface_tension_mdp'
    initial_gro = None
    index_file = None
    run_em = True
    em_double = False
    run_anneal = False
    run_eq = True  # include anneal and eq
    run_prod = True

System Topology Generation
---------------------------

The system topology is generated based on provided molecular topology information. Bond and angle parameters are adjusted to appropriate values based on the molecular structure.

.. code-block:: python

    # Generating system topology
    system_topology = generate_system_top(mols=molecule_topology_list, num_mols=None)

Objective Function Definition
------------------------------

An objective function (`opt_loss_function`) is defined to compute the loss during optimization. The function calculates various terms of the loss, including force match loss, Ur Boltzmann inversion loss, density loss, and surface tension loss.

.. code-block:: python

    def opt_loss_function(para_array, opt_folder, nt):
        # Unpacking parameters and updating system topology
        new_topology = opt_para.unpack_updated_parameters_to_top(updated_parameters_array=para_array)
        bottom_up_obj.update_system_topology(new_system_top=new_topology)
        bottom_up_obj.update_opt_folder(new_opt_folder=opt_folder)

        tmp_result = os.path.join(opt_folder, 'iter_result.dat')

        # Writing header to result file
        with open(tmp_result, 'w') as f:
            line = f'{"Force_match_loss":<12} {"Ur_loss":<12}  {"Density(kg/m3, exp:1127)":<12}  {"Surface_tension(mN/m, exp:42.6)":<12} \n'
            f.write(line)

        total_loss = 0
        each_term_loss = {}

        # Computing force match loss
        print('\nComputing force match loss...')
        force_match_loss = force_match_loss_ratio * bottom_up_obj.force_match_loss(cg_mdp_file=force_match_mdp, fg_resname_list=resname_cg_from_fg_coord, begin_frame=None, end_frame=None,
                                                                                   skip_frame=force_match_skip_frames)
        each_term_loss.update({'force_match': force_match_loss})
        total_loss += force_match_loss

        # Computing Ur Boltzmann inversion loss
        print('\nComputing Ur Boltzmann inversion loss...')
        bottom_up_obj.run_cg_simulation(initial_gro=initial_gro, fg_resname_list=resname_cg_from_fg_coord,
                                        mdp_folder=mdp_folder, index_file=index_file, em_double_version=em_double,
                                        cg_simulation_folder=opt_folder, em=run_em, anneal=run_anneal,
                                        eq=run_eq, prod=run_prod, nt=nt, gpu_id=None, gpu_acceleration=False)
        cg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(rdf_pairs_list=cg_rdf_pairs_list, tag='cg',
                                                            Temperature=temperature, bin_width=rdf_binwidth, max_distance=rdf_cutoff)
        Ur_loss = Ur_loss_ratio * square_diff_of_elements_in_2D_list(list1=fg_pair_Ur_list, list2=cg_pair_Ur_list)
        each_term_loss.update({'Ur': Ur_loss})
        total_loss += Ur_loss
    
        # Computing density loss
        print('\nComputing density loss...')
        cg_density, _ = compute_density(topology=bottom_up_obj.cg_topology, trajectory=bottom_up_obj.cg_trajectory)
        print(f'density(kg/m3): \nexp: {exp_denisty} cg: {cg_density}')
        density_loss = density_loss_ratio * abs(cg_density - exp_denisty)
        each_term_loss.update({'density': density_loss})
        total_loss += density_loss
    
        # Computing surface tension loss
        print('\nComputing surface tension loss...')
        surface_tension_folder = os.path.join(opt_folder, 'surface_tension')
        mkdir(surface_tension_folder)
        init_gro = os.path.join(surface_tension_folder, 'init.gro')
        unwrap_trajectory(topology=bottom_up_obj.cg_topology, trajectory=bottom_up_obj.final_cg_gro, save_file=init_gro)
        with open(init_gro, 'r') as f:
            box_vector = f.readlines()[-1].split()
            box_x, box_y = box_vector[0], box_vector[1]
        surface_gro = os.path.join(surface_tension_folder, 'surface.gro')
        extend_z_command = f'{gmx} editconf -f {init_gro} -o {surface_gro} -c -box {box_x} {box_y} {surface_box_z/10}'
        subprocess.run(extend_z_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Running complete simulation for surface tension calculation
        run_complete_simulation(mdp_folder=surface_tension_mdp, initial_gro=surface_gro, cg_top_file=bottom_up_obj.cg_top_file,
                                cg_simulation_folder=surface_tension_folder, em=True, em_double_version=em_double, anneal=False, eq=True, prod=True, gpu_acceleration=False,
                                nt=nt, gpu_id=None)
        surface_tension_dat = os.path.join(surface_tension_folder, 'surface_tension_average.dat')
        gen_surface_tension_dat_command = f'{gmx} energy -f  {surface_tension_folder}/prod/prod.edr -o {surface_tension_folder}/surface_tension.xvg'
        child = pexpect.spawn(gen_surface_tension_dat_command)
        # Expecting user input for group selection
        try:
            child.expect("End your selection with an empty line or a zero.")  # Waiting for group selection prompt
            child.sendline(pressure_group)  # Sending the desired group number
            child.expect(pexpect.EOF)  # Waiting for command execution completion
        except pexpect.EOF:
            raise Exception('gmx trjconv process terminated unexpectedly.')
        except pexpect.TIMEOUT:
            raise Exception('gmx trjconv process timed out.')
        
        output = child.before.decode()
        child.expect(pexpect.EOF)
        if child.exitstatus != 0:
            print(output)
            raise Exception('\ngmx energy compute surface tension failed')
        
        # Writing output to file
        with open(surface_tension_dat, 'w') as file:
            file.write(output)
        
        # Computing surface tension from the output
        cg_surface_tension = compute_interface_tension(pressure_dat=surface_tension_dat, num_interface=2)
        print(f'surface_tension(mN/m):\n exp: {exp_surface_tension}  cg: {cg_surface_tension}')
        surface_tension_loss = surface_tension_loss_ratio * abs(cg_surface_tension - exp_surface_tension)
        each_term_loss.update({'surface tension': surface_tension_loss})
        total_loss += surface_tension_loss
        
        # Writing results to file
        with open(tmp_result, 'a+') as f:
            line = f'{force_match_loss:<12}  {Ur_loss:<12}  {cg_density:<12}  {cg_surface_tension:<12}\n'
            f.write(line)
        
        print(f'\nDone!')
        return total_loss, each_term_loss

Optimization Execution
-----------------------

The :ref:`BayesianOptimizer` from the AMOFMS package is utilized for the optimization process. The optimizer iteratively adjusts the optimization parameters to minimize the total loss function.

.. code-block:: python

    # Creating optimizer object
    optimizer = BayesianOptimizer(objective_function=opt_loss_function, bounds=opt_para_boundary, opt_folder=opt_folder,
                                  n_initial_points=initial_points, max_iter=max_iter, max_no_improvement_iters=max_no_improvement_iters)

Result Recording and Output
---------------------------

Optimization results, including optimized parameters and corresponding loss values, are recorded. The running time of the optimization task is also recorded for evaluation.

.. code-block:: python

    # Running optimization and recording results
    best_para, best_score, recorder = optimizer.optimize(initial_sample_mpi=True, initial_sample_max_processes=max_processes)
    recorder.write_losses_to_file(filepath=loss_dat)

Conclusion
----------

This script provides a framework for conducting optimization tasks involving molecular dynamics simulations and force field parameter optimization. It integrates various functionalities from the AMOFMS package and allows for flexible customization based on specific optimization requirements.


