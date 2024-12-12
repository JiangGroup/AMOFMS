POPC
====

1-palmitoyl-2-oleoylphosphatidylcholine (POPC) is a phospholipid commonly used in biomolecular simulations due to its prevalence in biological membranes. Optimizing its parameters is crucial for accurately representing its behavior in molecular dynamics simulations. (Full scripts used in this example can be found in `here </home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/package_log/repository_test/doc/source/example/POPC>`_.)

Optimization Objective
-----------------------
The primary objective of this optimization process is to fine-tune the force field parameters of the POPC system to match its behavior in molecular dynamics simulations with experimental data and fine-grained simulation. Key properties targeted for optimization include membrane thickness, area per lipid, and potential of mean force U(r).

.. note::

   To fit the microscopic properties from fine-grained simulaiton, it is needed to prepare the fine-grained trajectory and topology file in advance.

Optimization Workflow
----------------------
#. **System Setup**: The molecular topology of the POPC system is generated based on provided structural information. Bond and angle parameters are adjusted to ensure the proper representation of POPC molecules.

#. **Objective Function Definition**: An objective function is defined to quantify the deviation between simulation results and target vaules. This function computes various terms, including force match loss, U(r) Boltzmann inversion loss, membrane thickness loss, area per lipid loss.

#. **Optimization Execution**: The optimization process is executed using the Particle Swarm Optimization technique. The optimizer iteratively adjusts the force field parameters to minimize the total loss function.

#. **Result Analysis**: The optimized parameters and corresponding loss values are recorded. Additionally, the running time of the optimization task is logged for evaluation.

Environment Setup and Module Importing
---------------------------------------
The script sets up the environment by adding the path to the GROMACS executable directory to the ``PATH`` environment variable. It imports necessary modules from the AMOFMS package for optimization, force field parameter handling, and utility functions.

.. code-block:: python

    import os
    import time

    os.environ['PATH'] += ':/home/xiaoyedi/data/research/tools/gromacs-2023.3/bin'

    from AMOFMS.Optimization import OptParametersProcess, ParticleSwarmOptimizer, Particle
    from AMOFMS.ObjectiveFunction import BottomUpObjectiveFunction
    from AMOFMS.tools.utilies import mkdir
    from AMOFMS.tools.math_tools import square_diff_of_elements_in_2D_list
    from AMOFMS.tools.properties import MembraneProperties
    from AMOFMS.tools.gromacs import find_gmx_executable

Initialization and Configuration
--------------------------------
Paths, parameters, and options for the optimization task are initialized and configured. Initial settings include paths to input files, output folders, simulation parameters, and optimization parameters.

.. code-block:: python

    time_start = int(time.time())
    timeArray_start = time.localtime(time_start)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray_start)
    print(f'\nStarting Optimization Task at {start_time} \n')

    fg_topology = 'aa/prod/prod.tpr'
    fg_trajectory = 'aa/prod/prod_whole.xtc'
    gmx = find_gmx_executable()

    opt_folder = './opt'
    opt_folder = os.path.realpath(opt_folder)
    loss_dat = os.path.join(opt_folder, 'loss.dat')
    mdp_folder = './mdp'
    initial_gro = 'packmol/mix.gro'
    initial_pdb = 'packmol/mix.pdb'
    index_file = './index.ndx'
    run_em = True
    em_double = False
    run_anneal = False
    run_eq = True
    run_prod = True

Objective Function Definition
------------------------------
An objective function (`opt_loss_function`) is defined to compute the loss during optimization. The function calculates various terms of the loss, including force match loss, U(r) Boltzmann inversion loss, membrane thickness loss, area per lipid loss.

.. code-block:: python

    tmp_result = os.path.join(opt_folder, 'iter_result.dat')
    with open(tmp_result, 'w') as f:
        line = f'{"Iteration":<12}  {"Bead_id":<12}  {"Ur_loss":<12}  {"APL(Angstrom**-2, exp:63)":<12}  {"Thickness(Angstrom, exp:37)":<12}  ' \
               f'{"Ka(mN/m, exp:255)":<12} \n'
        f.write(line)

    def opt_loss_function(particle: Particle):
        print(f'\n{particle.iter}-iter:  {particle.idx}-th particle processing...')
        new_topology = opt_para.unpack_updated_parameters_to_top(updated_parameters_array=particle.position)
        iter_folder = os.path.join(opt_folder, f'iters/iter_{particle.iter}')
        idx_folder = os.path.join(iter_folder, f'{particle.idx}')
    
        bottom_up_obj.update_system_topology(new_system_top=new_topology)
        bottom_up_obj.update_opt_folder(new_opt_folder=idx_folder)
    
        total_loss = 0
        each_term_loss = {}
    
        print('\nComputing Ur Boltzmann inversion loss...')
        bottom_up_obj.run_cg_simulation(initial_gro=initial_gro, fg_resname_list=resname_cg_from_fg_coord,
                                        mdp_folder=mdp_folder, index_file=index_file, table_file=None,
                                        cg_simulation_folder=idx_folder, em=run_em, em_double_version=em_double, anneal=run_anneal,
                                        gpu_acceleration=False,
                                        eq=run_eq, prod=run_prod, nt=cpu_nt, gpu_id=None)
        cg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(rdf_pairs_list=cg_rdf_pairs_list, tag='cg',
                                                            Temperature=temperature, bin_width=rdf_binwidth, max_distance=rdf_cutoff)
        Ur_loss = Ur_loss_ratio * square_diff_of_elements_in_2D_list(list1=fg_pair_Ur_list, list2=cg_pair_Ur_list)
        each_term_loss.update({'Ur': Ur_loss})
        total_loss += Ur_loss
    
    
        print('\nComputing membrane properties...')
        mem = MembraneProperties(topology=bottom_up_obj.cg_topology,
                                 trajectory=bottom_up_obj.cg_trajectory)
        cg_apl, _ = mem.compute_apl(headgroup_selection=head_group_expression)
        cg_thickness, _ = mem.compute_membrane_thickness(headgroup_selection=head_group_expression)
        cg_ka = mem.compute_Ka(head_group_expression, temperature)
    
        print(f'{particle.iter}-iter {particle.idx}-th particle APL(Angstrom**-2): \nexp: {exp_apl} cg: {cg_apl}')
        print(f'{particle.iter}-iter {particle.idx}-th particle Thickness(Angstrom): \nexp: {exp_thickness} cg: {cg_thickness}')
        print(f'{particle.iter}-iter {particle.idx}-th particle Ka(mN/m): \nexp: {exp_ka} cg: {cg_ka}')
    
        apl_loss = apl_loss_ratio * abs(exp_apl - cg_apl)  # https://doi.org/10.1021/acs.jpcb.6b01870
        thickness_loss = thickness_loss_ratio * abs(exp_thickness - cg_thickness)  # https://doi.org/10.1063/1.4936909
        ka_loss = ka_loss_ratio * abs(exp_ka - cg_ka)
    
        each_term_loss.update({'apl': apl_loss})
        each_term_loss.update({'thickness': thickness_loss})
        each_term_loss.update({'ka': ka_loss})
        total_loss = total_loss + apl_loss + thickness_loss + ka_loss
    
        with open(tmp_result, 'a+') as f:
            line = f'{particle.iter:<12}  {particle.idx:<12}  {Ur_loss:<12}  {cg_apl:<12}  {cg_thickness:<12} {cg_ka:<12}\n'
            f.write(line)
    
        print(f'\n{particle.iter}-iter:  {particle.idx}-th particle Done!')
        return total_loss, each_term_loss

Optimization Execution
-----------------------
The :ref:`ParticleSwarmOptimizer` is utilized for the optimization process. The optimizer iteratively adjusts the optimization parameters to minimize the total loss function.

.. code-block:: python

    optimizer = ParticleSwarmOptimizer(objective_function=opt_loss_function, update_boundary_frequency=max_iter,
                                       bounds=opt_para_boundary, num_particles=num_particles, max_iter=max_iter, max_no_improvement_iters=max_iter)

    best_para, best_score, recorder = optimizer.optimize_mpi(max_processes=max_processes)
    recorder.write_losses_to_file(filepath=loss_dat)

Conclusion
----------
This script provides a framework for conducting optimization tasks involving molecular dynamics simulations and force field parameter optimization for the POPC system. It integrates various functionalities from the AMOFMS package and allows for flexible customization based on specific optimization requirements.

