.. _BottomUpObjectiveFunction:

:class:`BottomUpObjectiveFunction`
==================================
Defines the bottom-up approach for optimizing Coarse-Grained (CG) models based on Fine-Grained (FG) simulation data.

This class is responsible for managing the overall process of converting FG simulation data into a CG model,
running CG simulations, and evaluating the performance of the CG model through various objective functions.

Attributes
----------
- ``system_top`` (dict): The system topology information derived from the FG simulation.
- ``molecules`` (list): A list of dictionaries, each representing a molecule in the system.
- ``molecule_list`` (list): A list of molecule names present in the system.
- ``pdb_file`` (str, optional): Path to the PDB file, if available.
- ``fg_topology`` (str): Path to the FG topology file.
- ``fg_trajectory`` (str): Path to the FG trajectory file.
- ``cg_topology`` (str, optional): Path to the generated CG topology file, initialized as None.
- ``cg_trajectory`` (str, optional): Path to the CG trajectory file, initialized as None.
- ``opt_folder`` (str): Path to the optimization folder where results and intermediate files are stored.
- ``cg_top_folder`` (str): Path to the folder containing the CG topology files.
- ``cg_top_file`` (str): Path to the CG topology file.

Methods
-------
- ``__init__(self, system_top, fg_topology, fg_trajectory, opt_folder, cg_top_file_name='cg.top', pdb_file=None)``: Initializes the class with system topology, FG simulation data, and optimization settings.
- ``update_system_topology(self, new_system_top)``: Updates the system topology information.
- ``update_opt_folder(self, new_opt_folder)``: Updates the optimization folder path and related settings.
- ``force_match_loss(self, cg_mdp_file, fg_resname_list=None, begin_frame=None, end_frame=None, skip_frame=None)``: Calculates the force matching loss between FG and CG simulations.
- ``run_cg_simulation(self, mdp_folder, initial_gro=None, fg_resname_list=None, index_file=None, cg_simulation_folder=None, table_file=None, gpu_acceleration=True, em_double_version=True, em=True, anneal=True, eq=True, prod=True, nt=8, gpu_id=0)``: Runs the CG simulation based on the provided settings and updates the CG trajectory and topology paths.
- ``Boltzmann_inversion(self, rdf_pairs_list, tag, max_distance, rdf_folder=None, begin_frame_id=None, end_frame_id=None, skip_frames=None, Temperature=300, bin_width=0.002)``: Performs Boltzmann inversion based on the Radial Distribution Function (RDF) to derive potential functions.

Example Usage
-------------

Basic usage of the `BottomUpObjectiveFunction` class:

.. code-block:: python

    from CGPropertiesFromFGSimulation import ComputeCGPropertiesFromFGSimulation_All, extrat_cg_force_all
    from tools.gromacs import generate_top_file, run_gromacs_simulation, unwrap_trajectory
    import os
    import numpy as np
    from tools.utilies import mkdir, delete_files
    from tools.properties import RDF

    system = {'molecules': [{'mol_name': '12oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'],
                             'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
                             'aa_groups': [[0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                           [4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28],
                                           [8, 9, 10, 11, 29, 30, 31, 32, 33, 34, 35]],
                             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47],
                                               ('C1', 'C1'): [3.5, 0.47]},
                             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
                             'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 100},
                            {'mol_name': '16oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'C1', 'P1'],
                             'id': [0, 1, 2, 3], 'charge': [0.0, 0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 56.1067, 59.0869],
                             'aa_groups': [[0, 1, 2, 3, 16, 17, 18, 19, 20,21,22,23,24],
                                           [4, 5, 6, 7, 25, 26, 27, 28,29,30,31,32],
                                           [8, 9, 10, 11, 33, 34, 35,36,37,38,39,40],
                                           [12,13,14,15,41,42,43,44,45,46,47]],
                             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47],
                                               ('C1', 'C1'): [3.5, 0.47]},
                             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0], (2, 3): [0.47, 1250.0]},
                             'angle_parameters': {(0, 1, 2): [180.0, 25.0], (1, 2, 3): [180.0, 25.0]}, 'num_mols': 100}],
              'lj_cross_terms': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
              'cgmodel': 'MARTINI2'}

    opt_object = BottomUpObjectiveFunction(system_top=system, fg_topology='path_to_fg_top', fg_trajectory='path_to_fg_traj', opt_folder='path_to_opt_folder')
    opt_object.force_match_loss(cg_mdp_file='path_to_cg_mdp_file')

