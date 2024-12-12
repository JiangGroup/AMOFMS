:class:`ComputeCGPropertiesFromFGSimulation`
============================================

A module for computing Coarse-Grained (CG) properties from Fine-Grained (FG) simulation data.

Attributes
----------

- ``topology`` (str): Path to the topology file.
- ``trajectory`` (str): Path to the trajectory file.
- ``cg_groups`` (dict, optional): Definition of CG groups.
- ``atom_num`` (int, optional): Number of atoms per molecule.
- ``mol_num`` (int, optional): Number of molecules.
- ``residue_name`` (str, optional): Name of the residue to select atoms from.
- ``universe`` (MDAnalysis.Universe): The Universe object containing loaded simulation data.

Methods
-------

- ``__init__(self, topology, trajectory, begin_frame_id=None, end_frame_id=None, skip_frames=None)``: Constructor to initialize the ComputeCGPropertiesFromFGSimultion class.
- ``set_cg_groups(self, cg_groups, atom_num, mol_num, residue_name)``: Sets the CG groups and related properties for analysis.
- ``extract_atoms_by_residue(self)``: Extracts atoms by residue name from the loaded simulation data.
- ``compute_centroid(self, atom_group)``: Computes the centroid (center of mass) of a given atom group.
- ``compute_cg_bond_distribution(self, group1, group2)``: Computes the distribution of distances between centroids of two CG groups across the trajectory.
- ``compute_cg_angle_distribution(self, group1, group2, group3)``: Computes the distribution of angles formed by centroids of three CG groups.
- ``compute_cg_dihedral_distribution(self, group1, group2, group3, group4)``: Computes the distribution of dihedrals formed by centroids of four CG groups.
- ``plot_distribution(self, data, title, xlabel, ylabel)``: Plots the distribution of a given data set using matplotlib.
- ``compute_fg_group_force(self, fg_groups)``: Computes forces on FG groups mapped to CG groups.

Example Usage
-------------

This example demonstrates how to use the ComputeCGPropertiesFromFGSimulation module:

.. code-block:: python

    from AMOFMS import ComputeCGPropertiesFromFGSimulation

    # Define paths to trajectory and topology files
    traj = '/path/to/trajectory.trr'
    top = '/path/to/topology.tpr'

    # Define system properties
    system = {
        'molecules': [
            {'mol_name': '12oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'],
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
             'angle_parameters': {(0, 1, 2): [180.0, 25.0], (1, 2, 3): [180.0, 25.0]}, 'num_mols': 100}
        ],
        'lj_cross_terms': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
        'cgmodel': 'MARTINI2'
    }

    # Create an instance of the analyzer
    analyzer = ComputeCGPropertiesFromFGSimulation(topology=top, trajectory=traj)

    # Set CG groups and related properties
    analyzer.set_cg_groups(atom_num=36, mol_num=100, cg_groups=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11]], residue_name='12oh')

    # Compute bond distribution between CG groups 0 and 1
    distances_per_frame, _, _ = analyzer.compute_cg_bond_distribution(group1=0, group2=1)

    # Compute angle distribution between CG groups 0, 1, and 2
    angle = analyzer.compute_cg_angle_distribution(group1=0, group2=1, group3=2)

    # Compute dihedral distribution between CG groups 0, 1, 2, and 3
    dihedral = analyzer.compute_cg_dihedral_distribution(group1=0, group2=1, group3=2, group4=3)


:func:`wrap_coordinates`
========================

Wraps coordinates within the boundaries of the simulation box.

This function ensures that coordinates are wrapped back into the simulation box, which is essential for periodic boundary conditions. It is applicable for rectangular simulation boxes.

Parameters
----------

- ``coord`` (np.ndarray): The coordinates to be wrapped. Can be a single set of coordinates or an array of coordinates.
- ``dimensions`` (np.ndarray): The dimensions of the simulation box, typically in the form [Lx, Ly, Lz], where Lx, Ly, and Lz are the lengths of the simulation box along the x, y, and z axes, respectively.

Returns
-------

np.ndarray: The wrapped coordinates, ensuring all points are within the simulation box boundaries.


:func:`extrat_cg_force_all`
===========================

Extracts the forces acting on all Coarse-Grained (CG) groups within the system for a specified frame.

This function iterates over all molecules described in the system topology, selecting specific residues based on the provided names, and then computes the total force acting on each CG group within those molecules.

Parameters
----------

- ``topology`` (str): The path to the topology file.
- ``trajectory`` (str): The path to the trajectory file.
- ``system_top`` (dict): A dictionary representing the system's topology, including molecules and their properties.
- ``frame_id`` (int, optional): The frame index at which to compute the forces. Default is 0.

Returns
-------

dict: A dictionary with molecule names as keys and numpy arrays of forces (for each CG group in those molecules) as values.


:class:`ComputeCGPropertiesFromFGSimulation_All`
================================================

Computes Coarse-Grained (CG) properties from Fine-Grained (FG) simulation data for an entire system.

This class is designed to handle multiple molecule types and calculate CG properties, such as centroids and forces, for all molecules present in the simulation. It supports operations on the entire system, making it suitable for analyses that require a global perspective of the simulation data.

Attributes
----------

- ``topology`` (str): Path to the topology file.
- ``trajectory`` (str): Path to the trajectory file.
- ``system_top`` (dict): A dictionary representing the system's topology, including information about molecules.
- ``molecules`` (list): A list containing dictionaries for each molecule type in the system.
- ``universe`` (MDAnalysis.Universe): The Universe object containing loaded simulation data.

Methods
-------

- ``__init__(self, topology, trajectory, system_top, begin_frame_id=None, end_frame_id=None, skip_frames=None)``: Constructor to initialize the class with simulation and system topology data.
- ``get_num_frames(self)``: Returns the number of frames in the trajectory.
- ``compute_centroid(self, atom_group)``: Computes the centroid (center of mass) for a given atom group.
- ``compute_all_fg_group_force(self, fg_resname_list=None, frame_id=0)``: Computes the forces acting on all FG groups within the system for a specified frame.
- ``save_cg_coord_from_fg(self, save_file='cg.gro', fg_resname_list=None, save_molecule_list=None, frame_id=-1, method='com', cg_wrap=False)``: Saves the CG coordinates derived from FG data to a file.

    Explanation of Parameters:
    - ``save_file`` (str, optional): The name of the file to save the CG coordinates to. Default is 'cg.gro'.
    - ``fg_resname_list`` (list of str, optional): A list of residue names to use instead of molecule names. Default is None.
    - ``save_molecule_list`` (list of int, optional): A list of indices specifying which molecules to save. Default is None (save all).
    - ``frame_id`` (int, optional): The frame index from which to save the CG coordinates. Default is -1 (last frame).
    - ``method`` (str, optional): The method to compute CG coordinates ('com' for center of mass, 'cog' for center of geometry). Default is 'com'.
    - ``cg_wrap`` (bool, optional): Whether to wrap the coordinates back into the simulation box. Default is False.

Example Usage
-------------

This example demonstrates how to use the ComputeCGPropertiesFromFGSimulation_All module:

.. code-block:: python

    from AMOFMS import ComputeCGPropertiesFromFGSimulation
    
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

    traj2 = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/force_match/simulation.trr'
    top2 = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/force_match/simulation.tpr'

    a = extrat_cg_force_all(topology=top2, trajectory=traj2, system_top=system, frame_id=0)

    computation = ComputeCGPropertiesFromFGSimulation_All(topology=top, trajectory=traj, system_top=system)
    computation.compute_all_fg_group_force(frame_id=0)
    computation.save_cg_coord_from_fg(save_file='cg.gro', frame_id=-1, method='com')

