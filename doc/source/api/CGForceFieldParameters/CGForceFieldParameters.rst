InitCGForceFieldParameters
==========================

Initializes and calculates the Coarse-Grained (CG) force field parameters based on the mapping from a Fine-Grained (FG) simulation.

This class processes the mapping from FG to CG representation and calculates necessary force field parameters including Lennard-Jones (LJ) parameters, bond lengths and strengths, angle and dihedral parameters. It supports setting ring constraints and customizing the force field parameter calculation for different CG models.

Attributes
----------
- ``mapping_item`` (object): Stores the provided mapping information.
- ``mol_name`` (str): Name of the molecule.
- ``cg_model`` (str): Coarse-Grained model name.
- ``num_mols`` (int): Number of molecules.
- ``bond_parameters`` (dict): Dictionary storing bond parameters.
- ``angle_parameters`` (dict): Dictionary storing angle parameters.
- ``dihedral_parameters`` (dict): Dictionary storing dihedral parameters.
- ``lj_parameters`` (dict, optional): Dictionary storing Lennard-Jones parameters, initialized upon calling ``init_lj_parameters``.

Methods
-------
- ``add_fg_groups(fg_groups)``: Assigns Fine-Grained groups to Coarse-Grained beads.
- ``init_lj_parameters()``: Initializes the Lennard-Jones parameters based on CG model.
- ``init_bond_parameters()``: Initializes bond parameters for the CG model.
- ``init_angle_parameters()``: Initializes angle parameters for the CG model.
- ``init_dihedral_parameters()``: Initializes dihedral parameters for the CG model.
- ``set_bond_parameters(traj, top, res_name, num_mol, num_atom_with_H, bond=None, group_list=None)``: Sets the bond parameters for the CG model using simulation data.
- ``set_angle_parameters(traj, top, res_name, num_mol, num_atom_with_H, angle=None, group_list=None)``: Sets the angle parameters for the CG model using simulation data.
- ``set_dihedral_parameters(traj, top, res_name, num_mol, num_atom_with_H, dihedral=None, group_list=None)``: Sets the dihedral parameters for the CG model using simulation data.
- ``set_bond_constraint(atom1, atom2, constraint_length, constraint_type=1)``: Sets a bond constraint between two atoms.

Example Usage
-------------
.. code-block:: python

    from AMOFMS import InitCGForceFieldParameters

    mapping_item = ...  # Obtain mapping information
    num_mols = 100
    init_cg_ff = InitCGForceFieldParameters(mapping_item, num_mols)
    init_cg_ff.init_lj_parameters()
    init_cg_ff.set_bond_parameters(traj, top, res_name, num_mol, num_atom_with_H)
    init_cg_ff.set_angle_parameters(traj, top, res_name, num_mol, num_atom_with_H)
    cg_ff_parameters = init_cg_ff.cg_ff_parameters_item

.. _generate_system_top:

generate_system_top
===================

Generates the system topology for a Coarse-Grained model.

Parameters
----------
- ``mols`` (list of dicts): List of molecule information dictionaries.
- ``num_mols`` (list of int, optional): List specifying the number of each molecule type.

Returns
-------
dict
    A dictionary representing the system topology, including molecules and Lennard-Jones cross terms.

Example Usage
-------------
.. code-block:: python

    from AMOFMS import generate_system_top

    mol_A = {'mol_name': 'A', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
             'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
             'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}

    mol_B = {'mol_name': 'B', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
             'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
             'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}

    system_top = generate_system_top(mols=[mol_A, mol_B], num_mols=[2, 3])
