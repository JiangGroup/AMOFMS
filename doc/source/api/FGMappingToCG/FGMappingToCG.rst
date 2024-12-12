MappingItem
===========

Represents a mapping item for converting a fine-grained (FG) molecule to a coarse-grained (CG) model, facilitating the transition from detailed molecular structures to simplified representations for computational efficiency.

Attributes
----------

- ``smiles`` (str): The SMILES representation of the molecule.
- ``mol_name`` (str, optional): An optional name for the molecule.
- ``cg_model`` (str): The name of the coarse-grained model being applied.
- ``cg_groups`` (list of lists, optional): A list where each sublist specifies the atoms that make up each coarse-grained group within the model.
- ``cg_group_id`` (list, optional): A list of identifiers for each coarse-grained group.
- ``cg_group_type`` (list, optional): A list indicating the type of each coarse-grained group.
- ``num_cg_bead`` (int): The total number of coarse-grained beads or groups into which the molecule has been simplified.
- ``mol`` (RDKit Mol object): An RDKit molecule object created from the SMILES string.
- ``num_atom`` (int): The total number of atoms in the molecule.
- ``atom_element`` (list, optional): A list containing the element type of each atom in the molecule.
- ``cg_group_mass`` (list): A list of calculated masses for each coarse-grained group.
- ``cg_bond`` (list): A list describing the bonds between coarse-grained groups.
- ``cg_angle`` (set): A set of angles between coarse-grained groups.
- ``cg_dihedral`` (set): A set of dihedral angles involving coarse-grained groups.
- ``atom_cg_group_id`` (list): A list mapping each atom to its corresponding coarse-grained group ID.
- ``atom_coord_matrix`` (np.ndarray, optional): An array of coordinates for each atom in the molecule.
- ``cg_coord_matrix`` (np.ndarray, optional): An array of coordinates for each coarse-grained bead.
- ``cg_charge`` (list): A list of charges for each coarse-grained group.
- ``cg_dicts`` (list): A list of additional coarse-grained properties.

Methods
-------

- ``__init__(self, smiles, mol_name=None, cgmodel='MARTINI2', cg_groups=None, cg_group_id=None, cg_group_type=None, atom_element=None)``:
    Initializes the MappingItem with molecular information and prepares the CG model setup.

- ``set_cg_groups(self, cg_groups)``:
    Sets the CG groups for the molecule, organizing atoms into simplified representations.

- ``set_cg_group_id(self, cg_group_id)``:
    Assigns identifiers to each CG group, facilitating their manipulation and analysis.

- ``set_cg_group_type(self, cg_group_type)``:
    Defines the type for each CG group, which may influence their interactions in simulations.

- ``check_cg_groups_id_type_match(self)``:
    Ensures consistency among CG groups, their identifiers, and types, verifying the integrity of the mapping.

- ``compute_cg_group_mass(self)``:
    Calculates the mass of each CG group based on its constituent atoms, essential for dynamics simulations.

- ``compute_atom_group_cg_id(self)``:
    Establishes a mapping from each atom to its corresponding CG group, bridging detailed and simplified models.

- ``compute_cg_bond(self, edges)``:
    Determines the bonds between CG groups, constructing the simplified model's structural framework.

- ``are_connected(self, atom1, atom2)``:
    Checks if two atoms are connected within the CG model, aiding in structural analysis.

- ``compute_cg_angle(self)``:
    Identifies angles formed between CG groups, important for maintaining molecular geometry.

- ``compute_cg_dihedral(self)``:
    Finds dihedral angles involving CG groups, crucial for understanding conformational properties.

- ``set_cg_charge(self, charge_list)``:
    Assigns charges to CG groups, a key factor in modeling electrostatic interactions.

- ``set_cg_mass(self, mass_list)``:
    Specifies the mass for each CG group, influencing the molecule's dynamics.

- ``compute_atom_coords_from_smiles(self)``:
    Derives atom coordinates from the SMILES representation, providing spatial information for the molecule.

- ``compute_cg_coords(self, method='mass center')``:
    Calculates the coordinates of CG beads, offering a simplified spatial representation of the molecule.

- ``save_atom_coords_file(self, output='./molecule.pdb')``:
    Saves the molecule's atom coordinates to a file, supporting various formats for further analysis or visualization.

Example Usage
-------------

Create a MappingItem instance for a molecule with a given SMILES representation:

.. code-block:: python

    from rdkit import Chem
    from AMOFMS import MappingItem

    # Define the molecule's SMILES representation
    smiles = 'CCCCC'
    # Create a MappingItem instance
    mapping_item = MappingItem(smiles)

    # Access the attributes and methods of the MappingItem instance
    print(mapping_item.smiles)
    print(mapping_item.num_atom)

.. _MappingToCGfromDSGPM_TP:

MappingToCGfromDSGPM_TP
=======================

Facilitates the conversion of a molecule from a detailed structural representation to a specified coarse-grained (CG) model using Deep Supervised Graph Partitioning Model with Type Prediction Enhancement (DSGPM-TP).

Attributes
----------

- ``CGmodel`` (str): The coarse-grained model to apply.
- ``smiles`` (str): The SMILES string of the molecule.
- ``CG_num_bead`` (int): The desired number of coarse-grained beads in the final model.
- ``output_dir`` (str): The directory where output files will be saved.
- ``mol_json`` (dict): A JSON-like dictionary containing the resulting CG structure information.
- ``mol_name`` (str, optional): An optional name for the molecule.

Methods
-------

- ``__init__(self, CG_num_bead, CGmodel='MARTINI2', mol_name=None, mol_form='sml', smiles=None, pdb_file=None, output_dir='./Mapping')``:
    Initializes the mapping process, setting up the target CG model, molecular structure, and output specifications.

- ``get_mapping_item(self)``:
    Returns the MappingItem instance created during the initialization.

Example Usage
-------------

Convert a molecule from a detailed structural representation to a coarse-grained model using MappingToCGfromDSGPM_TP:

.. code-block:: python

    from AMOFMS import MappingToCGfromDSGPM_TP

    # Define the SMILES representation of the molecule
    smiles = 'CCCCC'
    # Define the number of coarse-grained beads
    num_bead = 4
    # Create a MappingToCGfromDSGPM_TP instance
    mol_mapping = MappingToCGfromDSGPM_TP(CG_num_bead=num_bead, smiles=smiles)
    # Get the MappingItem instance for further analysis
    mapping_item = mol_mapping.get_mapping_item()

    # Access the attributes and methods of the MappingItem instance
    print(mapping_item.cg_groups)
    print(mapping_item.cg_bond)


