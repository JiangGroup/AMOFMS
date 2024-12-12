from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import openbabel


class MappingItem:
    """
    Represents a mapping item for converting a fine-grained (FG) molecule to a coarse-grained (CG) model,
    facilitating the transition from detailed molecular structures to simplified representations for computational efficiency.

    Attributes:
    -----------
    smiles : str
        The SMILES representation of the molecule, providing a textual identifier that encodes its structure.
    mol_name : str, optional
        An optional name for the molecule, allowing for easier identification and reference.
    cg_model : str
        The name of the coarse-grained model being applied (e.g., MARTINI2), which dictates the rules for simplification.
    cg_groups : list of lists, optional
        A list where each sublist specifies the atoms that make up each coarse-grained group within the model.
    cg_group_id : list, optional
        A list of identifiers for each coarse-grained group, facilitating tracking and manipulation.
    cg_group_type : list, optional
        A list indicating the type of each coarse-grained group, which can influence interactions in simulations.
    num_cg_bead : int
        The total number of coarse-grained beads or groups into which the molecule has been simplified.
    mol : RDKit Mol object
        An RDKit molecule object created from the SMILES string, used for various molecular computations.
    num_atom : int
        The total number of atoms in the molecule, derived from the RDKit molecule object.
    atom_element : list, optional
        A list containing the element type of each atom in the molecule, used for detailed analysis or reconstructions.
    cg_group_mass : list
        A list of calculated masses for each coarse-grained group, important for dynamics and thermodynamics calculations.
    cg_bond : list
        A list describing the bonds between coarse-grained groups, essential for maintaining structural integrity in models.
    cg_angle : set
        A set of angles between coarse-grained groups, crucial for preserving molecular geometry.
    cg_dihedral : set
        A set of dihedral angles involving coarse-grained groups, affecting conformational flexibility and energetics.
    atom_cg_group_id : list
        A list mapping each atom to its corresponding coarse-grained group ID, linking detailed and simplified representations.
    atom_coord_matrix : np.ndarray, optional
        An array of coordinates for each atom in the molecule, used for spatial analysis and visualization.
    cg_coord_matrix : np.ndarray, optional
        An array of coordinates for each coarse-grained bead, representing the simplified structure's spatial configuration.
    cg_charge : list
        A list of charges for each coarse-grained group, integral to electrostatic calculations in simulations.
    cg_dicts : list
        A list of additional coarse-grained properties, providing a mechanism for extending the model's detail (not fully implemented).

    Methods:
    --------
    __init__(self, smiles, mol_name=None, cgmodel='MARTINI2', cg_groups=None, cg_group_id=None, cg_group_type=None, atom_element=None):
        Initializes the MappingItem with molecular information and prepares the CG model setup.

    set_cg_groups(self, cg_groups):
        Sets the CG groups for the molecule, organizing atoms into simplified representations.

    set_cg_group_id(self, cg_group_id):
        Assigns identifiers to each CG group, facilitating their manipulation and analysis.

    set_cg_group_type(self, cg_group_type):
        Defines the type for each CG group, which may influence their interactions in simulations.

    check_cg_groups_id_type_match(self):
        Ensures consistency among CG groups, their identifiers, and types, verifying the integrity of the mapping.

    compute_cg_group_mass(self):
        Calculates the mass of each CG group based on its constituent atoms, essential for dynamics simulations.

    compute_atom_group_cg_id(self):
        Establishes a mapping from each atom to its corresponding CG group, bridging detailed and simplified models.

    compute_cg_bond(self, edges):
        Determines the bonds between CG groups, constructing the simplified model's structural framework.

    are_connected(self, atom1, atom2):
        Checks if two atoms are connected within the CG model, aiding in structural analysis.

    compute_cg_angle(self):
        Identifies angles formed between CG groups, important for maintaining molecular geometry.

    compute_cg_dihedral(self):
        Finds dihedral angles involving CG groups, crucial for understanding conformational properties.

    set_cg_charge(self, charge_list):
        Assigns charges to CG groups, a key factor in modeling electrostatic interactions.

    set_cg_mass(self, mass_list):
        Specifies the mass for each CG group, influencing the molecule's dynamics.

    compute_atom_coords_from_smiles(self):
        Derives atom coordinates from the SMILES representation, providing spatial information for the molecule.

    compute_cg_coords(self, method='mass center'):
        Calculates the coordinates of CG beads, offering a simplified spatial representation of the molecule.

    save_atom_coords_file(self, output='./molecule.pdb'):
        Saves the molecule's atom coordinates to a file, supporting various formats for further analysis or visualization.

    """

    def __init__(self, smiles, mol_name=None, cgmodel='MARTINI2', cg_groups=None, cg_group_id=None, cg_group_type=None, atom_element=None):
        """
        Initializes a MappingItem object with the given molecular information and prepares it for the coarse-graining process.

        Parameters:
        -----------
        smiles (str): The SMILES string of the molecule to be coarse-grained.
        mol_name (str, optional): An optional name for the molecule.
        cgmodel (str): The coarse-grained model to be applied (e.g., MARTINI2).
        cg_groups (list of lists, optional): The initial grouping of atoms into coarse-grained beads.
        cg_group_id (list, optional): Identifiers for the coarse-grained groups.
        cg_group_type (list, optional): The type of each coarse-grained group.
        atom_element (list, optional): The element type of each atom in the molecule.

        This method sets up the molecular model based on the provided SMILES string, initializes coarse-grained group
        definitions, and calculates initial properties such as group masses. It lays the groundwork for further
        computations needed to complete the coarse-graining process.

        The initialization process involves several key steps:
        - Conversion of the SMILES string into an RDKit molecule object, enabling detailed molecular manipulation and analysis.
        - Calculation of the total number of atoms in the molecule, providing a foundation for the coarse-graining process.
        - Setup of coarse-grained groups, if provided, which involves assigning atoms to CG beads based on the input configuration.

        Additional steps include the initialization of lists and sets for storing CG properties such as masses, charges, bonds, angles, and dihedrals. These properties are essential for constructing the CG model and for accurately capturing the molecule's physical and chemical characteristics in a simplified manner.

        """

        self.smiles = smiles
        self.mol_name = mol_name
        self.cg_model = cgmodel
        self.cg_groups = cg_groups
        self.cg_group_id = cg_group_id
        self.cg_group_type = cg_group_type
        self.num_cg_bead = 0

        self.mol = Chem.MolFromSmiles(self.smiles)  # Convert SMILES to an RDKit molecule ob
        self.num_atom = self.mol.GetNumAtoms()  # Get the number of atoms from the molecule
        self.atom_element = atom_element
        self.cg_group_mass = []
        self.cg_bond = []
        self.cg_angle = set()
        self.cg_dihedral = set()
        self.atom_cg_group_id = list(range(self.num_atom))  # Initially map each atom to itself
        self.atom_coord_matrix = None
        self.cg_coord_matrix = None
        self.cg_charge = []

        self.cg_dicts = []

        # Verify and set CG groups if provided
        if cg_groups is not None:
            self.num_cg_bead = len(self.cg_groups)
            self.cg_group_id = list(range(self.num_cg_bead))  # Initialize mapping of atoms to CG groups
            self.cg_charge = [0.0 for _ in range(self.num_cg_bead)]  # Initialize charges for CG groups

            all_atom_id = set()
            for i in self.cg_groups:
                all_atom_id.update(set(i))
            assert self.num_atom == len(all_atom_id)
            assert not len(all_atom_id - set(range(self.num_atom))), \
                'the group in cg groups have overlapped or some atom id are exceeded to self.num_cg_bead.'

            self.compute_atom_group_cg_id()

    def set_cg_groups(self, cg_groups):
        """
        Assigns coarse-grained groups to the molecule and updates related attributes.

        Parameters:
        cg_groups (list of lists): Each sublist represents a coarse-grained group and contains atom indices belonging to that group.

        This method updates the coarse-grained groups, their count, and ensures each atom is uniquely assigned to a group,
        ensuring no overlap or missing atom indices.
        """
        self.cg_groups = cg_groups
        self.num_cg_bead = len(self.cg_groups)   # Update the number of CG beads/groups based on the input
        self.cg_group_id = range(self.num_cg_bead)

        # Collect all atom indices from the CG groups to ensure unique assignment
        all_atom_id = set()
        for i in self.cg_groups:
            all_atom_id.update(set(i))  # Add atom indices from each group to the set

        # The number of unique atoms should match the length of all_atom_id set
        self.num_atom = len(all_atom_id)
        assert not len(all_atom_id - set(range(self.num_atom))), \
            'the group in cg groups have overlapped or some atom id are exceeded to self.num_cg_bead.'

    def set_cg_group_id(self, cg_group_id):
        """
        Sets the identifiers for each coarse-grained group.

        Parameters:
        cg_group_id (list): A list of identifiers for the coarse-grained groups.

        Validates that the number of identifiers matches the number of CG groups.
        """

        self.cg_group_id = cg_group_id
        assert len(self.cg_group_id) == self.num_cg_bead, \
            'Number of cg groups dose not match with cg group id!'

    def set_cg_group_type(self, cg_group_type):
        """
        Specifies the type for each coarse-grained group.

        Parameters:
        cg_group_type (list): A list describing the type of each coarse-grained group (e.g., hydrophobic, polar).

        Ensures the list length matches the number of CG groups.
        """

        self.cg_group_type = cg_group_type
        assert len(self.cg_group_type) == self.num_cg_bead, \
            'Number of cg groups dose not match with cg type!'

    def check_cg_groups_id_type_match(self):
        """
        Validates that the number of coarse-grained groups matches with the number of IDs and types specified.

        Ensures consistent and complete information for CG groups.
        """

        assert len(self.cg_groups) == len(self.cg_group_type) == len(self.cg_group_id), \
            'Number of cg groups dose not match with cg id or cg type!'

    def compute_cg_group_mass(self):
        """
        Calculates the mass for each coarse-grained group by summing the masses of its constituent atoms and any associated hydrogen atoms.

        Utilizes the standard mass of a hydrogen atom for calculations.
        """

        hydrogen_mass = 1.00784
        for sublist in self.cg_groups:
            total_mass = 0
            for atom_index in sublist:
                atom = self.mol.GetAtomWithIdx(atom_index)
                # Sum the mass of the atom and its bonded hydrogens
                total_mass += atom.GetMass()
                total_mass += atom.GetTotalNumHs() * hydrogen_mass

            self.cg_group_mass.append(round(total_mass, 4))  # Append the rounded total mass for the group

    def compute_atom_group_cg_id(self):
        """
        Maps each atom to its corresponding coarse-grained group ID.

        Iterates over the CG groups to assign each atom in the group its CG ID.
        """
        for idx, i in enumerate(self.cg_groups):
            for j in i:
                self.atom_cg_group_id[j] = idx  # Assign CG group ID to atom

    def compute_cg_bond(self, edges):
        """
        Identifies and stores bonds between coarse-grained groups based on the edges of the molecular graph.

        Parameters:
        edges (list of dicts): Each dictionary represents an edge with 'source' and 'target' keys indicating connected atoms.

        Only bonds between different CG groups are considered.
        """
        self.compute_atom_group_cg_id()  # Ensure atoms are mapped to their CG group IDs
        for i in edges:
            a, b = i['source'],  i['target']
            if self.atom_cg_group_id[a] != self.atom_cg_group_id[b]:
                # If atoms belong to different CG groups, record the bond
                self.cg_bond.append([self.atom_cg_group_id[a],
                                     self.atom_cg_group_id[b]])

    def are_connected(self, atom1, atom2):
        """
        Checks if two atoms are connected within the coarse-grained model.

        Parameters:
        atom1, atom2 (int): Atom indices to check for connectivity.

        Returns:
        bool: True if atoms are connected, False otherwise.
        """

        return [atom1, atom2] in self.cg_bond or [atom2, atom1] in self.cg_bond

    def compute_cg_angle(self):
        """
        Identifies and stores angles within the coarse-grained model by examining connected CG groups.

        Utilizes CG bonds to find connected triples of CG groups forming angles.
        """
        for bond1 in self.cg_bond:
            for bond2 in self.cg_bond:
                if bond1 == bond2:
                    continue

                angle1 = None

                if bond1[1] == bond2[0]:
                    angle1 = (bond1[0], bond1[1], bond2[1])
                elif bond1[0] == bond2[0]:
                    angle1 = (bond1[1], bond1[0], bond2[1])
                elif bond1[1] == bond2[1]:
                    angle1 = (bond1[0], bond1[1], bond2[0])
                elif bond1[0] == bond2[1]:
                    angle1 = (bond1[1], bond1[0], bond2[0])

                if angle1 and (angle1 not in self.cg_angle) and (angle1[::-1] not in self.cg_angle):
                    self.cg_angle.add(angle1)

    def compute_cg_dihedral(self):
        """
        Identifies and stores dihedrals within the coarse-grained model by examining sequences of connected CG groups.

        Utilizes CG bonds to find connected quadruples of CG groups forming dihedrals.
        """
        for bond1 in self.cg_bond:
            for bond2 in self.cg_bond:
                if bond1 == bond2:
                    continue

                for bond3 in self.cg_bond:
                    if bond3 == bond1 or bond3 == bond2:
                        continue

                    dihedral = None

                    if bond1[1] == bond2[0] and bond2[1] == bond3[0]:
                        dihedral = (bond1[0], bond1[1], bond2[1], bond3[1])
                    elif bond1[0] == bond2[0] and bond2[1] == bond3[0]:
                        dihedral = (bond1[1], bond1[0], bond2[1], bond3[1])
                    elif bond1[1] == bond2[1] and bond2[0] == bond3[0]:
                        dihedral = (bond1[0], bond1[1], bond2[0], bond3[1])
                    elif bond1[0] == bond2[1] and bond2[0] == bond3[0]:
                        dihedral = (bond1[1], bond1[0], bond2[0], bond3[1])
                    elif bond1[1] == bond2[0] and bond2[1] == bond3[1]:
                        dihedral = (bond1[0], bond1[1], bond2[1], bond3[0])
                    elif bond1[0] == bond2[0] and bond2[1] == bond3[1]:
                        dihedral = (bond1[1], bond1[0], bond2[1], bond3[0])
                    elif bond1[1] == bond2[1] and bond2[0] == bond3[1]:
                        dihedral = (bond1[0], bond1[1], bond2[0], bond3[0])
                    elif bond1[0] == bond2[1] and bond2[0] == bond3[1]:
                        dihedral = (bond1[1], bond1[0], bond2[0], bond3[0])

                    if dihedral and (dihedral not in self.cg_dihedral) and (dihedral[::-1] not in self.cg_dihedral):
                        self.cg_dihedral.add(dihedral)

    def set_cg_charge(self, charge_list):
        """
        Assigns charges to the coarse-grained groups.

        Parameters:
        charge_list (list): A list of charges for each CG group.

        Validates that the length of the charge list matches the number of CG groups.
        """

        assert len(charge_list) == self.num_cg_bead, "CG charge list must have the same length"
        self.cg_charge = charge_list

    def set_cg_mass(self, mass_list):
        """
        Updates the mass of each coarse-grained group.

        Parameters:
        mass_list (list): A list of masses for each CG group.

        Validates that the length of the mass list matches the number of CG groups.
        """

        assert len(mass_list) == self.num_cg_bead, "CG mass list must have the same length"
        self.cg_group_mass = mass_list

    def compute_atom_coords_from_smiles(self):
        """
        Generates 3D coordinates for atoms in the molecule from its SMILES representation.

        Utilizes RDKit's ETKDG algorithm for generating the 3D structure.
        """

        AllChem.EmbedMolecule(self.mol, AllChem.ETKDG())
        coords = self.mol.GetConformer().GetPositions()
        self.atom_coord_matrix = np.array(coords)

    def compute_cg_coords(self, method='mass center'):
        """
        Calculates the coordinates of each coarse-grained bead.

        Parameters:
        method (str): The method to use for calculating CG coordinates ('mass center' or 'geometric center').

        Computes either the mass or geometric center of atoms in each CG group to represent the position of the CG bead.
        """

        atomic_masses = [atom.GetMass() for atom in self.mol.GetAtoms()]  # List of atomic masses
        cg_coords = []  # List to store CG coordinates

        for group in self.cg_groups:
            if method == 'mass center':
                # Calculate the mass center for each CG group
                total_mass = sum(atomic_masses[i] for i in group)
                weighted_coords = sum(self.atom_coord_matrix[i] * atomic_masses[i] for i in group)
                cg_coords.append(weighted_coords / total_mass)
            elif method == 'geometric center':
                # Calculate the geometric center for each CG group
                cg_coords.append(np.mean(self.atom_coord_matrix[group], axis=0))

        self.cg_coord_matrix = np.array(cg_coords)  # Store the calculated CG coordinates

    def save_atom_coords_file(self, output='./molecule.pdb'):
        """
        Saves the atomic coordinates to a file in the specified format.

        Parameters:
        output (str): The file path and name for saving the coordinates. The format is determined by the file extension.

        Supports 'pdb' format via RDKit and other formats via Open Babel conversion.
        """

        file_format = output.split('.')[-1].lower()  # Determine the file format from the extension

        # Use RDKit to write PDB format
        if file_format == 'pdb':
            with open(output, 'w') as file:
                file.write(Chem.MolToPDBBlock(self.mol))
        # Use Open Babel for other formats
        else:
            obConversion = openbabel.OBConversion()
            obConversion.SetOutFormat(file_format)
            obmol = openbabel.OBMol()
            _ = obConversion.ReadString(obmol, Chem.MolToMolBlock(self.mol))
            obConversion.WriteFile(obmol, output)


class MappingToCGfromDSGPM_TP:
    """
    Facilitates the conversion of a molecule from a detailed structural representation to a specified coarse-grained (CG) model
    using Deep Supervised Graph Partitioning Model with Type Prediction Enhancement (DSGPM-TP). This class automates the process of mapping based on the
    specified CG model and the molecular structure provided in SMILES format or as a PDB file.

    Attributes:
    -----------
    CGmodel : str
        The coarse-grained model to apply, such as 'MARTINI2', which defines the rules for simplification.
    smiles : str
        The SMILES string of the molecule, providing a compact textual representation of its structure.
    CG_num_bead : int
        The desired number of coarse-grained beads in the final model.
    output_dir : str
        The directory where output files, including mappings and potentially modified structures, will be saved.
    mol_json : dict
        A JSON-like dictionary containing the resulting CG structure information, including nodes and edges representing CG beads and connections.
    mol_name : str, optional
        An optional name for the molecule for identification purposes.

    Methods:
    --------
    __init__(self, CG_num_bead, CGmodel='MARTINI2', mol_name=None, mol_form='sml', smiles=None, pdb_file=None, output_dir='./Mapping'):
        Initializes the mapping process, setting up the target CG model, molecular structure, and output specifications.
        It supports initializing the molecular structure from a SMILES string or directly from a PDB file, providing flexibility in input formats.

    get_mapping_item(self):
        Returns the MappingItem instance created during the initialization,
        which contains all the detailed information about the mapping from the FG to the CG model.
        This method allows for easy access to the resulting CG structure for further analysis or simulation.
    """

    def __init__(self, CG_num_bead, CGmodel='MARTINI2', mol_name=None, mol_form='sml', smiles=None, pdb_file=None, output_dir='./Mapping'):
        """
        Initializes the object with the necessary information for mapping, including the target coarse-grained model and the molecule's representation (SMILES or PDB).
        """

        self.CGmodel = CGmodel
        self.smiles = smiles
        self.CG_num_bead = CG_num_bead
        self.output_dir = output_dir
        self.mol_json = None
        self.mol_name = mol_name

        if not mol_form == 'sml':
            self.smiles = Chem.MolToSmiles(AllChem.MolFromPDBFile(pdb_file))
        else:
            assert self.smiles, 'Please provide SMILES of molecule'

        if self.CGmodel == 'MARTINI2':
            # Specific implementation for converting using the MARTINI2 model.
            from DSGPM_TP_MARTINI2.CGPredictionFromDSGPM_TP import DSGPM_TPtoCG
            self.mol_json = DSGPM_TPtoCG(smiles=self.smiles, file_dir=self.output_dir, num_cg_bead=self.CG_num_bead)

        self.mapping_item = MappingItem(smiles=self.smiles, mol_name=mol_name, cg_groups=self.mol_json['cgnodes'], cgmodel=self.CGmodel,
                                        cg_group_type=[self.mol_json['cgnode_types'][i[0]] for i in self.mol_json['cgnodes']],
                                        atom_element=[i['element'] for i in self.mol_json['nodes']])

        self.mapping_item.check_cg_groups_id_type_match()
        self.mapping_item.compute_cg_group_mass()
        self.mapping_item.compute_cg_bond(self.mol_json['edges'])
        self.mapping_item.compute_cg_angle()
        self.mapping_item.compute_cg_dihedral()
        # self.mapping_item.save_atom_coords_file()

    def get_mapping_item(self):
        """
        Retrieves the detailed mapping information encapsulated in the MappingItem object, ready for use in simulations or further analysis.
        """
        return self.mapping_item

def main():
    mol_form = 'sml'
    # smile = 'CCCCCCCCCCCCOCCOCCOCCOCCO'
    # num_bead = 8
    # smile = 'CCCCCCCCOC1C(C(C(C(O1)CO)O)O)O'
    # num_bead = 5
    # mapping_output = './mapping_test'

    # smile = 'CCCCCCCCCCCCCCCO'
    # num_bead = 4
    # mapping_output = './mapping_test/AA'
    #
    # mol_mapping = MappingToCG(CG_num_bead=num_bead, CGmodel='MARTINI2', mol_form=mol_form, smiles=smile, output_dir=mapping_output)
    # mol_mapping.mapping_item.save_atom_coords_file('mapping_test/AA/12OH.pdb')
    #
    # # print('yes')


if __name__ == "__main__":
    main()