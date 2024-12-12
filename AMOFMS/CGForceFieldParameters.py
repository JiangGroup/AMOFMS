from FGMappingToCG import MappingToCGfromDSGPM_TP
from tools.lj_query import extract_lj_params, extract_bond_params, extract_angle_params, extract_dihedral_params
from CGPropertiesFromFGSimulation import ComputeCGPropertiesFromFGSimulation
import itertools


class InitCGForceFieldParameters:
    """
    Initializes and calculates the Coarse-Grained (CG) force field parameters
    based on the mapping from a Fine-Grained (FG) simulation.

    This class processes the mapping from FG to CG representation and calculates
    necessary force field parameters including Lennard-Jones (LJ) parameters,
    bond lengths and strengths, angle and dihedral parameters. It supports
    setting ring constraints and customizing the force field parameter calculation
    for different CG models.

    Parameters
    ----------
    mapping_item : object
        An object containing the mapping information from FG to CG representation.
    num_mols : int
        Number of molecules in the system.
    setbond : bool, optional
        Flag to enable calculation of bond parameters. Default is True.
    setangle : bool, optional
        Flag to enable calculation of angle parameters. Default is True.
    setdihedral : bool, optional
        Flag to enable calculation of dihedral parameters. Default is False.
    set_ring_constraint : bool, optional
        Flag to enable setting ring constraints in the CG model. Default is False.

    Attributes
    ----------
    mapping_item : object
        Stores the provided mapping information.
    mol_name : str
        Name of the molecule.
    cg_model : str
        Coarse-Grained model name.
    num_mols : int
        Number of molecules.
    bond_parameters : dict
        Dictionary storing bond parameters.
    angle_parameters : dict
        Dictionary storing angle parameters.
    dihedral_parameters : dict
        Dictionary storing dihedral parameters.
    lj_parameters : dict, optional
        Dictionary storing Lennard-Jones parameters, initialized upon calling
        `init_lj_parameters`.

    Methods
    -------
    add_fg_groups(fg_groups)
        Assigns Fine-Grained groups to Coarse-Grained beads.
    init_lj_parameters()
        Initializes the Lennard-Jones parameters based on CG model.
    init_bond_parameters()
        Initializes bond parameters for the CG model.
    init_angle_parameters()
        Initializes angle parameters for the CG model.
    init_dihedral_parameters()
        Initializes dihedral parameters for the CG model.
    set_bond_parameters(traj, top, res_name, num_mol, num_atom_with_H, bond=None, group_list=None)
        Sets the bond parameters for the CG model using simulation data.
    set_angle_parameters(traj, top, res_name, num_mol, num_atom_with_H, angle=None, group_list=None)
        Sets the angle parameters for the CG model using simulation data.
    set_dihedral_parameters(traj, top, res_name, num_mol, num_atom_with_H, dihedral=None, group_list=None)
        Sets the dihedral parameters for the CG model using simulation data.
    set_bond_constraint(atom1, atom2, constraint_length, constraint_type=1)
        Sets a bond constraint between two atoms.

    """

    def __init__(self, mapping_item, num_mols, setbond=True, setangle=True, setdihedral=False, set_ring_constraint=False):
        """
        Initializes the CG force field parameter object with mapping and system
        configuration information.
        """
        self.mapping_item = mapping_item
        self.mol_name = mapping_item.mol_name
        self.cg_model = mapping_item.cg_model
        self.cg_groups = mapping_item.cg_groups
        self.cg_group_id = mapping_item.cg_group_id
        self.cg_group_type = mapping_item.cg_group_type
        self.num_cg_bead = mapping_item.num_cg_bead
        self.num_atom = mapping_item.num_atom
        self.atom_element = mapping_item.atom_element
        self.cg_group_mass = mapping_item.cg_group_mass
        self.cg_bond = mapping_item.cg_bond
        self.cg_angle = mapping_item.cg_angle
        self.cg_dihedral = mapping_item.cg_dihedral
        self.cg_charge = mapping_item.cg_charge
        self.atom_cg_group_id = mapping_item.atom_cg_group_id
        self.cg_coord_matrix = mapping_item.cg_coord_matrix
        self.num_mols = num_mols

        self.fg_groups = None

        self.setbond_flag = setbond
        self.setangle_flag = setangle
        self.setdihedral_flag = setdihedral
        self.set_ring_constraint_flag = set_ring_constraint


        self.lj_parameters = None
        self.bond_parameters = {}
        self.init_bond_parameters()

        self.angle_parameters = {}
        self.init_angle_parameters()

        self.dihedral_parameters = {}
        self.init_dihedral_parameters()

        if set_ring_constraint:
            self.bond_constraint_dict = {}
            if self.cg_model == 'MARTINI2':
                for bond in self.bond_parameters.copy().keys():
                    if self.cg_group_type[bond[0]][0] == 'S' and self.cg_group_type[bond[1]][0] == 'S':
                        bond_constraint = self.set_bond_constraint(atom1=bond[0], atom2=bond[1],
                                                                   constraint_length=self.bond_parameters[bond][0],
                                                                   constraint_type=1)
                        del self.bond_parameters[bond]
                        self.bond_constraint_dict.update(bond_constraint)
            elif self.cg_model == 'MARTINI3':
                for bond in self.bond_parameters.copy().keys():
                    if self.cg_group_type[bond[0]][0] == 'S' and self.cg_group_type[bond[1]][0] == 'S':
                        bond_constraint = self.set_bond_constraint(atom1=bond[0], atom2=bond[1],
                                                                   constraint_length=self.bond_parameters[bond][0],
                                                                   constraint_type=1)
                        del self.bond_parameters[bond]
                        self.bond_constraint_dict.update(bond_constraint)

        self.cg_ff_parameters_item = {}
        self.generate_cg_ff_parameters_item()

    def add_fg_groups(self, fg_groups):
        """
        Assigns Fine-Grained (FG) groups to Coarse-Grained (CG) beads.

        Parameters
        ----------
        fg_groups : list
            A list of FG groups to be assigned to CG beads.

        """
        assert len(fg_groups) == self.num_cg_bead
        self.fg_groups = fg_groups

    def init_lj_parameters(self):
        """
        Initializes the Lennard-Jones (LJ) parameters for the CG model.

        This method calculates the LJ parameters based on the types of CG
        particles defined in the mapping information.
        """
        cg_types = set(self.cg_group_type)
        lj_pairs = list(itertools.combinations(cg_types, 2))
        for i in cg_types:
            lj_pairs.append((i, i))
        lj_para_dict = {}

        for i in lj_pairs:
            lj_para_dict[i] = list(extract_lj_params(particle_pair=i, cgmodel=self.cg_model))

        self.lj_parameters = lj_para_dict

    def init_bond_parameters(self):
        """
        Initializes the bond parameters for the CG model.

        This method computes the bond lengths and force constants for all bonds
        defined in the CG model.
        """
        bond_para_dict = {}
        for i in self.cg_bond:
            bond_para_dict.update({(i[0], i[1]): list(extract_bond_params(bond=(self.cg_group_type[i[0]], self.cg_group_type[i[1]]), cgmodel=self.cg_model))})
        self.bond_parameters = bond_para_dict

    def init_angle_parameters(self):
        """
        Initializes the angle parameters for the CG model.

        This method calculates the equilibrium angles and their force constants
        for all angle terms defined in the CG model.
        """
        angle_para_dict = {}
        for i in self.cg_angle:
            angle_para_dict.update({(i[0], i[1], i[2]): list(extract_angle_params(angle=(self.cg_group_type[i[0]], self.cg_group_type[i[1]], self.cg_group_type[i[2]]), cgmodel=self.cg_model))})
        self.angle_parameters = angle_para_dict

    def init_dihedral_parameters(self):
        """
        Initializes the dihedral parameters for the CG model.

        This method computes the parameters for dihedral angles, including the
        phase, multiplicity, and force constants.
        """
        dihedral_para_dict = {}
        for i in self.cg_dihedral:
            dihedral_para_dict.update({(i[0], i[1], i[2], i[3]): list(extract_dihedral_params(dihedral=(self.cg_group_type[i[0]], self.cg_group_type[i[1]], self.cg_group_type[i[2]], self.cg_group_type[i[3]]), cgmodel=self.cg_model))})
        self.dihedral_parameters = dihedral_para_dict

    def set_bond_parameters(self, traj, top, res_name, num_mol, num_atom_with_H, bond=None, group_list=None):
        """
        Sets the bond parameters using simulation data.

        Parameters
        ----------
        traj : str
            Path to the trajectory file.
        top : str
            Path to the topology file.
        res_name : str
            Name of the residue.
        num_mol : int
            Number of molecules.
        num_atom_with_H : int
            Number of atoms with hydrogen.
        bond : tuple, optional
            Specific bond for which parameters are set. Default is None.
        group_list : list, optional
            List of groups to consider. Default is None.

        """

        assert self.setbond_flag, \
            "Bond parameters have not been used. Please set setbond=True first."

        analyzer = ComputeCGPropertiesFromFGSimulation(topology=top, trajectory=traj)
        if group_list is None:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=self.cg_groups)
        else:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=group_list)


        if bond:
            avg_bond, std_bond, _ = analyzer.compute_cg_bond_distribution(group1=bond[0], group2=bond[1])
            eq_bond = avg_bond
            # k_bond = constants.Avogadro * constants.Boltzmann * temperature / std_bond ** 2 / 1000
            self.bond_parameters.update({tuple(bond): [round(eq_bond, 4), self.bond_parameters[tuple(bond)][1]]})
        else:
            # default for all cg bond
            for i in self.cg_bond:
                k_bond = self.bond_parameters[tuple(i)][1]
                avg_bond, std_bond, _ = analyzer.compute_cg_bond_distribution(group1=i[0], group2=i[1])
                eq_bond = avg_bond
                # k_bond = constants.Avogadro * constants.Boltzmann * temperature / std_bond ** 2 / 1000
                self.bond_parameters.update({(i[0], i[1]): [round(eq_bond, 4), k_bond]})

    def set_angle_parameters(self, traj, top, res_name, num_mol, num_atom_with_H, angle=None, group_list=None):
        """
        Sets the angle parameters using simulation data.

        Parameters are similar to set_bond_parameters method.
        """

        assert self.setangle_flag, \
            "Angle parameters have not been used. Please set setangle=True first."

        analyzer = ComputeCGPropertiesFromFGSimulation(topology=top, trajectory=traj)
        if group_list is None:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=self.cg_groups)
        else:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=group_list)


        if angle:
            avg_angle, std_angle, _ = analyzer.compute_cg_angle_distribution(group1=angle[0], group2=angle[1], group3=angle[2])
            eq_angle = avg_angle
            self.angle_parameters.update({tuple(angle): [round(eq_angle, 4), self.angle_parameters[tuple(angle)][1]]})
        else:
            # default for all cg angle
            for i in self.cg_angle:
                k_angle = self.angle_parameters[tuple(i)][1]
                avg_angle, std_angle, _ = analyzer.compute_cg_angle_distribution(group1=i[0], group2=i[1], group3=i[2])
                eq_angle = avg_angle
                self.angle_parameters.update({(i[0], i[1], i[2]): [round(eq_angle, 4), k_angle]})

    def set_dihedral_parameters(self, traj, top, res_name, num_mol, num_atom_with_H, dihedral=None, group_list=None):
        """
        Sets the dihedral parameters using simulation data.

        Parameters are similar to set_bond_parameters method.
        """

        assert self.setdihedral_flag, \
            "Dihedral parameters have not been used. Please set setdihedral=True first."

        analyzer = ComputeCGPropertiesFromFGSimulation(topology=top, trajectory=traj)
        if group_list is None:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=self.cg_groups)
        else:
            analyzer.set_cg_groups(atom_num=num_atom_with_H, mol_num=num_mol, residue_name=res_name,
                                   cg_groups=group_list)

        if dihedral:
            avg_dihedral, std_dihedral, _ = analyzer.compute_cg_dihedral_distribution(group1=dihedral[0], group2=dihedral[1], group3=dihedral[2], group4=dihedral[3])
            eq_dihedral = avg_dihedral
            self.dihedral_parameters.update({tuple(dihedral):
                                                 [round(eq_dihedral, 4), self.dihedral_parameters[tuple(dihedral)][1]]})
        else:
            for i in self.cg_dihedral:
                k_dihedral = self.dihedral_parameters[tuple(i)][1]
                avg_dihedral, std_dihedral, _ = analyzer.compute_cg_dihedral_distribution(group1=i[0], group2=i[1], group3=i[2], group4=i[3])
                eq_dihedral = avg_dihedral
                self.dihedral_parameters.update({(i[0], i[1], i[2], i[3]): [round(eq_dihedral, 4), k_dihedral]})

    def set_bond_constraint(self, atom1, atom2, constraint_length, constraint_type=1):
        """
        Sets a bond constraint between two atoms.

        Parameters
        ----------
        atom1 : int
            ID of the first atom.
        atom2 : int
            ID of the second atom.
        constraint_length : float
            Length of the constraint.
        constraint_type : int, optional
            Type of the constraint. Default is 1.

        Returns
        -------
        dict
            A dictionary with the bond constraint parameters.

        """
        return {(atom1, atom2): [constraint_type, constraint_length]}

    def generate_cg_ff_parameters_item(self):
        """
        Generates the complete set of CG force field parameters.

        This method compiles all calculated parameters into a dictionary
        for easy access and manipulation.

        Returns
        -------
        dict
            A dictionary containing all CG force field parameters.

        """
        cg_ff_parameters = {}
        cg_ff_parameters.update({'mol_name': self.mol_name})
        cg_ff_parameters.update({'model': self.cg_model})
        cg_ff_parameters.update({'types': self.cg_group_type})
        cg_ff_parameters.update({'id': self.cg_group_id})
        cg_ff_parameters.update({'charge': self.cg_charge})
        cg_ff_parameters.update({'mass': self.cg_group_mass})
        cg_ff_parameters.update({'num_mols': self.num_mols})
        # cg_ff_parameters.update({'lj_parameters': self.lj_parameters})

        cg_ff_parameters.update({'fg_groups': self.fg_groups})

        if self.setbond_flag:
            cg_ff_parameters.update({'bond_parameters': self.bond_parameters})
        if self.setangle_flag:
            cg_ff_parameters.update({'angle_parameters': self.angle_parameters})
        if self.setdihedral_flag:
            cg_ff_parameters.update({'dihedral_parameters': self.dihedral_parameters})
        if self.set_ring_constraint_flag:
            cg_ff_parameters.update({'bond_constraint': self.bond_constraint_dict})

        self.cg_ff_parameters_item = cg_ff_parameters

        return cg_ff_parameters

def generate_system_top(mols, num_mols=None):
    """
    Generates the system topology for a Coarse-Grained model.

    Parameters
    ----------
    mols : list of dicts
        List of molecule information dictionaries.
    num_mols : list of int, optional
        List specifying the number of each molecule type.

    Returns
    -------
    dict
        A dictionary representing the system topology, including molecules and
        Lennard-Jones cross terms.

    """
    system_top = {}
    if num_mols is not None:
        for idx, i in enumerate(mols):
            i.update({'num_mols': num_mols[idx]})
    system_top['molecules'] = mols
    lj_cross_terms = {}
    all_types = set()
    for i in mols:
        all_types |= set(i['types'])

    lj_pairs = list(itertools.combinations(all_types, 2))
    for i in all_types:
        lj_pairs.append((i, i))

    for i in lj_pairs:
        lj_cross_terms[i] = list(extract_lj_params(particle_pair=i, cgmodel=mols[0]['model']))

    system_top['lj_cross_terms'] = lj_cross_terms

    system_top['cgmodel'] = mols[0]['model']
    return system_top


def main():
    pass
    # mol_A = {'mol_name': 'A', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
    #          'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
    #          'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
    #          'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
    #          'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}
    #
    # mol_B = {'mol_name': 'B', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
    #          'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
    #          'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
    #          'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
    #          'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}
    #
    # system_top = generate_system_top(mols=[mol_A, mol_B], num_mols=[2, 3])

    # mol_form = 'sml'
    # # smile = 'CCCCCCCCCCCCOCCOCCOCCOCCO'
    # # num_bead = 8
    # smile = 'CCCCCCCCCCCO'
    # num_bead = 3
    # mapping_output = './mapping_test'
    #
    # mol_mapping = MappingToCGfromDSGPM_TP(CG_num_bead=num_bead, CGmodel='MARTINI2', mol_form=mol_form, smiles=smile,
    #                           output_dir=mapping_output)
    #
    # traj = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq_whole.trr'
    # top = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq.tpr'
    #
    # FF_parameters_item = InitCGForceFieldParameters(mapping_item=mol_mapping.get_mapping_item())
    # # FF_parameters_item.set_bond_parameters(traj=traj, top=top, num_mol=100, num_atom_with_H=36, res_name='12oh')
    # FF_parameters_item.set_angle_parameters(traj=traj, top=top, num_mol=100, num_atom_with_H=36, res_name='12oh')
    #
    # print('yes')

if __name__ == "__main__":
    main()