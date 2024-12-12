import MDAnalysis as mda
import numpy as np
import MDAnalysis.analysis.distances as distances
import matplotlib.pyplot as plt
from tools.math_tools import compute_angle, compute_dihedral
from MDAnalysis.exceptions import NoDataError
import logging


logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


# gmx trjconv -f eq.trr -s eq.tpr -o eq_whole.trr  -pbc whole -force yes ;  makes broken molecules whole in the boundary.


class ComputeCGPropertiesFromFGSimulation:
    """
    A class for computing Coarse-Grained (CG) properties from Fine-Grained (FG) simulation data.

    Attributes:
    -----------
    topology : str
        Path to the topology file.
    trajectory : str
        Path to the trajectory file.
    cg_groups : dict, optional
        Definition of CG groups.
    atom_num : int, optional
        Number of atoms per molecule.
    mol_num : int, optional
        Number of molecules.
    residue_name : str, optional
        Name of the residue to select atoms from.
    universe : MDAnalysis.Universe
        The Universe object containing loaded simulation data.

    Methods:
    --------
    __init__(self, topology, trajectory)
        Initializes the class with topology and trajectory files.
    set_cg_groups(self, cg_groups, atom_num, mol_num, residue_name)
        Sets CG groups and related properties.
    extract_atoms_by_residue(self)
        Extracts atoms based on the residue name.
    compute_centroid(self, atom_group)
        Computes the centroid of a given atom group.
    compute_cg_bond_distribution(self, group1, group2)
        Computes the distribution of distances between centroids of two CG groups.
    compute_cg_angle_distribution(self, group1, group2, group3)
        Computes the distribution of angles formed by centroids of three CG groups.
    compute_cg_dihedral_distribution(self, group1, group2, group3, group4)
        Computes the distribution of dihedrals formed by centroids of four CG groups.
    plot_distribution(self, data, title, xlabel, ylabel)
        Plots the distribution of a given data set.
    compute_fg_group_force(self, fg_groups)
        Computes forces on FG groups mapped to CG groups.
    """
    def __init__(self, topology, trajectory, begin_frame_id=None, end_frame_id=None, skip_frames=None):
        """
        Constructor to initialize the ComputeCGPropertiesFromFGSimultion class.

        Parameters:
        -----------
        topology : str
            Path to the topology file.
        trajectory : str
            Path to the trajectory file.
        """
        self.topology = topology
        self.trajectory = trajectory
        self.cg_groups = None
        self.atom_num = None
        self.mol_num = None
        self.residue_name = None

        self.universe = mda.Universe(topology, trajectory)

        self.begin_frame_id = 0 if begin_frame_id is None else begin_frame_id
        self.end_frame_id = len(self.universe.trajectory) if end_frame_id is None else end_frame_id
        self.skip_frames = 1 if skip_frames is None else skip_frames

        try:
            forces = self.universe.trajectory[0].forces
            # print("Forces available.")
        except NoDataError:
            print("No force data available in the trajectory.")

    def set_cg_groups(self, cg_groups, atom_num, mol_num, residue_name):
        """
        Sets the CG groups and related properties for analysis.

        Parameters:
        -----------
        cg_groups : dict
            Definition of CG groups.
        atom_num : int
            Number of atoms per molecule.
        mol_num : int
            Number of molecules.
        residue_name : str
            Name of the residue to select atoms from.
        """
        self.cg_groups = cg_groups
        self.atom_num = atom_num
        self.mol_num = mol_num
        self.residue_name = residue_name

    def extract_atoms_by_residue(self):
        """
        Extracts atoms by residue name from the loaded simulation data.

        This method selects atoms based on the specified residue name attribute,
        facilitating further analysis such as computing centroids or distributions.

        Returns:
        --------
        MDAnalysis.core.groups.AtomGroup
            An AtomGroup containing all atoms with the specified residue name.
        """
        return self.universe.select_atoms(f"resname {self.residue_name}")

    def compute_centroid(self, atom_group, method='com'):
        """
        Computes the centroid (center of mass or center of geometry) of a given atom group.

        Parameters:
        -----------
        atom_group : MDAnalysis.core.groups.AtomGroup
            The atom group for which to compute the centroid.
        method : str, optional, default 'com'
            The method to use for computing the centroid. 'com' for center of mass,
            'cog' for center of geometry.

        Returns:
        --------
        np.ndarray
            A 3-element array containing the x, y, z coordinates of the centroid.
        """
        if method == 'com':
            return atom_group.center_of_mass()
        elif method == 'cog':
            return atom_group.center_of_geometry()
        else:
            raise ValueError("Invalid method. Choose 'com' for center of mass or 'cog' for center of geometry.")

    def compute_cg_bond_distribution(self, group1, group2, center='com'):
        """
        Computes the distribution of distances between centroids of two CG groups across the trajectory.

        Parameters:
        -----------
        group1 : int
            Index of the first CG group.
        group2 : int
            Index of the second CG group.

        Returns:
        --------
        tuple
            A tuple containing the mean distance, standard deviation, and a list of distances for each frame.
        """
        centroid_distances = []

        # selection for specified residue
        specific_atoms = self.extract_atoms_by_residue()
        offset = specific_atoms[1].id
        self.distance_all = []

        for ts in self.universe.trajectory[self.begin_frame_id:self.end_frame_id:self.skip_frames]:
            distances_per_frame = []

            for mol_id in range(self.mol_num):
                group1_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group1]]
                group2_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group2]]
                group1_atoms = specific_atoms.select_atoms('bynum ' + ' '.join(map(str, group1_indices)))
                group2_atoms = specific_atoms.select_atoms('bynum ' + ' '.join(map(str, group2_indices)))

                # compute com
                centroid1 = self.compute_centroid(group1_atoms, method=center)
                centroid2 = self.compute_centroid(group2_atoms, method=center)

                # compute distance between com under pbc
                distance = distances.distance_array(centroid1[None, :], centroid2[None, :])[0][0]/10
                distances_per_frame.append(distance)
                self.distance_all.append(distance)

            centroid_distances.append(np.mean(distances_per_frame))  # average

        return np.mean(centroid_distances), np.std(centroid_distances, ddof=0), centroid_distances

    def compute_cg_angle_distribution(self, group1, group2, group3, center='com'):
        """
        Computes the distribution of angles formed by centroids of three CG groups.

        Parameters:
        -----------
        group1 : int
            Index of the first CG group.
        group2 : int
            Index of the second CG group.
        group3 : int
            Index of the third CG group.

        Returns:
        --------
        tuple
            A tuple containing the mean angle, standard deviation, and a list of angles for each frame.
        """
        angle_values = []

        specific_atoms = self.extract_atoms_by_residue()
        offset = specific_atoms[1].id

        self.angle_all = []

        for ts in self.universe.trajectory[self.begin_frame_id:self.end_frame_id:self.skip_frames]:
            angles_per_frame = []

            for mol_id in range(self.mol_num):
                g1_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group1]]
                g2_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group2]]
                g3_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group3]]

                # compute com
                centroid1 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g1_indices))), method=center)
                centroid2 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g2_indices))), method=center)
                centroid3 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g3_indices))), method=center)

                # compute angle
                angle = compute_angle(centroid1, centroid2, centroid3)
                angles_per_frame.append(angle)
                self.angle_all.append(angle)

            angle_values.append(np.mean(angles_per_frame))

        return np.mean(angle_values), np.std(angle_values, ddof=0), angle_values

    def compute_cg_dihedral_distribution(self, group1, group2, group3, group4, center='com'):
        """
        Computes the distribution of dihedrals formed by centroids of four CG groups.

        Parameters:
        -----------
        group1 : int
            Index of the first CG group.
        group2 : int
            Index of the second CG group.
        group3 : int
            Index of the third CG group.
        group4 : int
            Index of the fourth CG group.

        Returns:
        --------
        tuple
            A tuple containing the mean dihedral angle, standard deviation, and a list of dihedral angles for each frame.
        """
        dihedral_values = []

        # selection for specified residue
        specific_atoms = self.extract_atoms_by_residue()
        offset = specific_atoms[1].id
        self.dihedral_all = []

        for ts in self.universe.trajectory[self.begin_frame_id:self.end_frame_id:self.skip_frames]:
            dihedrals_per_frame = []

            for mol_id in range(self.mol_num):
                g1_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group1]]
                g2_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group2]]
                g3_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group3]]
                g4_indices = [idx + mol_id * self.atom_num + offset for idx in self.cg_groups[group4]]

                # compute com
                centroid1 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g1_indices))), method=center)
                centroid2 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g2_indices))), method=center)
                centroid3 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g3_indices))), method=center)
                centroid4 = self.compute_centroid(
                    specific_atoms.select_atoms('bynum ' + ' '.join(map(str, g4_indices))), method=center)

                # compute dihedral
                dihedral = compute_dihedral(centroid1, centroid2, centroid3, centroid4)
                dihedrals_per_frame.append(dihedral)
                self.dihedral_all.append(dihedral)

            dihedral_values.append(np.mean(dihedrals_per_frame))

        return np.mean(dihedral_values), np.std(dihedral_values, ddof=0), dihedral_values

    def plot_distribution(self, data, title, xlabel, ylabel):
        """
        Plots the distribution of a given data set using matplotlib.

        Parameters:
        -----------
        data : list or np.ndarray
            The data set to plot.
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        """
        plt.hist(data, bins=50)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def compute_fg_group_force(self, fg_groups):
        """
        Computes forces on FG groups mapped to CG groups.

        Parameters:
        -----------
        fg_groups : list
            A list of FG groups for which to compute forces.

        Returns:
        --------
        np.ndarray
            A numpy array containing the forces for each FG group across the trajectory.
        """
        assert len(fg_groups) == len(self.cg_groups)

        forces_array = np.zeros((len(self.universe.trajectory), len(fg_groups) * self.mol_num, 3))  # 3D vector array

        for ts_idx, ts in enumerate(self.universe.trajectory):

            # selection for specified residue
            specific_atoms = self.extract_atoms_by_residue()
            offset = specific_atoms[1].id

            group_idx_in_system = 0
            for mol_id in range(self.mol_num):
                for group_id in fg_groups:
                    group_force = np.zeros(3)  # 3D vector array
                    for idx in group_id:
                        fg_indices = idx + mol_id * self.atom_num + offset
                        atom = specific_atoms.select_atoms('bynum ' + str(fg_indices))
                        group_force += atom.forces.sum(axis=0)

                    forces_array[ts_idx, group_idx_in_system] = group_force
                    group_idx_in_system += 1
        return forces_array


class ComputeCGPropertiesFromFGSimulation_All:
    """
    Computes Coarse-Grained (CG) properties from Fine-Grained (FG) simulation data for an entire system.

    This class is designed to handle multiple molecule types and calculate CG properties,
    such as centroids and forces, for all molecules present in the simulation. It supports
    operations on the entire system, making it suitable for analyses that require a global
    perspective of the simulation data.

    Attributes:
    -----------
    topology : str
        Path to the topology file.
    trajectory : str
        Path to the trajectory file.
    system_top : dict
        A dictionary representing the system's topology, including information about molecules.
    molecules : list
        A list containing dictionaries for each molecule type in the system.
    universe : MDAnalysis.Universe
        The Universe object containing loaded simulation data.

    Methods:
    --------
    __init__(self, topology, trajectory, system_top)
        Constructor to initialize the class with simulation and system topology data.
    get_num_frames(self)
        Returns the number of frames in the trajectory.
    compute_centroid(self, atom_group)
        Computes the centroid (center of mass) for a given atom group.
    compute_all_fg_group_force(self, fg_resname_list=None, frame_id=0)
        Computes forces for all FG groups in the system at a specified frame.
    save_cg_coord_from_fg(self, save_file='cg.gro', fg_resname_list=None, save_molecule_list=None, frame_id=-1, method='com', cg_wrap=False)
        Saves the CG coordinates derived from FG data to a file.
    """
    def __init__(self, topology, trajectory, system_top, begin_frame_id=None, end_frame_id=None, skip_frames=None):
        """
        Initializes the ComputeCGPropertiesFromFGSimultion_All class with system topology and trajectory data.

        Parameters:
        -----------
        topology : str
            Path to the topology file.
        trajectory : str
            Path to the trajectory file.
        system_top : dict
            System topology information, including molecules and their properties.
        """
        self.topology = topology
        self.trajectory = trajectory
        self.system_top = system_top
        self.molecules = system_top['molecules']  # Extract molecule information from system topology
        self.universe = mda.Universe(topology, trajectory)  # Create a Universe object for analysis

        self.begin_frame_id = 0 if begin_frame_id is None else begin_frame_id
        self.end_frame_id = len(self.universe.trajectory) if end_frame_id is None else end_frame_id
        self.skip_frames = 1 if skip_frames is None else skip_frames

        # Attempt to access force data, with a fallback if not available
        try:
            forces = self.universe.trajectory[0].forces
            # print("Forces available.")
        except NoDataError:
            print("No force data available in the trajectory.")

    def get_num_frames(self):
        """
        Returns the total number of frames in the trajectory.

        Returns:
        --------
        int
            The number of frames in the trajectory.
        """
        return self.universe.trajectory.n_frames

    def compute_centroid(self, atom_group, method='com'):
        """
        Computes the centroid (center of mass or center of geometry) of a given atom group.

        Parameters:
        -----------
        atom_group : MDAnalysis.core.groups.AtomGroup
            The atom group for which to compute the centroid.
        method : str, optional, default 'com'
            The method to use for computing the centroid. 'com' for center of mass,
            'cog' for center of geometry.

        Returns:
        --------
        np.ndarray
            A 3-element array containing the x, y, z coordinates of the centroid.
        """
        if method == 'com':
            return atom_group.center_of_mass()
        elif method == 'cog':
            return atom_group.center_of_geometry()
        else:
            raise ValueError("Invalid method. Choose 'com' for center of mass or 'cog' for center of geometry.")

    def compute_all_fg_group_force(self, fg_resname_list=None, frame_id=0):
        """
        Computes the forces acting on all FG groups within the system for a specified frame.

        Parameters:
        -----------
        fg_resname_list : list of str, optional
            A list of residue names corresponding to the FG groups. If None, the molecule names from system_top are used.
        frame_id : int, optional
            The frame index at which to compute the forces. Default is 0.

        Returns:
        --------
        dict
            A dictionary with molecule names as keys and numpy arrays of forces as values.
        """
        # Select the specified frame for analysis
        self.universe.trajectory[frame_id]

        all_fg_group_force = {}  # Initialize a dictionary to store force data

        # Loop through each molecule type in the system
        for idxs, i in enumerate(self.molecules):
            fg_groups = i['fg_groups']  # Fine-Grained groups within the molecule
            mol_num = i['num_mols']  # Number of molecules of this type
            atom_num = sum(len(row) for row in fg_groups)  # Total number of atoms across all FG groups in a molecule

            # Residue name for this molecule type
            if fg_resname_list is None:
                residue_name = i['mol_name']
            else:
                residue_name = fg_resname_list[idxs]
            forces_array = np.zeros((len(fg_groups) * mol_num, 3))   # Initialize an array to store forces for each FG group

            # selection for specified residue
            specific_atoms = self.universe.select_atoms(f"resname {residue_name}")
            offset = specific_atoms[1].id

            group_idx_in_system = 0
            # Iterate over each molecule instance
            for mol_id in range(mol_num):
                # Iterate over each FG group within the molecule
                for group in fg_groups:
                    group_force = np.zeros(3)  # Initialize a zero force vector for this group

                    # Compute the force on each atom in the group and sum them
                    for idx in group:
                        fg_indices = idx + mol_id * atom_num + offset
                        atom = specific_atoms.select_atoms('bynum ' + str(fg_indices))
                        group_force += atom.forces.sum(axis=0)

                    # Store the computed group force
                    forces_array[group_idx_in_system] = group_force
                    group_idx_in_system += 1

            all_fg_group_force.update({i['mol_name']: forces_array})

        return all_fg_group_force

    def save_cg_coord_from_fg(self, save_file='cg.gro', fg_resname_list=None, save_molecule_list=None, frame_id=-1, method='com', cg_wrap=False):
        """
        Saves CG coordinates, derived from FG simulation data, to a specified file.

        Parameters:
        -----------
        save_file : str, optional
            The name of the file to save the CG coordinates to. Default is 'cg.gro'.
        fg_resname_list : list of str, optional
            A list of residue names to use instead of molecule names. Default is None.
        save_molecule_list : list of int, optional
            A list of indices specifying which molecules to save. Default is None (save all).
        frame_id : int, optional
            The frame index from which to save the CG coordinates. Default is -1 (last frame).
        method : str, optional
            The method to compute CG coordinates ('com' for center of mass, 'cog' for center of geometry). Default is 'com'.
        cg_wrap : bool, optional
            Whether to wrap the coordinates back into the simulation box. Default is False.

        """

        assert method in ['com', 'cog'], "Method must be 'com' or 'cog'"

        # Select the specified frame for analysis
        self.universe.trajectory[frame_id]

        cg_coords = []  # Initialize a list to store CG coordinates
        mol_id_list, mol_name_list, atom_name_list = [], [], []  # Lists to store molecule and atom identifiers
        mol_id_offset = 0  # Initialize an offset for molecule IDs

        # Determine the subset of molecules to save, if specified
        molecules = self.molecules
        if save_molecule_list is not None:
            molecules = [self.molecules[i] for i in save_molecule_list]

        # Loop through each molecule type to compute and save CG coordinates
        for idx, i in enumerate(molecules):
            fg_groups = i['fg_groups']  # FG groups for the current molecule type
            mol_num = i['num_mols']  # Number of molecules of this type in the system
            atom_num = sum(len(row) for row in fg_groups)  # Total number of atoms in a single molecule of this type

            # Use specified residue name if provided
            if fg_resname_list is None:
                residue_name = i['mol_name']
            else:
                residue_name = fg_resname_list[idx]
            cg_num = len(i['types'])

            assert cg_num == len(fg_groups)

            # Prepare lists for molecule IDs, names, and atom names for CG representation
            for n in range(mol_num):
                mol_id_list += [n + 1 + mol_id_offset] * cg_num
                mol_name_list += [residue_name] * cg_num
                atom_name_list += i['types']

            mol_id_offset += mol_num  # Update molecule ID offset for the next molecule type

            # selection for specified residue
            specific_atoms = self.universe.select_atoms(f"resname {residue_name}")
            offset = specific_atoms[1].id

            # Compute CG coordinates for each molecule
            for mol_id in range(mol_num):
                for group in fg_groups:
                    # Select atoms in the current FG group for the current molecule instance
                    group_indices = [idxs + mol_id * atom_num + offset for idxs in group]
                    group_atoms = specific_atoms.select_atoms('bynum ' + ' '.join(map(str, group_indices)))

                    # Compute the CG coordinate (center of mass or geometry) for the current group
                    if method == 'com':
                        # center of mass
                        cg_coord = group_atoms.center_of_mass()
                    else:
                        # center of geometry
                        cg_coord = group_atoms.centroid()

                    # Optionally wrap the CG coordinate into the simulation box
                    if cg_wrap:
                        # wrapped in simulation box
                        cg_coord = wrap_coordinates(cg_coord, self.universe.dimensions)

                    cg_coords.append(cg_coord)

        # Save CG coordinates to a file
        with open(save_file, 'w') as file:
            # Write header and atom count
            file.write("CG coordinates\n")
            file.write(f"{len(cg_coords)}\n")  # Total number of CG beads

            # Write CG coordinates for each bead
            for i, coord in enumerate(cg_coords):
                id = mol_id_list[i]
                mol = mol_name_list[i][:5].replace(" ", "")
                atom = atom_name_list[i][:5]
                atom_id = i + 1  # count from 1

                file.write(f"{id:5d}{mol:<5s}{atom:<5s}{atom_id:5d}{coord[0]/10:8.3f}{coord[1]/10:8.3f}{coord[2]/10:8.3f}\n")

            # Optionally write simulation box dimensions
            if self.universe.dimensions.size:
                box = self.universe.dimensions[:3] / 10   # Convert to nm
                file.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")

        print(f"CG coordinates saved to {save_file}")

# Note: The method `centroid` used above for method 'cog' is a placeholder and may need implementation
# based on the attributes of the atom group. Typically, the centroid is the arithmetic mean of the atom positions.


def wrap_coordinates(coord, dimensions):
    """
    Wraps coordinates within the boundaries of the simulation box.

    This function ensures that coordinates are wrapped back into the simulation box,
    which is essential for periodic boundary conditions. It is applicable for rectangular
    simulation boxes.

    Parameters:
    -----------
    coord : np.ndarray
        The coordinates to be wrapped. Can be a single set of coordinates or an array of coordinates.
    dimensions : np.ndarray
        The dimensions of the simulation box, typically in the form [Lx, Ly, Lz], where Lx, Ly, and Lz
        are the lengths of the simulation box along the x, y, and z axes, respectively.

    Returns:
    --------
    np.ndarray
        The wrapped coordinates, ensuring all points are within the simulation box boundaries.
    """
    box_size = dimensions[:3]  # Extract the lengths of the simulation box
    return np.mod(coord, box_size)  # Use modulo operation to wrap coordinates


def extrat_cg_force_all(topology, trajectory, system_top, frame_id=0):
    """
    Extracts the forces acting on all Coarse-Grained (CG) groups within the system for a specified frame.

    This function iterates over all molecules described in the system topology, selecting specific residues
    based on the provided names, and then computes the total force acting on each CG group within those molecules.

    Parameters:
    -----------
    topology : str
        The path to the topology file.
    trajectory : str
        The path to the trajectory file.
    system_top : dict
        A dictionary representing the system's topology, including molecules and their properties.
    frame_id : int, optional
        The frame index at which to compute the forces. Default is 0.

    Returns:
    --------
    dict
        A dictionary with molecule names as keys and numpy arrays of forces (for each CG group in those molecules) as values.
    """
    universe = mda.Universe(topology, trajectory)  # Load the simulation data
    universe.trajectory[frame_id]  # Set the trajectory frame for analysis

    all_cg_force = {}  # Initialize a dictionary to store the forces on CG groups
    molecules = system_top['molecules']

    # Attempt to access force data, with a fallback if not available
    try:
        forces = universe.trajectory[0].forces
        # print("Forces available.")
    except NoDataError:
        print("No force data available in the trajectory.")
        return {}

    # Loop over each molecule defined in the system topology
    for i in molecules:
        mol_num = i['num_mols']  # Number of molecules of this type
        residue_name = i['mol_name']  # Residue name for this molecule type
        cg_ids = i['id']  # CG group identifiers
        cg_num = len(cg_ids)  # Number of CG groups
        forces_array = np.zeros((cg_num * mol_num, 3))  # Initialize an array to hold forces for each CG group

        # Select atoms of the specified residue
        specific_atoms = universe.select_atoms(f"resname {residue_name}")

        cg_idx_in_system = 0  # Counter for indexing CG groups across all molecules
        for mol_id in range(mol_num):
            for cg in range(cg_num):
                # Placeholder: Actual force extraction logic should be implemented
                # For example, summing forces on all atoms in each CG group
                forces_array[cg_idx_in_system] = specific_atoms[cg_idx_in_system].force
                cg_idx_in_system += 1

        all_cg_force.update({i['mol_name']: forces_array})  # Update the dictionary with forces for this molecule type

    return all_cg_force


def main():
    pass

    # gmx trjconv -f eq.trr -s eq.tpr -o eq_whole.trr  -pbc whole -force yes ;  makes broken molecules whole in the boundary.
    traj = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq_whole.trr'
    top = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq.tpr'

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



    analyzer = ComputeCGPropertiesFromFGSimulation(topology=top, trajectory=traj)
    analyzer.set_cg_groups(atom_num=36, mol_num=100, cg_groups=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11]], residue_name='12oh')


    distances_per_frame, _, _ = analyzer.compute_cg_bond_distribution(group1=0, group2=1)
    angle = analyzer.compute_cg_angle_distribution(group1=0, group2=1, group3=2)
    dihedral = analyzer.compute_cg_dihedral_distribution(group1=0, group2=1, group3=2, group4=3)
    #
    # # 绘制分布图
    # analyzer.plot_distribution(distances_per_frame, 'Centroid Distance Distribution', 'Distance', 'Frequency')
    # print(np.mean(distances_per_frame))

if __name__ == '__main__':
    main()