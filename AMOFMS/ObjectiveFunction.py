from CGPropertiesFromFGSimulation import ComputeCGPropertiesFromFGSimulation_All, extrat_cg_force_all
from tools.gromacs import generate_top_file, run_gromacs_simulation, unwrap_trajectory
import os
import numpy as np
from tools.utilies import mkdir, delete_files
from tools.properties import RDF


class BottomUpObjectiveFunction:
    """
    Defines the bottom-up approach for optimizing Coarse-Grained (CG) models based on Fine-Grained (FG) simulation data.

    This class is responsible for managing the overall process of converting FG simulation data into a CG model,
    running CG simulations, and evaluating the performance of the CG model through various objective functions.

    Attributes:
    -----------
    system_top : dict
        The system topology information derived from the FG simulation.
    molecules : list
        A list of dictionaries, each representing a molecule in the system.
    molecule_list : list
        A list of molecule names present in the system.
    pdb_file : str, optional
        Path to the PDB file, if available.
    fg_topology : str
        Path to the FG topology file.
    fg_trajectory : str
        Path to the FG trajectory file.
    cg_topology : str, optional
        Path to the generated CG topology file, initialized as None.
    cg_trajectory : str, optional
        Path to the CG trajectory file, initialized as None.
    opt_folder : str
        Path to the optimization folder where results and intermediate files are stored.
    cg_top_folder : str
        Path to the folder containing the CG topology files.
    cg_top_file : str
        Path to the CG topology file.

    Methods:
    --------
    __init__(self, system_top, fg_topology, fg_trajectory, opt_folder, cg_top_file_name='cg.top', pdb_file=None)
        Initializes the class with system topology, FG simulation data, and optimization settings.
    update_system_topology(self, new_system_top)
        Updates the system topology information.
    update_opt_folder(self, new_opt_folder)
        Updates the optimization folder path and related settings.
    force_match_loss(self, cg_mdp_file, fg_resname_list=None, begin_frame=None, end_frame=None, skip_frame=None)
        Calculates the force matching loss between FG and CG simulations.
    run_cg_simulation(self, mdp_folder, initial_gro=None, fg_resname_list=None, index_file=None, cg_simulation_folder=None, table_file=None, gpu_acceleration=True, em_double_version=True, em=True, anneal=True, eq=True, prod=True, nt=8, gpu_id=0)
        Runs the CG simulation based on the provided settings and updates the CG trajectory and topology paths.
    Boltzmann_inversion(self, rdf_pairs_list, tag, max_distance, rdf_folder=None, begin_frame_id=None, end_frame_id=None, skip_frames=None, Temperature=300, bin_width=0.002)
        Performs Boltzmann inversion based on the Radial Distribution Function (RDF) to derive potential functions.
    """

    def __init__(self, system_top, fg_topology, fg_trajectory, opt_folder, cg_top_file_name='cg.top', pdb_file=None):
        """
        Initializes the BottomUpObjectiveFunction class with system topology, FG simulation data, and optimization settings.

        Parameters:
        -----------
        system_top : dict
            The system topology information derived from the FG simulation.
        fg_topology : str
            Path to the FG topology file.
        fg_trajectory : str
            Path to the FG trajectory file.
        opt_folder : str
            Path to the optimization folder.
        cg_top_file_name : str, optional
            Name of the CG topology file, defaults to 'cg.top'.
        pdb_file : str, optional
            Path to the PDB file, if available.
        """
        self.system_top = system_top
        self.molecules = self.system_top['molecules']
        self.molecule_list = [mol['mol_name'] for mol in self.molecules]
        self.pdb_file = pdb_file

        self.fg_topology = fg_topology
        self.fg_trajectory = fg_trajectory

        self.cg_topology = None
        self.cg_trajectory = None

        self.opt_folder = opt_folder
        mkdir(self.opt_folder)
        self.cg_top_folder = os.path.join(self.opt_folder, 'cg_top')
        mkdir(self.cg_top_folder)
        self.cg_top_file = os.path.join(self.cg_top_folder, cg_top_file_name)
        generate_top_file(system_top=self.system_top, save_file=self.cg_top_file, pdb_file=self.pdb_file)

    def update_system_topology(self, new_system_top):
        """
        Updates the system topology with new information and regenerates the CG topology file.

        Parameters:
        -----------
        new_system_top : dict
            The new system topology information.
        """
        self.system_top = new_system_top

    def update_opt_folder(self, new_opt_folder):
        self.opt_folder = new_opt_folder
        mkdir(self.opt_folder)
        self.cg_top_folder = os.path.join(self.opt_folder, 'cg_top')
        self.cg_top_file = os.path.join(self.cg_top_folder, 'cg.top')
        mkdir(self.cg_top_folder)
        generate_top_file(system_top=self.system_top, save_file=self.cg_top_file, pdb_file=self.pdb_file)  # Regenerate the CG topology file

    def force_match_loss(self,  cg_mdp_file, fg_resname_list=None, begin_frame=None, end_frame=None, skip_frame=None):
        """
        Calculates the force matching loss between FG and CG simulations for a range of frames.

        Parameters:
        -----------
        cg_mdp_file : str
            Path to the CG simulation parameter file (MDP file).
        fg_resname_list : list, optional
            List of residue names to include in the force matching process, defaults to None (all residues).
        begin_frame : int, optional
            The beginning frame for the force matching calculation, defaults to None (start from the first frame).
        end_frame : int, optional
            The ending frame for the force matching calculation, defaults to None (end at the last frame).
        skip_frame : int, optional
            The number of frames to skip between evaluations, defaults to None (no skipping).

        Returns:
        --------
        float
            The mean force matching loss across the evaluated frames.
        """

        # Perform computations to extract CG properties from the FG simulation
        fg_computation = ComputeCGPropertiesFromFGSimulation_All(topology=self.fg_topology, trajectory=self.fg_trajectory,
                                                                system_top=self.system_top)
        num_frame = fg_computation.get_num_frames()

        fm_loss_list = []  # Initialize the list to store force matching loss values

        tmp_operation_folder = os.path.join(self.opt_folder, 'force_match')
        mkdir(tmp_operation_folder)  # Create a temporary folder for force matching operations

        # Set default frame range and skipping interval if not specified
        bf, ef, sf = 0, num_frame, 1
        if begin_frame is not None:
            bf = begin_frame
        if end_frame is not None:
            ef = end_frame
        if skip_frame is not None:
            sf = skip_frame

        for i in range(bf, ef, sf):
            print(f'\nprocessing frame {i}')
            fm_loss = 0

            # Save the CG coordinates derived from the current FG frame
            cg_gro_file_from_fg = os.path.join(tmp_operation_folder, 'cg.gro')
            fg_computation.save_cg_coord_from_fg(save_file=cg_gro_file_from_fg, fg_resname_list=fg_resname_list, frame_id=i, method='com')

            # Run the CG simulation for force matching
            task_name = 'force_match'
            run_gromacs_simulation(top_file=self.cg_top_file, gro_file=cg_gro_file_from_fg, mdp_file=cg_mdp_file,
                                   output_folder=tmp_operation_folder, task_name=task_name)

            # Load the CG simulation results for force analysis
            cg_topology = os.path.join(tmp_operation_folder, f'{task_name}.tpr')
            cg_trajectory = os.path.join(tmp_operation_folder, f'{task_name}.trr')

            # Extract CG forces and compute FG group forces for comparison
            cg_forces = extrat_cg_force_all(topology=cg_topology, trajectory=cg_trajectory, system_top=self.system_top, frame_id=0)
            fg_group_forces = fg_computation.compute_all_fg_group_force(frame_id=i, fg_resname_list=fg_resname_list)

            for molecule in self.molecule_list:
                fg_mol_group_forces = fg_group_forces[molecule]
                cg_mol_forces = cg_forces[molecule]

                if fg_mol_group_forces.shape != cg_mol_forces.shape:
                    raise ValueError("The shapes of the force arrays must match.")

                # Calculate the difference in forces
                force_diff = fg_mol_group_forces - cg_mol_forces

                # Calculate the magnitude of the force loss
                force_loss = np.mean(np.linalg.norm(force_diff, axis=1))

                fm_loss += force_loss

                # Save the force comparison data to a file
                fm_mol_file = os.path.join(tmp_operation_folder, f'{i}-th-frame-{molecule}.xvg')
                with open(fm_mol_file, 'w') as file:
                    # Write header comments
                    file.write(f'#    fg_froce     cg_force (kJ/(mol·Å)) \n')
                    for x, y in zip(np.linalg.norm(fg_mol_group_forces, axis=1), np.linalg.norm(cg_mol_forces, axis=1)):
                        # Format the string to retain six decimal places and ensure right alignment
                        file.write(f'{x:>15.6f} {y:>15.6f}\n')

            # Clean up temporary files generated during the force matching process
            delete_files(del_file=os.path.join(tmp_operation_folder, f'{task_name}*'))

            # Store the rounded force matching loss for the current frame
            fm_loss_list.append(np.round(fm_loss, 6))

        # Calculate and return the mean force matching loss across all evaluated frames
        fm_loss_mean = np.mean(fm_loss_list)
        return fm_loss_mean

    def run_cg_simulation(self, mdp_folder, initial_gro=None, fg_resname_list=None, index_file=None, cg_simulation_folder=None,
                          table_file=None, gpu_acceleration=True, em_double_version=True, em=True, anneal=True, eq=True, prod=True, nt=8, gpu_id=0):
        """
        Runs the CG simulation process, including energy minimization, annealing, equilibration, and production phases.

        Parameters:
        -----------
        mdp_folder : str
            Path to the folder containing the MDP files for different simulation stages.
        initial_gro : str, optional
            Path to the initial GRO file for the simulation, if not provided, it will be generated.
        fg_resname_list : list, optional
            List of residue names to include in the CG model, if not provided, all residues will be included.
        index_file : str, optional
            Path to the index file for GROMACS simulation, if required.
        cg_simulation_folder : str, optional
            Path to store the results of the CG simulation, a default folder will be created if not provided.
        table_file : str, optional
            Path to the custom interaction table file for GROMACS, if required.
        gpu_acceleration : bool, optional
            Whether to use GPU acceleration for the simulation, defaults to True.
        em_double_version : bool, optional
            Whether to use double precision for the energy minimization step, defaults to True.
        em : bool, optional
            Whether to perform energy minimization, defaults to True.
        anneal : bool, optional
            Whether to perform annealing, defaults to True.
        eq : bool, optional
            Whether to perform equilibration, defaults to True.
        prod : bool, optional
            Whether to perform production, defaults to True.
        nt : int, optional
            Number of threads to use for the simulation, defaults to 8.
        gpu_id : int, optional
            GPU ID to use for the simulation, defaults to 0.

        Notes:
        ------
        This method sequentially runs the different stages of a GROMACS simulation process for a CG model, updating
        the class attributes with the paths to the final CG topology and trajectory files.
        """

        # Create simulation folder if not provided
        if cg_simulation_folder is None:
            cg_simulation_folder = os.path.join(self.opt_folder, 'simulation')
        mkdir(cg_simulation_folder)

        # Generate initial GRO file from FG data if not provided
        if initial_gro is None:
            initial_gro = os.path.join(cg_simulation_folder, 'initial.gro')
            cg_coord = ComputeCGPropertiesFromFGSimulation_All(topology=self.fg_topology, trajectory=self.fg_trajectory,
                                                              system_top=self.system_top)
            cg_coord.save_cg_coord_from_fg(save_file=initial_gro, fg_resname_list=fg_resname_list, frame_id=-1, method='com')

        tmp_gro = initial_gro  # Temporary variable to hold the current GRO file path

        folder = None
        task_name = None

        # run energy minimization
        if em:
            task_name = 'em'
            folder = os.path.join(cg_simulation_folder, task_name)
            mkdir(folder)
            em_mdp = os.path.join(mdp_folder, 'em.mdp')
            run_gromacs_simulation(top_file=self.cg_top_file, double_version=em_double_version, gro_file=tmp_gro,
                                   index_file=index_file, mdp_file=em_mdp, table_file=table_file, em=True,
                                   task_name=task_name, output_folder=folder, nt=nt, gpu_id=gpu_id)
            tmp_gro = os.path.join(folder, 'em.gro')  # Update GRO file path for the next stage

        #  run annealing process
        if anneal:
            task_name = 'anneal'
            folder = os.path.join(cg_simulation_folder, task_name)
            mkdir(folder)
            anneal_mdp = os.path.join(mdp_folder, 'anneal.mdp')
            run_gromacs_simulation(top_file=self.cg_top_file, gro_file=tmp_gro, index_file=index_file, mdp_file=anneal_mdp,
                                   table_file=table_file, task_name=task_name, output_folder=folder,
                                   gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
            tmp_gro = os.path.join(folder, 'anneal.gro')  # Update GRO file path for the next stage

        # run equlibrium process
        if eq:
            task_name = 'eq'
            folder = os.path.join(cg_simulation_folder, task_name)
            mkdir(folder)
            eq_mdp = os.path.join(mdp_folder, 'eq.mdp')
            run_gromacs_simulation(top_file=self.cg_top_file, gro_file=tmp_gro, index_file=index_file, mdp_file=eq_mdp,
                                   table_file=table_file, task_name=task_name, output_folder=folder,
                                   gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
            tmp_gro = os.path.join(folder, 'eq.gro')  # Update GRO file path for the next stage

        # production
        if prod:
            task_name = 'prod'
            folder = os.path.join(cg_simulation_folder, task_name)
            mkdir(folder)
            prod_mdp = os.path.join(mdp_folder, 'prod.mdp')
            run_gromacs_simulation(top_file=self.cg_top_file, gro_file=tmp_gro, index_file=index_file, mdp_file=prod_mdp,
                                   table_file=table_file, task_name=task_name, output_folder=folder,
                                   gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
            tmp_gro = os.path.join(folder, 'prod.gro')  # Update GRO file path for the next stage

        self.final_cg_gro = tmp_gro

        # Update class attributes with the final CG model files
        self.cg_topology = os.path.join(folder, f'{task_name}.tpr')
        self.cg_trajectory = os.path.join(folder, f'{task_name}.trr')
        unwrap_traj = os.path.join(folder, f'{task_name}_unwrapped.trr')
        unwrap_trajectory(topology=self.cg_topology, trajectory=self.cg_trajectory, save_file=unwrap_traj)
        self.cg_trajectory = unwrap_traj

    def Boltzmann_inversion(self, rdf_pairs_list, tag, max_distance, rdf_folder=None, begin_frame_id=None,
                            end_frame_id=None, skip_frames=None, Temperature=300, bin_width=0.002):
        """
        Performs the Boltzmann inversion method based on Radial Distribution Function (RDF) analysis to derive potential functions.

        Parameters:
        -----------
        rdf_pairs_list : list
            A list of dictionaries, each specifying the atom groups for RDF calculation.
        tag : str
            Specifies whether to perform the calculation for the 'cg' (Coarse-Grained) or 'fg' (Fine-Grained) model.
        max_distance : float
            The maximum distance for the RDF calculation.
        rdf_folder : str, optional
            Path to save the RDF data, a default folder will be created if not provided.
        begin_frame_id : int, optional
            The starting frame for RDF calculation, defaults to None (start from the first frame).
        end_frame_id : int, optional
            The ending frame for RDF calculation, defaults to None (end at the last frame).
        skip_frames : int, optional
            The number of frames to skip between evaluations, defaults to None (no skipping).
        Temperature : float, optional
            Temperature for the Boltzmann inversion calculation, defaults to 300 K.
        bin_width : float, optional
            The width of the bins for RDF calculation, defaults to 0.002 nm.

        Returns:
        --------
        list
            A list of potential energy functions derived from the RDF analysis for each pair of atom groups.

        Notes:
        ------
        This method calculates the RDF for specified pairs of atom groups and performs Boltzmann inversion to derive
        potential energy functions, which are useful for parameterizing interaction potentials in CG models.
        """

        Ur_list = []  # List to store potential energy functions

        if tag == 'cg':
            topology = self.cg_topology
            trajectory = self.cg_trajectory
            if rdf_folder is None:
                rdf_folder = os.path.join(self.opt_folder, 'cg_rdf')
        elif tag == 'fg':
            topology = self.fg_topology
            trajectory = self.fg_trajectory
            if rdf_folder is None:
                rdf_folder = os.path.join(self.opt_folder, 'fg_rdf')
        else:
            raise ValueError("Invalid tag value: must be 'cg' or 'fg'")

        mkdir(rdf_folder)  # Ensure the RDF data folder exists

        for idx, i in enumerate(rdf_pairs_list):
            rdf_save_file = os.path.join(rdf_folder, f'{idx}th_pair.dat')
            rdf = RDF(topology=topology, trajectory=trajectory, begin_frame_id=begin_frame_id, end_frame_id=end_frame_id, skip_frames=skip_frames)
            # Configure atom groups for RDF calculation
            rdf.set_atom_group(target='group1', selection=i['selection'][0])
            rdf.set_atom_group(target='group2', selection=i['selection'][1])
            rdf.configure_atom_group_centers(target='group1', groups=i['groups'][0], method=i['method'], num_atom=i['num_atom'][0])
            rdf.configure_atom_group_centers(target='group2', groups=i['groups'][1], method=i['method'], num_atom=i['num_atom'][1])
            # Perform RDF calculation and derive potential energy function
            r, rdf_y = rdf.compute_rdf(bin_width=bin_width, max_distance=max_distance)
            Ur_list.append(rdf.compute_Ur_from_rdf(T=Temperature))
            rdf.save_rdf(save_file=rdf_save_file)  # Save RDF data to file

        return Ur_list



def main():
    pass
    #
    # traj = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq_whole.trr'
    # top = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq.tpr'
    #
    # system = {'molecules': [{'mol_name': '12oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'],
    #                          'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
    #                          'aa_groups': [[0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    #                                        [4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28],
    #                                        [8, 9, 10, 11, 29, 30, 31, 32, 33, 34, 35]],
    #                          'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47],
    #                                            ('C1', 'C1'): [3.5, 0.47]},
    #                          'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
    #                          'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 100},
    #                         {'mol_name': '16oh', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'C1', 'P1'],
    #                          'id': [0, 1, 2, 3], 'charge': [0.0, 0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 56.1067, 59.0869],
    #                          'aa_groups': [[0, 1, 2, 3, 16, 17, 18, 19, 20,21,22,23,24],
    #                                        [4, 5, 6, 7, 25, 26, 27, 28,29,30,31,32],
    #                                        [8, 9, 10, 11, 33, 34, 35,36,37,38,39,40],
    #                                        [12,13,14,15,41,42,43,44,45,46,47]],
    #                          'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47],
    #                                            ('C1', 'C1'): [3.5, 0.47]},
    #                          'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0], (2, 3): [0.47, 1250.0]},
    #                          'angle_parameters': {(0, 1, 2): [180.0, 25.0], (1, 2, 3): [180.0, 25.0]}, 'num_mols': 100}],
    #           'lj_cross_terms': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
    #           'cgmodel': 'MARTINI2'}
    #
    # opt_object = ObjectiveFunction(system_top=system)
    # opt_object.force_match_loss(aa_topology=top, aa_trajectory=traj)


if __name__ == '__main__':
    main()



