import MDAnalysis as mda
from scipy.constants import Avogadro
from scipy.constants import Boltzmann
import numpy as np
import logging
import warnings

from MDAnalysis.analysis.msd import EinsteinMSD
from scipy.stats import linregress

from lipyphilic.lib.assign_leaflets import AssignLeaflets
from lipyphilic.lib.area_per_lipid import AreaPerLipid
from lipyphilic.lib.memb_thickness import MembThickness
from lipyphilic.lib.lateral_diffusion import MSD
from lipyphilic.lib.order_parameter import SCC

# from lipyphilic.leaflets.assign_leaflets import AssignLeaflets


# Set the logging level to WARNING
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


# suit for wrapped or unwrapped positions
def compute_distance_under_pbc(pos1, pos2, box_size):
    delta = pos2 - pos1

    # Apply periodic boundary conditions to each dimension
    for i in range(len(box_size)):
        while delta[i] > 0.5 * box_size[i]:
            delta[i] -= box_size[i]
        while delta[i] < -0.5 * box_size[i]:
            delta[i] += box_size[i]

    distance = np.linalg.norm(delta)
    return distance


def compute_density(topology, trajectory):
    universe = mda.Universe(topology, trajectory)

    density_list = []

    # Convert relative molecular mass from g/mol to kg
    # Convert mass from g/mol to kg
    mass = sum(atom.mass for atom in universe.atoms) / Avogadro * 1e-3

    # Iterate over each frame in the trajectory
    for ts in universe.trajectory:
        # Convert volume from cubic angstroms to cubic meters
        volume = ts.volume * (1e-10)**3

        # compute density (kg/m^3)
        density = mass / volume
        density_list.append(density)

    # compute average density (kg/m^3)
    average_density = sum(density_list) / len(density_list)

    return average_density, density_list


def compute_interface_tension(pressure_dat, num_interface=2):
    """
    Calculate the surface tension from the specified file. (mN/m)

    :param pressure_dat: The filename containing surface tension data
    :param num_interface: Number of interfaces, defaults to 2
    :return: The calculated surface tension, or None if unable to calculate
    """
    try:
        with open(pressure_dat, 'r') as f:
            all_lines = f.readlines()
            for line in all_lines:
                parts = line.split()
                if parts and parts[0] == '#Surf*SurfTen':
                    interface_tension = round(float(parts[1]) / 10 / num_interface, 6)
                    return interface_tension
    except FileNotFoundError:
        print(f"File '{pressure_dat}' not found.")
        return None
    except IndexError:
        print("Incorrect file format or missing data.")
        return None
    except ValueError:
        print("Data in the file cannot be converted to floating-point numbers.")
        return None

    # If the function hasn't returned within the loop, it means the required line was not found in the file
    print("'#Surf*SurfTen' not found in the file.")
    return None


class DiffusionCalculator:
    def __init__(self, topology, trajectory, selection='all'):
        """
        Initialize the DiffusionCalculator class.

        Parameters:
        topology: Path to the topology file, such as a PDB file.
        trajectory: Path to the trajectory file, such as an XTC or DCD file.
        """
        self.universe = mda.Universe(topology, trajectory)
        self.selection = self.universe.select_atoms(selection)
        self.msd = None

    def calculate_msd(self, start=None, stop=None, step=None):
        """
        Calculate the MSD (Mean Squared Displacement) for the selected molecules.

        Parameters:
        start, stop, step: Parameters for slicing the trajectory.
        """
        if start is None:
            start = 0

        if stop is None:
            stop = len(self.universe.trajectory)

        if step is None:
            step = 1

        sample_interval = self.universe.trajectory.dt  # ps
        # sample_interval = self.universe.trajectory[1].time - self.universe.trajectory[0].time
        self.msd_times = [sample_interval * x for x in range(start, stop, step)]   # ps
        self.msd_cal = EinsteinMSD(self.selection, msd_type='xyz', fft=True).run(start=start, stop=stop, step=step)
        self.msd = self.msd_cal.results.timeseries
        return self.msd, self.msd_times

    def find_linear_region(self, length=0.8):
        times = self.msd_times
        start = int(len(times) * ((1-length)/2))
        end = int(len(times) * (((1-length)/2) + length))
        return (start, end)

    def calculate_diffusion_coefficient(self, linear_region=None, region_length=0.8):
        """
        calculate diffusion coefficient  https://docs.mdanalysis.org/stable/documentation_pages/analysis/msd.html#module-MDAnalysis.analysis.msd

        return:
        diffusion coefficien (1e-5 cm^2/s)
        """
        if linear_region is None:
            linear_region = self.find_linear_region(length=region_length)

        start, end = linear_region[0], linear_region[1]
        times = self.msd_times[start:end]  # ps
        msds = self.msd[start:end]  # Angstrome

        # Perform linear regression on the identified linear region
        slope, intercept, r_value, p_value, std_err = linregress(times, msds)

        if 0.7 < r_value < 0.9:
            warnings.warn(f"The linear region (R={r_value}) is not suit to fit a linear regression.")
        elif r_value < 0.7:
            slope = 100
            print(f"The linear region (R={r_value}) is too small to fit a linear regression!")

        # The diffusion coefficient is half the slope of the MSD vs time plot in 3D
        diffusion_coefficient = slope * 10 / 6.0  # 1e-5 cm^2/s

        return diffusion_coefficient



class MembraneProperties:
    # according to lipyphilic https://lipyphilic.readthedocs.io/en/latest
    def __init__(self, topology, trajectory, begin_frame_id=None, end_frame_id=None, skip_frames=None):
        self.universe = mda.Universe(topology, trajectory)
        self.begin_frame_id = 0 if begin_frame_id is None else begin_frame_id
        self.end_frame_id = len(self.universe.trajectory) if end_frame_id is None else end_frame_id
        self.skip_frames = 1 if skip_frames is None else skip_frames

    def compute_apl(self, headgroup_selection):
        leaflets = AssignLeaflets(universe=self.universe, lipid_sel=headgroup_selection)

        leaflets.run(start=self.begin_frame_id, stop=self.end_frame_id, step=self.skip_frames, verbose=True)
        areas = AreaPerLipid(universe=self.universe, lipid_sel=headgroup_selection, leaflets=leaflets.leaflets)

        areas.run(start=self.begin_frame_id, stop=self.end_frame_id, step=self.skip_frames, verbose=True)
        area_frames = np.mean(areas.areas, axis=0)
        apl_avg = np.mean(area_frames)
        apl_std = np.std(area_frames)

        self.complete_apl_calculation = True
        self.apl_avg = apl_avg   # Angstrom**2
        self.apl_std = apl_std

        return apl_avg, apl_std

    def compute_Ka(self, headgroup_selection, T=298.15):
        if not hasattr(self, 'complete_apl_calculation') or not self.complete_apl_calculation:
            self.compute_apl(headgroup_selection=headgroup_selection)

        # Calculate the number of lipids (total)
        nl = self.universe.select_atoms(headgroup_selection).n_residues

        # compute Ka Isothermal area compressibility modulus
        # Ka = (2 * Boltzmann * T * self.apl_avg) / (nl * (self.apl_std ** 2))
        # https://gromacs.bioexcel.eu/t/is-it-possible-to-compute-isothermal-area-compressibility-modulus-in-gromacs/2369
        Ka = 2 * 1000 * (Boltzmann * T * self.apl_avg * 1e-20) / (nl * (self.apl_std ** 2) * 1e-40)  # mN m–1
        return Ka

    def compute_membrane_thickness(self, headgroup_selection):
        leaflets = AssignLeaflets(universe=self.universe, lipid_sel=headgroup_selection)
        leaflets.run(start=self.begin_frame_id, stop=self.end_frame_id, step=self.skip_frames, verbose=True)
        thickness = MembThickness(universe=self.universe, lipid_sel=headgroup_selection, leaflets=leaflets.leaflets)
        thickness.run(start=self.begin_frame_id, stop=self.end_frame_id, step=self.skip_frames, verbose=True)

        thickness_avg = np.mean(thickness.memb_thickness)  # Angstrom
        thickness_std = np.std(thickness.memb_thickness)

        return thickness_avg, thickness_std

    def compute_lateral_diffusion(self, lipid_selection, diffusion_coefficient=False, start_fit=None, end_fit=None,
                                  com_removal_selection=None):
        msd = MSD(universe=self.universe, lipid_sel=lipid_selection)
        if com_removal_selection is not None:
            msd = MSD(universe=self.universe, lipid_sel=lipid_selection, com_removal_sel=com_removal_selection)
        msd.run(start=self.begin_frame_id, stop=self.end_frame_id, step=self.skip_frames, verbose=True)
        msd_frames = np.mean(msd.msd, axis=0)

        if diffusion_coefficient:
            d_avg, d_std = msd.diffusion_coefficient(start_fit=start_fit, stop_fit=end_fit)
            return msd_frames, d_avg, d_std  # cm^2/s
        else:
            return msd_frames

    def compute_order_parameter(self, tail_sel, normals=None, start=None, stop=None, step=None):

        # https://lipyphilic.readthedocs.io/en/latest/reference/analysis/order_parameter.html
        scc_sn1 = SCC(universe=self.universe, tail_sel=tail_sel, normals=normals)
        scc_sn1.run(start=start, stop=stop, step=step)

        return scc_sn1.SCC


class RDF:
    def __init__(self, topology, trajectory, begin_frame_id=None, end_frame_id=None, skip_frames=None, minium_distance=1e-5):
        self.universe = mda.Universe(topology, trajectory)
        self.begin_frame_id = 0 if begin_frame_id is None else begin_frame_id
        self.end_frame_id = len(self.universe.trajectory) if end_frame_id is None else end_frame_id
        self.skip_frames = 1 if skip_frames is None else skip_frames
        self.minium_distance = minium_distance
        self.group1_position_array = None
        self.group2_position_array = None

        self.group1_masses = None
        self.group2_masses = None

        self.box_sizes = []
        for ts in self.universe.trajectory[self.begin_frame_id:self.end_frame_id:self.skip_frames]:
            box_dims = np.copy(ts.dimensions[0:3])
            self.box_sizes.append(box_dims)
        self.box_sizes = np.array(self.box_sizes)

        self.r = None
        self.rdf = None

    # selection expression according to https://docs.mdanalysis.org/stable/documentation_pages/selections.html
    def set_atom_group(self, target, selection):
        if target not in ['group1', 'group2']:
            raise ValueError("Target must be 'group1' or 'group2'")

        group = self.universe.select_atoms(selection)
        if target == 'group1':
            self.group1_masses = np.array([atom.mass for atom in group])
        elif target == 'group2':
            self.group2_masses = np.array([atom.mass for atom in group])

        positions = []
        for ts in self.universe.trajectory[self.begin_frame_id:self.end_frame_id:self.skip_frames]:
            group = self.universe.select_atoms(selection)
            positions.append(group.positions)

        if target == 'group1':
            self.group1_position_array = np.array(positions)
        elif target == 'group2':
            self.group2_position_array = np.array(positions)

    def configure_atom_group_centers(self, target, groups, num_atom=None, num_mol=None, method='com'):
        if method not in ['com', 'cog']:
            raise ValueError("Method must be 'com' or 'cog'")

        if target == 'group1':
            position_array = self.group1_position_array
            masses = self.group1_masses if method == 'com' else None
        elif target == 'group2':
            position_array = self.group2_position_array
            masses = self.group2_masses if method == 'com' else None
        else:
            raise ValueError("Target must be 'group1' or 'group2'")

        if position_array is None or (method == 'com' and masses is None):
            raise ValueError("Position array and masses must be set before setting group centers")

        num_frames = position_array.shape[0]

        if num_atom is None:
            num_atom = sum(len(group) for group in groups)
        if num_mol is None:
            num_mol = position_array.shape[1] // num_atom

        assert num_mol == position_array.shape[1] / num_atom, \
            "Number of molecules does not match the number of molecules in the position array"

        updated_positions = np.zeros((num_frames, len(groups) * num_mol, 3))

        for frame_idx in range(num_frames):
            for mol_id in range(num_mol):
                start_idx = mol_id * num_atom
                end_idx = (mol_id + 1) * num_atom

                for group_idx, group in enumerate(groups):
                    group_positions = position_array[frame_idx, start_idx:end_idx][group]

                    if method == 'com':
                        group_masses = masses[start_idx:end_idx][group]
                        total_mass = np.sum(group_masses)
                        weighted_positions = group_positions * group_masses[:, np.newaxis]
                        centroid = np.sum(weighted_positions, axis=0) / total_mass if total_mass != 0 else np.zeros(3)
                    else:
                        centroid = np.mean(group_positions, axis=0)

                    updated_positions[frame_idx, mol_id * len(groups) + group_idx] = centroid

        if target == 'group1':
            self.group1_position_array = updated_positions
        elif target == 'group2':
            self.group2_position_array = updated_positions

    def compute_rdf(self, bin_width=0.02, max_distance=None):
        if self.group1_position_array is None or self.group2_position_array is None:
            raise ValueError("Both group position arrays must be set before computing RDF.")

        if max_distance is None:
            min_box_size = np.min(self.box_sizes)
            max_distance = min_box_size / 2

        nbins = int(max_distance / bin_width)

        all_rdfs = []
        r = []

        for frame_idx, (frame1, frame2) in enumerate(zip(self.group1_position_array, self.group2_position_array)):
            box_size = self.box_sizes[frame_idx]
            box_volume = np.prod(box_size)
            number_density = len(frame2) / box_volume

            distances = []

            for pos1 in frame1:
                for pos2 in frame2:
                    distance = compute_distance_under_pbc(pos1, pos2, box_size)
                    if self.minium_distance < distance <= max_distance:
                        distances.append(distance)

            hist, edges = np.histogram(distances, bins=nbins, range=(0.0, max_distance), density=False)
            r = 0.5 * (edges[1:] + edges[:-1])

            # normalize RDF
            shell_volumes = 4.0 / 3.0 * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
            rdf_frame = hist / (number_density * shell_volumes * len(frame1))
            all_rdfs.append(rdf_frame)

        avg_rdf = np.mean(all_rdfs, axis=0)

        self.r = r[5:]
        self.rdf = avg_rdf[5:]
        return self.r, self.rdf

    def compute_Ur_from_rdf(self, T=298.15):
        # unit  kJ/mol
        return [-Avogadro * Boltzmann * T * np.log(g)/1000 if g > 1e-5 else -Avogadro * Boltzmann * T * np.log(1e-4)/1000 for g in self.rdf]

    def save_rdf(self, save_file='rdf.txt'):
        """
        Save RDF (Radial Distribution Function) data to a file.

        Parameters:
            save_file (str): The filename to save the data.
        """

        if self.r is None or self.rdf is None:
            raise ValueError("RDF data not computed yet")

        # Combine r and rdf into a 2D array
        data = np.vstack((self.r, self.rdf)).T

        # save file
        np.savetxt(save_file, data, header='r rdf', fmt='%-15.6f')

        print(f"RDF data saved to {save_file}")


def main():

    traj = '/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/example/doab/aa/prod/prod_msd.xtc'
    top = '/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/example/doab/aa/prod/prod.tpr'

    msd = DiffusionCalculator(topology=top, trajectory=traj, selection='resname DOA')
    msd.calculate_msd()
    d = msd.calculate_diffusion_coefficient()


    traj = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/POPC/iters/iter_2/2/prod/prod.trr'
    top = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/POPC/iters/iter_2/2/prod/prod.tpr'

    mem = MembraneProperties(topology=top, trajectory=traj)
    msd = mem.compute_lateral_diffusion(lipid_selection='resname POPC', diffusion_coefficient=True)
    apl = mem.compute_apl(headgroup_selection='name Q08')
    thickness = mem.compute_membrane_thickness(headgroup_selection='name Q08')

    traj = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq_whole.trr'
    top = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/eq.tpr'

    groups1 = [[0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20],
              [4, 5, 6, 7, 21, 22, 23, 24, 25, 26, 27, 28],
              [8, 9, 10, 11, 29, 30, 31, 32, 33, 34, 35]]

    groups2 = [[0, 1, 2, 3, 16, 17, 18, 19, 20,21,22,23,24],
               [4, 5, 6, 7, 25, 26, 27, 28,29,30,31,32],
               [8, 9, 10, 11, 33, 34, 35,36,37,38,39,40],
               [12,13,14,15,41,42,43,44,45,46,47]]

    groups1 = [[8, 9, 10, 11, 29, 30, 31, 32, 33, 34, 35]]

    groups2 = [[12, 13, 14, 15, 41, 42, 43, 44, 45, 46, 47]]


    rdf = RDF(topology=top, trajectory=traj, skip_frames=1)
    # rdf.set_atom_group(target='group1', selection='name O0B')
    # rdf.set_atom_group(target='group2', selection='name O0F')
    rdf.set_atom_group(target='group1', selection='resname 12oh')
    rdf.set_atom_group(target='group2', selection='resname 16oh')
    rdf.configure_atom_group_centers(target='group1', groups=groups1, num_atom=36, method='com')
    rdf.configure_atom_group_centers(target='group2', groups=groups2, num_atom=48, method='com')
    rdf.compute_rdf(bin_width=0.002)
    rdf_file = '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/AA/eq/rdf2.txt'
    rdf.save_rdf(save_file=rdf_file)
    average_density, density_list = compute_density(top, traj)
    print('yes')




# def main():
#     surface_dat = '/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/example/peo/aa/prod/surface_tension_test/result.dat'
#     interface_tension = compute_interface_tension(surface_dat, num_interface=2)
#     pass

if __name__ == "__main__":
    main()