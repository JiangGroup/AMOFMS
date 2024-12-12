import os
import time

os.environ['PATH'] += ':/home/xiaoyedi/data/research/tools/gromacs-2023.3/bin'

from AMOFMS.Optimization import OptParametersProcess, ParticleSwarmOptimizer, Particle
from AMOFMS.ObjectiveFunction import BottomUpObjectiveFunction
from AMOFMS.tools.utilies import mkdir
from AMOFMS.tools.math_tools import square_diff_of_elements_in_2D_list
from AMOFMS.tools.properties import MembraneProperties
from AMOFMS.tools.gromacs import find_gmx_executable

from opt_init_para import system_topology, opt_term_parse



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

rdf_skip_frames = 2
rdf_cutoff = 40  # Angstrom
rdf_binwidth = 0.4  # Angstrom
Ur_loss_ratio = 1.0
density_loss_ratio = 1.0
apl_loss_ratio = 10.0
thickness_loss_ratio = 10.0
# ka_loss_ratio = 10.0

resname_cg_from_fg_coord = None
head_group_expression = 'name Q08'


temperature = 300  # K
exp_apl = 63  # Angstrom**2  https://doi.org/10.1021/acs.jpcb.6b01870
exp_thickness = 37  # Angstrom  https://doi.org/10.1063/1.4936909
# exp_ka = 255  # mN/m  180~330

num_particles = 32
update_boundary_frequency = 5
max_iter = 20
max_processes = 8
cpu_nt = 12

aa_rdf_pairs_list = [{'selection': ['resname POPC', 'resname POPC'],
                      'groups': [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],  # CH2-PO4-CH2
                                 [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]],
                      'num_atom':[134, 134], 'method':'com'}]
cg_rdf_pairs_list = [{'selection': ['resname POPC', 'resname POPC'],
                      'groups': [[[7]],  [[7]]],
                      'num_atom':[12, 12], 'method':'com'}]

equivalent_term_id_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 2, 1] + list(range(3, 59))
equivalent_term_boundary_value_dict = {}

opt_para = OptParametersProcess(system_topology=system_topology, opt_term_parse=opt_term_parse,
                                equivalent_term_id_list=equivalent_term_id_list,
                                equivalent_term_boundary_value_dict=equivalent_term_boundary_value_dict)

opt_para_boundary = opt_para.opt_parameter_boundary_array


mkdir(opt_folder)
opt_iters_folder = os.path.join(opt_folder, 'iters')
mkdir(opt_iters_folder)

bottom_up_obj = BottomUpObjectiveFunction(system_top=system_topology, fg_trajectory=fg_trajectory, pdb_file=initial_pdb,
                                          fg_topology=fg_topology, opt_folder=opt_folder, cg_top_file_name='cg.top')

fg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(rdf_pairs_list=aa_rdf_pairs_list, tag='fg',
                                                    Temperature=temperature, bin_width=rdf_binwidth, max_distance=rdf_cutoff,
                                                    begin_frame_id=None, end_frame_id=None, skip_frames=rdf_skip_frames)

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

    print(f'{particle.iter}-iter {particle.idx}-th particle APL(Angstrom**-2): \nexp: {exp_apl} cg: {cg_apl}')
    print(f'{particle.iter}-iter {particle.idx}-th particle Thickness(Angstrom): \nexp: {exp_thickness} cg: {cg_thickness}')

    apl_loss = apl_loss_ratio * abs(exp_apl - cg_apl)  # https://doi.org/10.1021/acs.jpcb.6b01870
    thickness_loss = thickness_loss_ratio * abs(exp_thickness - cg_thickness)  # https://doi.org/10.1063/1.4936909
    ka_loss = ka_loss_ratio * abs(exp_ka - cg_ka)

    each_term_loss.update({'apl': apl_loss})
    each_term_loss.update({'thickness': thickness_loss})
    total_loss = total_loss + apl_loss + thickness_loss

    with open(tmp_result, 'a+') as f:
        line = f'{particle.iter:<12}  {particle.idx:<12}  {Ur_loss:<12}  {cg_apl:<12}  {cg_thickness:<12}\n'
        f.write(line)

    print(f'\n{particle.iter}-iter:  {particle.idx}-th particle Done!')
    return total_loss, each_term_loss


# num_particles do not set to 1
optimizer = ParticleSwarmOptimizer(objective_function=opt_loss_function, update_boundary_frequency=max_iter,
                                   bounds=opt_para_boundary, num_particles=num_particles, max_iter=max_iter, max_no_improvement_iters=max_iter)

best_para, best_score, recorder = optimizer.optimize_mpi(max_processes=max_processes)
# best_para, best_score, recorder = optimizer.optimize()
recorder.write_losses_to_file(filepath=loss_dat)

time_end = int(time.time())
timeArray_end = time.localtime(time_end)
end_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray_end)
print('Running date: ', start_time, ' - ', end_time)
print('Running Time (hour): ', (time_end-time_start)/3600)
