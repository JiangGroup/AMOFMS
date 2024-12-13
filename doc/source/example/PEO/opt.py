import os
import copy
import time

os.environ['PATH'] += ':/home/xiaoyedi/data/research/tools/gromacs-2023.3/bin'

from AMOFMS.Optimization import OptParametersProcess, ParticleSwarmOptimizer, Particle
from AMOFMS.CGForceFieldParameters import generate_system_top
from AMOFMS.tools.utilies import mkdir
from AMOFMS.ObjectiveFunction import BottomUpObjectiveFunction
from AMOFMS.tools.math_tools import square_diff_of_elements_in_2D_list
from AMOFMS.tools.properties import compute_density, compute_interface_tension

from AMOFMS.tools.gromacs import find_gmx_executable, unwrap_trajectory, run_complete_simulation
import subprocess
import pexpect

from opt_init_para import molecule_topology_list

time_start = int(time.time())
timeArray_start = time.localtime(time_start)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray_start)
print(f'\nStarting Optimization Task at {start_time} \n')


fg_topology = 'aa/prod/prod.tpr'
fg_trajectory = 'aa/prod/prod_whole.trr'
gmx = find_gmx_executable()

opt_folder = './opt'
opt_folder = os.path.realpath(opt_folder)
loss_dat = os.path.join(opt_folder, 'loss.dat')
mdp_folder = './mdp'
force_match_mdp = os.path.join(mdp_folder, 'force_match.mdp')
surface_tension_mdp = './surface_tension_mdp'
initial_gro = None
index_file = None
run_em = True
em_double = False
run_anneal = False
run_eq = False  # include anneal and eq
run_prod = True

force_match_skip_frames = 5000
rdf_skip_frames = 5000
rdf_cutoff = 20  # Angstrom
rdf_binwidth = 0.04  # Angstrom

force_match_loss_ratio = 0.1
Ur_loss_ratio = 1.0
density_loss_ratio = 1.0
surface_tension_loss_ratio = 1.0
pressure_group = '21 25 29 30 0'

temperature = 298  # K
exp_denisty = 1127  # kg/m3 https://pubs.acs.org/doi/10.1021/acs.macromol.8b01802
exp_surface_tension = 42.6  # mN/m  https://pubs.acs.org/doi/10.1021/acs.macromol.8b01802
surface_box_z = 200  # Angstrom

num_particles = 2
update_boundary_frequency = 5
max_iter = 2
max_processes = 2
cpu_nt = 8

aa_rdf_pairs_list = [{'selection': ['resname PEO OHter', 'resname PEO OHter'],
                      'groups': [[[0, 1, 2, 3, 4], [355, 356, 357, 358, 359]],  [[0, 1, 2, 3, 4], [355, 356, 357, 358, 359]]],
                      'num_atom':[360, 360], 'method':'com'}]
cg_rdf_pairs_list = [{'selection': ['resname PEO52', 'resname PEO52'],
                      'groups': [[[0], [51]],  [[0], [51]]], 'num_atom': [52, 52], 'method':'com'}]
resname_cg_from_fg_coord = ['PEO OHter']
system_topology = generate_system_top(mols=molecule_topology_list, num_mols=None)


for i in system_topology['molecules']:
    eq_bond = [bond[0] for bond in i['bond_parameters'].values()]
    eq_bond_termini = round((eq_bond[0] + eq_bond[-1])/2, 4)
    eq_bond_repeat_unit = round(sum(eq_bond[1:-1]) / len(eq_bond[1:-1]), 4)

    eq_angle = [angle[0] for angle in i['angle_parameters'].values()]
    eq_angle_termini = round((eq_angle[0] + eq_angle[-1])/2, 4)
    eq_angle_repeat_unit = round(sum(eq_angle[1:-1]) / len(eq_angle[1:-1]), 4)

    for idx, j in enumerate(i['bond_parameters'].keys()):
        if idx in [0, len(i['bond_parameters'].keys())-1]:
            i['bond_parameters'][j][0] = eq_bond_termini
        else:
            i['bond_parameters'][j][0] = eq_bond_repeat_unit
        i['bond_parameters'][j][1] = 7000

    for idx, j in enumerate(i['angle_parameters'].keys()):
        if idx in [0, len(i['angle_parameters'].keys())-1]:
            i['angle_parameters'][j][0] = eq_angle_termini
        else:
            i['angle_parameters'][j][0] = eq_angle_repeat_unit
        i['angle_parameters'][j][1] = 80


opt_term_parse = copy.deepcopy(system_topology)
for i in opt_term_parse['molecules']:

    del i['types']
    del i['id']
    del i['mass']
    del i['fg_groups']
    del i['charge']

    for j in i['bond_parameters'].keys():
        i['bond_parameters'][j] = (0, 0.1)
    for j in i['angle_parameters'].keys():
        i['angle_parameters'][j] = (0, 0.1)

for i in opt_term_parse['lj_cross_terms'].keys():
    opt_term_parse['lj_cross_terms'][i] = (0.1, 0.1)

equivalent_term_id_list = []

id_list = []
for _ in range(len(opt_term_parse['molecules'][0]['bond_parameters'].keys())):
    id_list.append(1)

id_list[0] = 0
id_list[-1] = 0
equivalent_term_id_list += id_list

id_list = []
for _ in range(len(opt_term_parse['molecules'][0]['angle_parameters'].keys())):
    id_list.append(3)

id_list[0] = 2
id_list[-1] = 2
equivalent_term_id_list += id_list

id_list = list(range(4, 10))
equivalent_term_id_list += id_list

equivalent_term_boundary_value_dict = {}
# for i in range(14):
#     equivalent_term_boundary_value_dict.update({i: (0.1, 0.1)})


opt_para = OptParametersProcess(system_topology=system_topology, opt_term_parse=opt_term_parse,
                                equivalent_term_id_list=equivalent_term_id_list,
                                equivalent_term_boundary_value_dict=equivalent_term_boundary_value_dict)

opt_para_boundary = opt_para.opt_parameter_boundary_array


mkdir(opt_folder)
opt_iters_folder = os.path.join(opt_folder, 'iters')
mkdir(opt_iters_folder)

bottom_up_obj = BottomUpObjectiveFunction(system_top=system_topology, fg_trajectory=fg_trajectory,
                                          fg_topology=fg_topology, opt_folder=opt_folder, cg_top_file_name='cg.top')

fg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(rdf_pairs_list=aa_rdf_pairs_list, tag='fg',
                                                    Temperature=temperature, bin_width=rdf_binwidth, max_distance=rdf_cutoff,
                                                    begin_frame_id=None, end_frame_id=None, skip_frames=rdf_skip_frames)

def opt_loss_function(particle: Particle):
    print(f'\n{particle.iter}-iter:  {particle.idx}-th particle processing...')
    new_topology = opt_para.unpack_updated_parameters_to_top(updated_parameters_array=particle.position)
    iter_folder = os.path.join(opt_folder, f'iters/iter_{particle.iter}')
    idx_folder = os.path.join(iter_folder, f'{particle.idx}')

    bottom_up_obj.update_system_topology(new_system_top=new_topology)
    bottom_up_obj.update_opt_folder(new_opt_folder=idx_folder)

    total_loss = 0
    each_term_loss = {}

    print('\nComputing force match loss...')
    force_match_loss = force_match_loss_ratio * bottom_up_obj.force_match_loss(cg_mdp_file=force_match_mdp, fg_resname_list=resname_cg_from_fg_coord, begin_frame=None, end_frame=None, skip_frame=force_match_skip_frames)
    each_term_loss.update({'force_match': force_match_loss})
    total_loss += force_match_loss

    print('\nComputing Ur Boltzmann inversion loss...')
    bottom_up_obj.run_cg_simulation(initial_gro=initial_gro, fg_resname_list=resname_cg_from_fg_coord,
                                    mdp_folder=mdp_folder, index_file=index_file, em_double_version=em_double,
                                    cg_simulation_folder=idx_folder, em=run_em, anneal=run_anneal,
                                    eq=run_eq, prod=run_prod, nt=cpu_nt, gpu_id=particle.idx % max_processes)
    cg_pair_Ur_list = bottom_up_obj.Boltzmann_inversion(rdf_pairs_list=cg_rdf_pairs_list, tag='cg',
                                                        Temperature=temperature, bin_width=rdf_binwidth, max_distance=rdf_cutoff)
    Ur_loss = Ur_loss_ratio * square_diff_of_elements_in_2D_list(list1=fg_pair_Ur_list, list2=cg_pair_Ur_list)
    each_term_loss.update({'Ur': Ur_loss})
    total_loss += Ur_loss

    print('\nComputing density loss...')
    cg_density, _ = compute_density(topology=bottom_up_obj.cg_topology, trajectory=bottom_up_obj.cg_trajectory)
    print(f'{particle.iter}-iter {particle.idx}-th particle density(kg/m3): \nexp: {exp_denisty} cg: {cg_density}')
    density_loss = density_loss_ratio * abs(cg_density - exp_denisty)
    each_term_loss.update({'density': density_loss})
    total_loss += density_loss

    print('\nComputing surface tension loss...')
    surface_tension_folder = os.path.join(idx_folder, 'surface_tension')
    mkdir(surface_tension_folder)
    init_gro = os.path.join(surface_tension_folder, 'init.gro')
    unwrap_trajectory(topology=bottom_up_obj.cg_topology, trajectory=bottom_up_obj.final_cg_gro, save_file=init_gro)
    with open(init_gro, 'r') as f:
        box_vector = f.readlines()[-1].split()
        box_x, box_y = box_vector[0], box_vector[1]
    surface_gro = os.path.join(surface_tension_folder, 'surface.gro')
    extend_z_command = f'{gmx} editconf -f {init_gro} -o {surface_gro} -c -box {box_x} {box_y} {surface_box_z/10}'
    subprocess.run(extend_z_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    run_complete_simulation(mdp_folder=surface_tension_mdp, initial_gro=surface_gro, cg_top_file=bottom_up_obj.cg_top_file,
                            cg_simulation_folder=surface_tension_folder, em=False, anneal=False, eq=True, prod=True,
                            nt=cpu_nt, gpu_id=0)
    surface_tension_dat = os.path.join(surface_tension_folder, 'surface_tension_average.dat')
    gen_surface_tension_dat_command = f'{gmx} energy -f  {surface_tension_folder}/prod/prod.edr -o {surface_tension_folder}/surface_tension.xvg'
    child = pexpect.spawn(gen_surface_tension_dat_command)
    # 需要时发送输入
    try:
        # time.sleep(2)
        child.expect("End your selection with an empty line or a zero.")  # 等待命令提示需要选择一个组
        child.sendline(pressure_group)  # 发送所需的组号
        child.expect(pexpect.EOF)  # 等待命令执行完成
    except pexpect.EOF:
        raise Exception('gmx trjconv process terminated unexpectedly.')
    except pexpect.TIMEOUT:
        raise Exception('gmx trjconv process timed out.')

    output = child.before.decode()
    child.expect(pexpect.EOF)
    if child.exitstatus != 0:
        print(output)
        raise Exception('\ngmx energy compute surface tension failed')

    # 将输出保存到文件
    with open(surface_tension_dat, 'w') as file:
        file.write(output)

    cg_surface_tension = compute_interface_tension(pressure_dat=surface_tension_dat, num_interface=2)
    print(f'{particle.iter}-iter {particle.idx}-th particle surface_tension(mN/m):\n exp: {exp_surface_tension}  cg: {cg_surface_tension}')
    surface_tension_loss = surface_tension_loss_ratio * abs(cg_surface_tension - exp_surface_tension)
    each_term_loss.update({'surface tension': surface_tension_loss})
    total_loss += surface_tension_loss

    print(f'\n{particle.iter}-iter:  {particle.idx}-th particle Done!')
    return total_loss, each_term_loss


# num_particles do not set 1
optimizer = ParticleSwarmOptimizer(objective_function=opt_loss_function, update_boundary_frequency=update_boundary_frequency,
                                   bounds=opt_para_boundary, num_particles=num_particles, max_iter=max_iter)

# best_para, best_score, recorder = optimizer.optimize_mpi(max_processes=max_processes)
best_para, best_score, recorder = optimizer.optimize()
recorder.write_losses_to_file(filepath=loss_dat)


time_end = int(time.time())
timeArray_end = time.localtime(time_end)
end_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray_end)
print('Running date: ', start_time, ' - ', end_time)
print('Running Time (hour): ', (time_end-time_start)/3600)

