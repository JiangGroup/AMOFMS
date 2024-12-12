import datetime
import subprocess
import os
import pexpect
import warnings
import shutil
import sys

tools_path = os.path.dirname(os.path.realpath(__file__))
spica_ff_prm = os.path.join(tools_path, "force_field/spica_protein_v2.prm")


def find_gmx_executable():
    try:
        gmx_path = subprocess.check_output("which gmx", shell=True).decode().strip()
        # print(f"GROMACS 'gmx' executable found at: {gmx_path}")
        return gmx_path
    except subprocess.CalledProcessError:
        print("GROMACS 'gmx' executable not found in the PATH.")
        return None

# os.environ['PATH'] += ':/home/xiaoyedi/data/research/tools/gromacs-2023.3/bin'
gmx_exec =find_gmx_executable()
# gmx_exec = '/home/xiaoyedi/data/research/tools/gromacs-2023/bin/gmx'


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_top_file(system_top, save_file="system.top", pdb_file=None):
    if system_top['cgmodel'] in ['MARTINI2', 'MARTINI3']:
         with open(save_file, 'w') as file:
            # record time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # write head of file
            file.write(f"; Topology file generated on {now}\n\n")

            file.write("[ defaults ]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ;\n")
            file.write("1 1 no 1.0 1.0\n\n")  # combine-rule

            # extract all atom types
            all_types = set()
            for molecule in system_top['molecules']:
                all_types |= set(molecule['types'])

            file.write("[ atomtypes ]\n")
            # atomtype atnum mass charge sigma epsilon\n
            for atomtype in all_types:
                mass, charge = 72.0, 0.0  # 示例值
                sigma, epsilon = 0.0, 0.0
                file.write(f"{atomtype} {mass} {charge} A {sigma} {epsilon}\n")
            file.write("\n")

            # write [ nonbond_params ]
            file.write("[ nonbond_params ]\n")
            for pair, params in system_top['lj_cross_terms'].items():
                # params[0]: sigma, params[1]: epsilon
                file.write(f"{pair[0]} {pair[1]} 1 {4*params[1]*(params[0])**6:.5E} {4*params[1]*(params[0])**12:.5E}\n")
            file.write("\n")

            # write [ moleculetype ]
            for molecule in system_top['molecules']:
                file.write(f"[ moleculetype ]\n")
                file.write(f"; Name nrexcl\n")
                file.write(f"{molecule['mol_name']} 1\n\n")

                file.write(f"[ atoms ]\n")
                file.write(f"; nr type resnr residue atom cgnr charge mass\n")
                for i, typ in enumerate(molecule['types']):
                    file.write(f"{i + 1} {typ} 1 {molecule['mol_name']} {typ}{i + 1} 1 {molecule['charge'][i]} {molecule['mass'][i]}\n")
                file.write("\n")

                # write bond information
                if 'bond_parameters' in molecule and molecule['bond_parameters']:
                    file.write(f"[ bonds ]\n")
                    for bond, params in molecule['bond_parameters'].items():
                        file.write(f"{bond[0] + 1} {bond[1] + 1} 1 {params[0]} {params[1]}\n")
                    file.write("\n")

                # write angle information
                if 'angle_parameters' in molecule and molecule['angle_parameters']:
                    file.write(f"[ angles ]\n")
                    for angle, params in molecule['angle_parameters'].items():
                        file.write(f"{angle[0] + 1} {angle[1] + 1} {angle[2] + 1} 2 {params[0]} {params[1]}\n")
                    file.write("\n")

                #  write dihedral information
                if 'dihedral_parameters' in molecule and molecule['dihedral_parameters']:
                    file.write(f"[ dihedrals ]\n")
                    for dihedral, params in molecule['dihedral_parameters'].items():
                        file.write(f"{dihedral[0] + 1} {dihedral[1] + 1} {dihedral[2] + 1} {dihedral[3] + 1} 1 {params[0]} {params[1]} 1\n")
                    file.write("\n")

                # write constraint
                if 'bond_constraint' in molecule and molecule['bond_constraint']:
                    file.write(f"[ constraints ]\n")
                    for constraint, params in molecule['bond_constraint'].items():
                        file.write(f"{constraint[0] + 1} {constraint[1] + 1} {params[0]} {params[1]}\n")

            #  write [ system ]
            file.write("[ system ]\n")
            file.write("Generated system\n\n")

            # write [ molecules ]
            file.write("[ molecules ]\n")
            for molecule in system_top['molecules']:
                file.write(f"{molecule['mol_name']} {molecule['num_mols']}\n")

    elif system_top['cgmodel'] == 'SPICA':
        check_spica_tools()
        top_folder = os.path.dirname(save_file)
        write_spica_top(system_top=system_top, output_folder=top_folder)
        update_spica_ff_file(system_top=system_top, original_spica_ff_file=spica_ff_prm, output_folder=top_folder)
        cg_spica_command = "cg_spica setup_gmx "
        for molecule in system_top['molecules']:
            mol_top = os.path.join(top_folder, f"{molecule['mol_name']}.top")
            cg_spica_command += f"{mol_top} {molecule['num_mols']}  "

        cg_spica_command += os.path.join(top_folder, 'spica_ff.prm')
        cg_spica_command += " -output_folder " + top_folder
        # child = pexpect.spawn(cg_spica_command)
        # # 等待命令执行完成
        # child.expect(pexpect.EOF)
        #
        # # 打印命令输出
        # print(child.before.decode())

        result = subprocess.run(cg_spica_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"generation of  .itp (SPICA_FF) fails:")
            print(result.stderr.decode())
            return 0

        ndx_file = os.path.join(top_folder, 'CGindex.ndx')
        gen_gmxin_command = f"cg_spica gen_gmxin -pdb {pdb_file} -ndx {ndx_file} -o {top_folder}"
        result2 = subprocess.run(gen_gmxin_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result2.returncode != 0:
            print(f"generation of table.xvg (SPICA_FF) fails:")
            print(result2.stderr.decode())
            return 0

        o_topo = os.path.join(top_folder, 'topol.top')

        shutil.move(o_topo, save_file)


def run_gromacs_simulation(top_file, gro_file, mdp_file, index_file=None, table_file=None, double_version=False, em=False,
                           output_folder='md_output', task_name='simulation', gpu_acceleration=True, nt=16, gpu_id=0, maxwarn=5):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gmx = gmx_exec

    # generate .tpr file
    if double_version:
        gmx = f'{gmx_exec}_d'
        gpu_acceleration = False

    tpr_file = os.path.join(output_folder, f"{task_name}.tpr")
    grompp_command = f"{gmx} grompp -f {mdp_file} -c {gro_file} -p {top_file} -o {tpr_file} -maxwarn {maxwarn} -v"
    if index_file is not None:
        grompp_command = f'{grompp_command} -n {index_file}'
    result = subprocess.run(grompp_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"generation of  {task_name} .tpr fails:")
        print(result.stderr.decode())
        sys.exit(1)
    print(f"successfully generate {task_name} .tpr file")

    # conduct simulation
    output = os.path.join(output_folder, f"{task_name}")
    log = f'{output}_run.log'
    mdrun_command = f"{gmx} mdrun -deffnm {output} -nt {nt} -v"

    if table_file is not None:
        gpu_acceleration = False
        mdrun_command = f'{mdrun_command} -table {table_file}'

    if gpu_acceleration:
        if em:
            mdrun_command = f'{mdrun_command} -gpu_id {gpu_id} '  # Bonded/pme/nb interactions can not be computed on a GPU: Cannot compute bonded interactions on a GPU, because GPU implementation requires a dynamical integrator (md, sd, etc). em is steep.
        else:
            mdrun_command = f'{mdrun_command} -gpu_id {gpu_id} -pme gpu -nb gpu -bonded gpu '

    mdrun_command = f'{mdrun_command} > {log} 2>&1'
    result = subprocess.run(mdrun_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"GROMACS {task_name} success")
    else:
        print(f"GROMACS {task_name} fails:")
        print(result.stderr.decode())
        sys.exit(1)


def unwrap_trajectory(topology, trajectory, save_file='traj_whole.trr'):
    unwrap_command = f'{gmx_exec} trjconv -f {trajectory} -s {topology} -o {save_file} -pbc whole -force yes'

    # execute the child process
    child = pexpect.spawn(unwrap_command)

    try:
        child.expect("Select a group:")
        child.sendline("0")
        child.expect(pexpect.EOF)
    except pexpect.EOF:
        raise Exception('gmx trjconv process terminated unexpectedly.')
    except pexpect.TIMEOUT:
        raise Exception('gmx trjconv process timed out.')

    output = child.before.decode()
    child.expect(pexpect.EOF)

    if child.exitstatus != 0:
        print(output)
        raise Exception('\ngmx trjconv unwrapped trajectory failed')


# use for calculation msd by mdanalysis
def nojump_trajectory(topology, trajectory, save_file='traj_nojump.trr'):
    nojump_command = f'{gmx_exec} trjconv -f {trajectory} -s {topology} -o {save_file} -pbc nojump'

    child = pexpect.spawn(nojump_command)

    try:
        child.expect("Select a group:")
        child.sendline("0")
        child.expect(pexpect.EOF)
    except pexpect.EOF:
        raise Exception('gmx trjconv process terminated unexpectedly.')
    except pexpect.TIMEOUT:
        raise Exception('gmx trjconv process timed out.')

    output = child.before.decode()
    child.expect(pexpect.EOF)

    if child.exitstatus != 0:
        print(output)
        raise Exception('\ngmx trjconv unwrapped trajectory failed')


def run_complete_simulation(mdp_folder, initial_gro, cg_top_file, cg_simulation_folder,
                            table_file=None, gpu_acceleration=True, em_double_version=True,
                            em=True, anneal=True, eq=True, prod=True, nt=8, gpu_id=0):
    """
    run a series simulation。
    :param mdp_folder: MDP file folder
    :param em: bool
    :param anneal: bool
    :param eq: bool
    :param prod: bool
    :return: bool
    """

    mkdir(cg_simulation_folder)

    tmp_gro = initial_gro

    folder = None
    task_name = None

    # run energy minimization before simulation
    if em:
        task_name = 'em'
        folder = os.path.join(cg_simulation_folder, task_name)
        mkdir(folder)
        em_mdp = os.path.join(mdp_folder, 'em.mdp')
        run_gromacs_simulation(top_file=cg_top_file, double_version=em_double_version, gro_file=tmp_gro, mdp_file=em_mdp,
                               task_name=task_name, output_folder=folder, table_file=table_file, em=True,
                               gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
        tmp_gro = os.path.join(folder, 'em.gro')

    #  annealing
    if anneal:
        task_name = 'anneal'
        folder = os.path.join(cg_simulation_folder, task_name)
        mkdir(folder)
        anneal_mdp = os.path.join(mdp_folder, 'anneal.mdp')
        run_gromacs_simulation(top_file=cg_top_file, gro_file=tmp_gro, mdp_file=anneal_mdp,
                               task_name=task_name, output_folder=folder, table_file=table_file,
                               gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
        tmp_gro = os.path.join(folder, 'anneal.gro')

    # equlibrium
    if eq:
        task_name = 'eq'
        folder = os.path.join(cg_simulation_folder, task_name)
        mkdir(folder)
        eq_mdp = os.path.join(mdp_folder, 'eq.mdp')
        run_gromacs_simulation(top_file=cg_top_file, gro_file=tmp_gro, mdp_file=eq_mdp,
                               task_name=task_name, output_folder=folder, table_file=table_file,
                               gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
        tmp_gro = os.path.join(folder, 'eq.gro')

    # production
    if prod:
        task_name = 'prod'
        folder = os.path.join(cg_simulation_folder, task_name)
        mkdir(folder)
        prod_mdp = os.path.join(mdp_folder, 'prod.mdp')
        run_gromacs_simulation(top_file=cg_top_file, gro_file=tmp_gro, mdp_file=prod_mdp,
                               task_name=task_name, output_folder=folder, table_file=table_file,
                               gpu_acceleration=gpu_acceleration, nt=nt, gpu_id=gpu_id)
        tmp_gro = os.path.join(folder, 'prod.gro')

    final_gro = tmp_gro

    cg_topology = os.path.join(folder, f'{task_name}.tpr')
    cg_trajectory = os.path.join(folder, f'{task_name}.trr')
    unwrap_traj = os.path.join(folder, f'{task_name}_unwrapped.trr')
    unwrap_trajectory(topology=cg_topology, trajectory=cg_trajectory, save_file=unwrap_traj)
    cg_trajectory = unwrap_traj

def write_spica_top(system_top, output_folder='spica_top'):
    mkdir(output_folder)
    for molecule in system_top['molecules']:
        mol_top = os.path.join(output_folder, f"{molecule['mol_name']}.top")
        with open(mol_top, 'w') as file:
            file.write(f"{'# atom_id':10} {'resname':<8} {'atomname':8} {'atomtype':<8} {'mass':<8} {'charge':<8} {'segid':<8}\n")
            for bead_id, bead in enumerate(molecule['types']):
                file.write(f"{'atom ' + str(bead_id +1):<10} {molecule['mol_name']:<8} {bead+str(bead_id):<8} {bead:<8} {molecule['mass'][bead_id]:<8} {molecule['charge'][bead_id]:<8} {molecule['mol_name']:<8}\n")

            if 'bond_parameters' in molecule and molecule['bond_parameters']:
                file.write('\n')
                for bond_id, bond in enumerate(molecule['bond_parameters'].keys()):
                    file.write(f"{'bond '+str(bond[0]+1):<10} {bond[1]+1:<8}\n")
            if 'angle_parameters' in molecule and molecule['angle_parameters']:
                file.write('\n')
                for angle_id, angle in enumerate(molecule['angle_parameters'].keys()):
                    file.write(f"{'angle '+str(angle[0]+1):<10} {angle[1]+1:<8} {angle[2]+1:<8}\n")


def update_spica_ff_file(system_top, original_spica_ff_file,  output_folder='spica_ff'):
    mkdir(output_folder)
    new_spica_ff_file = os.path.join(output_folder, 'spica_ff.prm')
    with open(original_spica_ff_file, 'r') as ofile:
        all_lines = ofile.readlines()
        for molecule in system_top['molecules']:
            if 'bond_parameters' in molecule and molecule['bond_parameters']:
                for bond in molecule['bond_parameters'].keys():
                    atom0 = molecule['types'][bond[0]]
                    atom1 = molecule['types'][bond[1]]
                    for idx, i in enumerate(all_lines):
                        parts = i.split()
                        if len(parts) < 5 or parts[0] != 'bond':
                            continue

                        if (parts[1], parts[2]) in [(atom0, atom1), (atom1, atom0)]:
                            if parts[-1] == 'updated':
                                warnings.warn(f"Bond type {'-'.join(parts[1:3])} already updated once. This type will be updated again")
                            parts[4] = str(molecule['bond_parameters'][bond][0]*10)  # sigma
                            parts[3] = str(molecule['bond_parameters'][bond][1]/(4.184*2.0*100))  # epsilon
                            new_line = '   '.join(parts) + '  # new updated\n'
                            all_lines[idx] = new_line
                            break

            if 'angle_parameters' in molecule and molecule['angle_parameters']:
                for angle in molecule['angle_parameters'].keys():
                    atom0 = molecule['types'][angle[0]]
                    atom1 = molecule['types'][angle[1]]
                    atom2 = molecule['types'][angle[2]]
                    for idx, i in enumerate(all_lines):
                        parts = i.split()
                        if len(parts) < 5 or parts[0] != 'angle':
                            continue

                        if (parts[1], parts[2], parts[3]) in [(atom0, atom1, atom2), (atom2, atom1, atom0)]:
                            if parts[-1] == 'updated':
                                warnings.warn(f"Angle type {'-'.join(parts[1:4])} already updated once. This type will be updated again.")
                            parts[5] = str(molecule['angle_parameters'][angle][0])  # eq_angle
                            parts[4] = str(molecule['angle_parameters'][angle][1]/4.184)  # k_angle
                            new_line = '   '.join(parts) + '  # new updated\n'
                            all_lines[idx] = new_line
                            break

            if 'dihedral_parameters' in molecule and molecule['dihedral_parameters']:
                for dihedral in molecule['dihedral_parameters'].keys():
                    atom0 = molecule['types'][[dihedral][0]]
                    atom1 = molecule['types'][[dihedral][1]]
                    atom2 = molecule['types'][[dihedral][2]]
                    atom3 = molecule['types'][[dihedral][3]]
                    for idx, i in enumerate(all_lines):
                        parts = i.split()
                        if len(parts) < 5 or parts[0] != 'dihedral':
                            continue
                        if (parts[1], parts[2], parts[3], parts[4]) in [(atom0, atom1, atom2, atom3), (atom3, atom2, atom1, atom0)]:
                            if parts[-1] == 'updated':
                                warnings.warn(f"Dihedral type {'-'.join(parts[1:5])} already updated once. This type will be updated again.")

                            parts[6] = str(molecule['dihedral_parameters'][dihedral][0])
                            parts[7] = str(molecule['dihedral_parameters'][dihedral][1])
                            parts[8] = str(molecule['dihedral_parameters'][dihedral][2])
                            new_line = '   '.join(parts) + '  # new updated\n'
                            all_lines[idx] = new_line
                            break

        for pair in system_top['lj_cross_terms'].keys():
            for idx, i in enumerate(all_lines):
                parts = i.split()
                if len(parts) < 5 or parts[0] != 'pair':
                    continue
                if (parts[1], parts[2]) in [pair, pair[::-1]]:
                    if parts[-1] == 'updated':
                        warnings.warn(
                            f"Pair type {'-'.join(parts[1:3])} already updated once. This type will be updated again.")
                    parts[4] = str(system_top['lj_cross_terms'][pair][1])  # epsilon
                    parts[5] = str(system_top['lj_cross_terms'][pair][0])  # sigma
                    new_line = '   '.join(parts) + '  # new updated\n'
                    all_lines[idx] = new_line
                    break

        with open(new_spica_ff_file, 'w') as f:
            f.write(''.join(all_lines))


def check_spica_tools():
    if not shutil.which('cg_spica'):
        raise RuntimeError("spica-tools is not installed. Please install it from https://github.com/SPICA-group/spica-tools.\n"
                           "And make sure the intallation of specific GROMACS version with the patch of SPICA force field from https://github.com/SPICA-group/gromacs-spica.")








def main():

    # input_folder = '/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/src/mapping_test/AA/'
    # top_file = os.path.join(input_folder, "force_match/test.top")
    # gro_file = os.path.join(input_folder, "force_match/cg.gro")
    # mdp_file = os.path.join(input_folder, "cg_mdp/eq_step0.mdp")
    # output = os.path.join(input_folder, "force_match")
    #
    # # run_gromacs_simulation(top_file=top_file, gro_file=gro_file, mdp_file=mdp_file, output_folder=output)
    #
    # system = {'molecules': [{'mol_name': 'A', 'model': 'SPICA', 'types': ['CT2', 'CM', 'CM'],
    #                         'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
    #                         'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
    #                         'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
    #                         'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 2},
    #                        {'mol_name': 'B', 'model': 'SPICA', 'types': ['OA', 'EO', 'OA'],
    #                         'id': [0, 1, 2], 'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
    #                         'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
    #                         'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
    #                         'angle_parameters': {(0, 1, 2): [180.0, 25.0]}, 'num_mols': 3}],
    #                  'lj_cross_terms': {('CM', 'CT2'): [2.7, 0.47],  ('CM', 'CM'): [4.5, 0.47], ('OA', 'EO'): [3.5, 0.47]},
    #                  'cgmodel': 'SPICA'}
    #
    # generate_top_file(system_top=system, save_file='/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/test_src/spica_top/system.top')

    # update_spica_ff_file(system_top=system, output_folder='/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/test_src/spica_top',
    #                      original_spica_ff_file='/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/test_src/spica_top/spica_protein_v2.prm')
    # write_spica_top(system_top=system, output_folder='/home/xiaoyedi/data/work/research/ML_DL/Autopara_CG/program/test_src/spica_top')
    # generate_top_file(system_top=system, save_file='test.top')
    pass




if __name__ == "__main__":
    main()
