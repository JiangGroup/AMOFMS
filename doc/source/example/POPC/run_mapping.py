from AMOFMS.FGMappingToCG import MappingToCGfromDSGPM_TP, MappingItem
from AMOFMS.CGForceFieldParameters import InitCGForceFieldParameters, generate_system_top
from opt_settings import opt_settings
import os

molecules = [
    {
        'molecule name': 'POPC',
        'molecular form': 'sml',  # sml or pdb; if use .pdb, please add 'pdb file root'='path/to/your/pdb/file'
        'smiles': 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCC=CCCCCCCCC',
        'number of mapping beads': 12,
        'mapping output folder': './mapping_test/POPC/mapping/POPC',
        'number of atom in molecule': 134,  # including H atom
        'number of molecule in system': 384,
        'residue name': 'POPC',

        # split 134 atom to 12 list corresponding to 12 cg bead(group)
        'aa_group_list': [[131, 132, 133, 128, 129, 130, 125, 126, 127, 121, 122, 123, 124],
                          [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
                          [100, 101, 102, 103, 104, 105, 106, 107, 108],
                          [41, 42, 43, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                          [35, 36, 37, 38, 39, 40],
                          [27, 28, 29, 30, 31],
                          [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          [32, 33, 34, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
                          [56, 57, 58, 59, 60, 61, 62, 63, 64, 65],
                          [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77],
                          [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
                          ]
    }
]

all_cg_molecule_top = []

for molecule in molecules:
    mol_mapping = MappingToCGfromDSGPM_TP(mol_name=molecule['molecule name'],
                                         mol_form=molecule['molecular form'],
                                         smiles=molecule['smiles'],
                                         CG_num_bead=molecule['number of mapping beads'],
                                         output_dir=molecule['mapping output folder'],
                                         CGmodel=opt_settings['coarse-grained model']
                                         )

    FF_parameter_item = InitCGForceFieldParameters(mapping_item=mol_mapping.get_mapping_item(),
                                                   num_mols=molecule['number of molecule in system'],
                                                   setbond=opt_settings['opt bond'],
                                                   setangle=opt_settings['opt angle'],
                                                   setdihedral=opt_settings['opt dihedral'],
                                                   )

    if opt_settings['eq bond from fg']:
        FF_parameter_item.set_bond_parameters(top=opt_settings['fine-grained topology file'],
                                              traj=opt_settings['fine-grained trajectory file'],
                                              num_atom_with_H=molecule['number of atom in molecule'],
                                              num_mol=molecule['number of molecule in system'],
                                              res_name=molecule['residue name'],
                                              group_list=molecule['aa_group_list'])

    if opt_settings['eq angle from fg']:
        FF_parameter_item.set_angle_parameters(top=opt_settings['fine-grained topology file'],
                                               traj=opt_settings['fine-grained trajectory file'],
                                               num_atom_with_H=molecule['number of atom in molecule'],
                                               num_mol=molecule['number of molecule in system'],
                                               res_name=molecule['residue name'],
                                               group_list=molecule['aa_group_list']
                                               )
    if opt_settings['eq dihedral from fg']:
        FF_parameter_item.set_dihedral_parameters(top=opt_settings['fine-grained topology file'],
                                                  traj=opt_settings['fine-grained trajectory file'],
                                                  num_atom_with_H=molecule['number of atom in molecule'],
                                                  num_mol=molecule['number of molecule in system'],
                                                  res_name=molecule['residue name'],
                                                  group_list=molecule['aa_group_list']
                                                  )
    cg_molecule_top = FF_parameter_item.generate_cg_ff_parameters_item()
    all_cg_molecule_top.append(cg_molecule_top)

# system_top = generate_system_top(mols=all_cg_molecule_top, num_mols=None)

with open(os.path.join(opt_settings['molecular topology list save path'], 'all_molecule_topology.py'), 'a+') as file:
    file.write('molecule_topology_list = [\n')
    for i in all_cg_molecule_top:
        file.write(str(i)+',\n')
    file.write(']')

