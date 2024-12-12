

opt_settings = {
    'coarse-grained model': 'MARTINI2',
    'opt bond': True,
    'opt angle': True,
    'opt dihedral': False,

    'eq bond from fg': True,
    'eq angle from fg': True,
    'eq dihedral from fg': False,

    'fine-grained topology file': '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/POPC/AA/prod/prod.tpr',
    'fine-grained trajectory file': '/home/xiaoyedi/data/work/research/ML&DL/Autopara_CG/program/src/mapping_test/POPC/AA/prod/prod.gro',

    'molecular topology list save path': './mapping_test/POPC/mapping/',

    # 'region expand ratio of sigma': 0.1,
    # 'region expand ratio of epsilon': 0.1,
    # 'region expand ratio of k_bond': 0.1,
    # 'region expand ratio of k_angle': 0.1,
    # 'region expand ratio of k_dihedral': None,

    'optimization folder': './mapping_test/POPC',
    'cg 0_setp mdp': './mapping_test/POPC/mdp/force_match.mdp',
    'cg simulation mdp folder': './mapping_test/POPC/mdp',
    'initial cg gro': './mapping_test/POPC/packmol/mix.gro',
    'index file': './mapping_test/POPC/index.ndx',

    'weight of force match': None,
    'weight of Ur Boltzmann inversion': 0.1,
    'weight of density': None,
    'experiment value of density': 880,

    'fg rdf pairs list': [{'selection': ['resname POPC', 'resname POPC'],
                           'groups': [[[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]],
                                      [[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]],
                           'num_atom':[134, 134], 'method':'com'}],


     'cg rdf pairs list': [{'selection': ['resname POPC', 'resname POPC'],
                           'groups': [[[6]],  [[6]]],
                           'num_atom':[12, 12], 'method':'com'}],
    'rdf bin width': 0.02,  # Å (ångström)
    'rdf cutoff': 50,   # Å (ångström)

    'Temperature': 300,  # K

    'opt method': 'PSO',
    'max ilteration': 4,
    'MPI': True,
    'MPI Processes': 4,

    'nt': 8,
    'gpu id': 0,
    'em': True,
    'anneal': False,
    'eq': True,
    'prod': True

}


# opt_term_parse is a dictionary used to specify which parameters in a molecular system need to be optimized and the extent of their optimization.
opt_term_parse = {
    # The 'molecules' key contains a list of dictionaries, each representing a molecule with specific parameters to be optimized.
    'molecules': [
        {
            # 'mol_name' specifies the name of the molecule. Here, '12oh' is the name of the first molecule.
            'mol_name': 'POPC',
            # 'charge' is a list representing the optimization factors for the charge parameters of the molecule.
            # A value of 0 indicates that the corresponding charge parameter will not be optimized.
            # 'charge': [0, 0, 0],  # No optimization for charge parameters in this molecule.
            # 'mass' is a list representing the optimization factors for the mass parameters of the molecule.
            # Here, all values are 0, indicating no optimization for mass parameters.
            # 'mass': [0, 0, 0]
            # 'bond_parameters': {(0, 1): [0, 0.1], (1, 2): [0, 0.1]},
            # 'angle_parameters': {(4, 5, 8): [0, 0.1], (5, 8, 9): [0, 0.1]},

        }],
    'lj_cross_terms':{('C3', 'NA'): [0.1, 0.1], ('C3', 'QA'): [0.1, 0.1],
                      ('C3', 'P4'): [0.1, 0.1], ('C3', 'BP4'): [0.1, 0.1],
                      ('C3', 'Q0'): [0.1, 0.1], ('C3', 'C1'): [0.1, 0.1],
                      ('NA', 'QA'): [0.1, 0.1], ('NA', 'P4'): [0.1, 0.1],
                      ('NA', 'BP4'): [0.1, 0.1], ('NA', 'Q0'): [0.1, 0.1],
                      ('NA', 'C1'): [0.1, 0.1], ('QA', 'P4'): [0.1, 0.1],
                      ('QA', 'BP4'): [0.1, 0.1], ('QA', 'Q0'): [0.1, 0.1],
                      ('QA', 'C1'): [0.1, 0.1], ('P4', 'BP4'): [0.1, 0.1],
                      ('P4', 'Q0'): [0.1, 0.1], ('P4', 'C1'): [0.1, 0.1],
                      ('BP4', 'Q0'): [0.1, 0.1], ('BP4', 'C1'): [0.1, 0.1],
                      ('Q0', 'C1'): [0.1, 0.1], ('C3', 'C3'): [0.1, 0.1],
                      ('NA', 'NA'): [0.1, 0.1], ('QA', 'QA'): [0.1, 0.1],
                      ('P4', 'P4'): [0.1, 0.1], ('BP4', 'BP4'): [0.1, 0.1],
                      ('Q0', 'Q0'): [0.1, 0.1], ('C1', 'C1'): [0.1, 0.1]}
}