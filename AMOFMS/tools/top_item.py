import itertools






















def main():
    mol_A = {'mol_name': 'A', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
             'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
             'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}

    mol_B = {'mol_name': 'B', 'model': 'MARTINI2', 'types': ['C1', 'C1', 'P1'], 'id': [0, 1, 2],
             'charge': [0.0, 0.0, 0.0], 'mass': [57.1146, 56.1067, 59.0869],
             'lj_parameters': {('P1', 'C1'): [2.7, 0.47], ('P1', 'P1'): [4.5, 0.47], ('C1', 'C1'): [3.5, 0.47]},
             'bond_parameters': {(0, 1): [0.47, 1250.0], (1, 2): [0.47, 1250.0]},
             'angle_parameters': {(0, 1, 2): [180.0, 25.0]}}

    generate_system_top(mols=[mol_A, mol_B], num_mols=[2, 3])




if __name__ == "__main__":
    main()