import os


tools_dir = os.path.dirname(__file__)

martini2_pathway = os.path.join(tools_dir, 'force_field/martini_v2.2.itp')
martini3_pathway = os.path.join(tools_dir, 'force_field/martini_v3.0.0.itp')
spica_pathway = os.path.join(tools_dir, 'force_field/spica_protein_v2.prm')


def query_interaction_params(particle_pair, pair_types, interaction_types, cgmodel='MARTINI2'):
    """
    query epsilon and sigma of the particle pair

    :param pair_types: A dictionary containing particle pairs and their interaction types.
    :param interaction_types: A dictionary containing interaction types and parameters (epsilon, sigma).
    :param particle_pair: The particle pair to be queried.
    :return: A tuple of epsilon and sigma values, or None if not found
    """

    # Query the interaction type of a particle pair
    interaction_type = pair_types.get(particle_pair)

    # If the interaction type exists, query the epsilon and sigma parameters
    if interaction_type:
        params = interaction_types.get(interaction_type)
        epsilon = params.get('epsilon')
        sigma = params.get('sigma')
        return epsilon, sigma
    else:
        print(f"The interaction type for particle pair {particle_pair} was not found.")
        return None


def extract_lj_params(particle_pair, cgmodel='MARTINI2'):
    """
    Extracts Lennard-Jones parameters for a given particle pair from a file.

    :param file_path: Path to the file containing the parameters.
    :param particle_pair: A tuple of the particle pair, e.g., ('P5', 'P6').
    :return: A dictionary with 'epsilon' and 'sigma' values, or None if the pair is not found.
    """

    if cgmodel == 'MARTINI2':
        ff_path = martini2_pathway
    elif cgmodel == 'MARTINI3':
        ff_path = martini3_pathway
    elif cgmodel == 'SPICA':
        ff_path = spica_pathway
    else:
        raise ValueError(f"Unsupported CG model: {cgmodel}")

    with open(ff_path, 'r') as file:
        if cgmodel in ['MARTINI2', 'MARTINI3']:
            particle_pair = tuple(map(str.upper, particle_pair))
            for line in file:
                if line.strip().startswith('[ nonbond_params ]'):
                    continue  # Skip the header line
                parts = line.split()
                if len(parts) < 5:
                    continue  # Skip invalid lines
                particles = (parts[0], parts[1])
                particles = tuple(map(str.upper, particles))

                if particles == particle_pair or particles == tuple(reversed(particle_pair)):
                    C6 = float(parts[3])
                    C12 = float(parts[4])
                    if cgmodel == 'MARTINI2':
                        sigma = (C12 / C6) ** (1.0/6.0)
                        epsilon = C6 ** (2.0) / 4 / C12
                    elif cgmodel == 'MARTINI3':
                        sigma = C6  # nm
                        epsilon = C12  # kJ/mol
                    return round(sigma, 4), round(epsilon, 4)

        elif cgmodel == 'SPICA':
            for line in file:
                words = line.split()
                if len(words) < 5:
                    continue
                if words[0] != 'pair':
                    continue
                # Check if both elements of the pair are in the line
                if (words[1], words[2]) in [particle_pair, particle_pair[::-1]]:
                    epsilon = float(words[4])  # Angstrom
                    sigma = float(words[5])   # kcal/mol
                    return sigma, epsilon

    raise ValueError("lj term not found in the force field file")


def extract_bond_params(bond, cgmodel='MARTINI2'):

    if cgmodel == 'MARTINI2':
        k_bond = 1250  # kJ mol-1 nm-2
        initial_letters = tuple(sorted(['R' if i[0] not in ['S', 'T'] else i[0] for i in bond]))
        if initial_letters == ('S', 'S'):
            eq_bond = 0.43
        elif initial_letters in [('S', 'T'), ('T', 'S')]:
            eq_bond = 0.43
        elif initial_letters == ('T', 'T'):
            eq_bond = 0.32
        elif initial_letters == ('R', 'R'):
            eq_bond = 0.47
        elif initial_letters in [('R', 'S'), ('S', 'R')]:
            eq_bond = 0.47
        elif initial_letters in [('R', 'T'), ('T', 'R')]:
            eq_bond = 0.47  # nm
        else:
            raise Exception('Invalid bond type')
        return eq_bond, k_bond

    elif cgmodel == 'MARTINI3':
        k_bond = 3800  # kJ mol-1 nm-2
        initial_letters = tuple(sorted(['R' if i[0] not in ['S', 'T'] else i[0] for i in bond]))
        if initial_letters == ('S', 'S'):
            eq_bond = 0.41
        elif initial_letters in [('S', 'T'), ('T', 'S')]:
            eq_bond = 0.365
        elif initial_letters == ('T', 'T'):
            eq_bond = 0.34
        elif initial_letters == ('R', 'R'):
            eq_bond = 0.47
        elif initial_letters in [('R', 'S'), ('S', 'R')]:
            eq_bond = 0.43
        elif initial_letters in [('R', 'T'), ('T', 'R')]:
            eq_bond = 0.395  # nm
        else:
            raise Exception('Invalid bond type')
        return eq_bond, k_bond

    elif cgmodel == 'SPICA':
        with open(spica_pathway, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) < 5:
                    continue
                if words[0] != 'bond':
                    continue
                # Check if both elements of the pair are in the line
                if (words[1], words[2]) in [bond, bond[::-1]]:
                    eq_bond = float(words[4])/10.0  # nm
                    k_bond = float(words[3])*4.184*2.0*100  # kJ mol-1 nm-2
                    return eq_bond, k_bond

            raise ValueError("Bond not found in the force field file")
    else:
        raise ValueError(f"Unsupported CG model: {cgmodel}")


def extract_angle_params(angle, cgmodel='MARTINI2'):
    if cgmodel == 'MARTINI2':
        k_angle = 25.0  # kJ mol-1
        eq_angle = 180.0  # degree
        return eq_angle, k_angle
    elif cgmodel == 'MARTINI3':
        k_angle = 35.0  # kJ mol-1
        eq_angle = 180.0  # degree
        return eq_angle, k_angle
    elif cgmodel == 'SPICA':
        with open(spica_pathway, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) < 6:
                    continue
                if words[0] != 'angle':
                    continue
                # Check if both elements of the pair are in the line
                if (words[1], words[2], words[3]) in [angle, angle[::-1]]:
                    k_angle = float(words[4])*4.184  # kJ/mol
                    eq_angle = float(words[5])  # degree
                    return eq_angle, k_angle

            raise ValueError("Angle not found in the force field file")
    else:
        raise ValueError(f"Unsupported CG model: {cgmodel}")


def extract_dihedral_params(dihedral, cgmodel='MARTINI2'):
    if cgmodel == 'MARTINI2':
        k_dihedral = 25.0  # kJ mol-1
        eq_dihedral = 180.0  # degree
        return eq_dihedral, k_dihedral
    elif cgmodel == 'MARTINI3':
        k_dihedral = 35.0  # kJ mol-1
        eq_dihedral = 180.0  # degree
        return eq_dihedral, k_dihedral
    else:
        raise ValueError(f"Unsupported CG model: {cgmodel}")

def main():
    a = extract_lj_params(('SN3R', 'TP1'), cgmodel='MARTINI3')
    b =extract_angle_params(('EO', 'EO', 'CM'), cgmodel='SPICA')

    print(b)

if __name__ == "__main__":
    main()