import numpy as np


def compute_angle(v1, v2, v3):
    """Calculate the angle defined by three vectors."""
    vec1 = v1 - v2
    vec2 = v3 - v2
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)


def compute_dihedral(v1, v2, v3, v4):
    """Calculate the dihedral angle defined by four vectors."""
    p1 = np.cross(v2 - v1, v3 - v2)
    p2 = np.cross(v3 - v2, v4 - v3)
    cos_phi = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
    phi = np.arccos(np.clip(cos_phi, -1, 1))
    return np.degrees(phi)


def square_diff_of_elements_in_2D_list(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same size.")

    result = []
    for row1, row2 in zip(list1, list2):
        if len(row1) != len(row2):
            raise ValueError("All rows must have the same length.")

        row_diff = [abs(a - b) for a, b in zip(row1, row2)]
        result.append(np.mean(row_diff))

    return np.mean(result)
