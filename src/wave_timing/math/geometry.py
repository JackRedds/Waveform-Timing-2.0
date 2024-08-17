import numpy as np


def dot_product(vec1, vec2):
    if len(vec1.shape) == 1 and len(vec2.shape) == 1:
        return np.sum(vec1 * vec2)

    return np.sum(vec1 * vec2, axis=1)

def angle_between(vec1: np.ndarray, vec2: np.ndarray):
    dot = dot_product(vec1, vec2)
    mag1 = np.sqrt(dot_product(vec1, vec1))
    mag2 = np.sqrt(dot_product(vec2, vec2))
    angle = np.arccos(dot / (mag1 * mag2))
    angle = np.degrees(angle)
    return angle


def sc_to_v1234_coordinates(vector: np.ndarray):
    theta = np.radians(55)
    rotation_matrix = np.array([[np.cos(theta),  np.sin(theta), 0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0,             0,             -1]])
    return np.dot(vector, rotation_matrix)
