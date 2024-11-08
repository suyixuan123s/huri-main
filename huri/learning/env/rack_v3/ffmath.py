import numpy as np

def rotation_matrix(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    if dot == 1:
        return np.identity(4)
    elif dot == -1:
        return -np.identity(4)
    else:
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)
        v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.identity(4)
        R[:3, :3] = np.identity(3) + v_x + np.dot(v_x, v_x) * ((1 - c) / (s ** 2))
        return R