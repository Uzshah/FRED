"""
Icosahedral geometry utilities for spherical patch sampling.
"""

import math
from typing import Tuple
import numpy as np


def build_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the vertices and faces of a regular icosahedron.

    Returns:
        V: Vertex positions (12, 3), normalized to unit sphere
        F: Face indices (20, 3)
    """
    phi = (1.0 + 5.0**0.5) / 2.0
    a, b = 1.0, phi
    V = np.array(
        [
            [-a, b, 0],
            [a, b, 0],
            [-a, -b, 0],
            [a, -b, 0],
            [0, -a, b],
            [0, a, b],
            [0, -a, -b],
            [0, a, -b],
            [b, 0, -a],
            [b, 0, a],
            [-b, 0, -a],
            [-b, 0, a],
        ],
        dtype=np.float64,
    )
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )
    return V, F


def subdivide_geodesic(V, F, level: int):
    """
    Subdivide the mesh geodesically by splitting each edge at its midpoint
    and projecting to the unit sphere.

    Args:
        V: Vertices (N, 3)
        F: Faces (M, 3)
        level: Number of subdivision iterations

    Returns:
        V_new: Subdivided vertices
        F_new: Subdivided faces
    """
    if level <= 0:
        return V, F

    def edge_key(i, j):
        return (i, j) if i < j else (j, i)

    for _ in range(level):
        midpoint_cache = {}
        new_faces = []
        V_list = V.tolist()

        def add_vertex(a):
            V_list.append(a.tolist())
            return len(V_list) - 1

        for f in F:
            i0, i1, i2 = map(int, f)
            v0, v1, v2 = V[i0], V[i1], V[i2]

            def midpoint(i, j, vi, vj):
                k = edge_key(i, j)
                if k in midpoint_cache:
                    return midpoint_cache[k]
                m = (vi + vj) * 0.5
                m = m / np.linalg.norm(m)
                idx = add_vertex(m)
                midpoint_cache[k] = idx
                return idx

            a = midpoint(i0, i1, v0, v1)
            b = midpoint(i1, i2, v1, v2)
            c = midpoint(i2, i0, v2, v0)
            new_faces.extend([[i0, a, c], [a, i1, b], [c, b, i2], [a, b, c]])

        V = np.array(V_list, dtype=np.float64)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        F = np.array(new_faces, dtype=np.int32)

    return V, F


def tri_face_centers(V, F):
    """
    Compute the center of each triangular face, projected to unit sphere.

    Args:
        V: Vertices (N, 3)
        F: Faces (M, 3)

    Returns:
        Centers (M, 3), normalized to unit sphere
    """
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def face_rotation_from_center(c):
    """
    Compute a rotation matrix that aligns the +Z axis with the face center.

    Args:
        c: Face center direction (3,)

    Returns:
        R: Rotation matrix (3, 3) where R[:, 2] = c/||c||
    """
    z = c / np.linalg.norm(c)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, z], axis=1)


def max_angular_radius(c, tri):
    """
    Compute the maximum angular distance from face center to triangle vertices.

    Args:
        c: Face center (3,)
        tri: Triangle vertices (3, 3)

    Returns:
        Maximum angular radius in radians
    """
    c = c / np.linalg.norm(c)
    ang = []
    for i in range(3):
        v = tri[i] / np.linalg.norm(tri[i])
        dot = float(np.clip(np.dot(c, v), -1.0, 1.0))
        ang.append(math.acos(dot))
    return max(ang)
