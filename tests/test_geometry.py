import numpy as np
from fred.geometry import build_icosahedron, subdivide_geodesic, tri_face_centers

def test_geometry():
    vertices, faces = build_icosahedron()
    assert vertices.shape == (12, 3)
    assert faces.shape == (20, 3)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0)
    vertices, faces = subdivide_geodesic(vertices, faces, 1)
    assert faces.shape == (80, 3)
    np.testing.assert_allclose(np.linalg.norm(tri_face_centers(vertices, faces), axis=1), 1.0)
