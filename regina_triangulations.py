"""Build a Regina triangulation of a closed 3-manifold from three permutations.

Input convention ("double pyramids")
------------------------------------
A closed 3-manifold is described by n triangular bipyramids ("double
pyramids") labelled 0..n-1, together with three permutations
``perms = [pi_0, pi_1, pi_2]`` of {0, ..., n-1}.  Each bipyramid has three
top faces and three bottom faces, labelled f = 0, 1, 2.  The top face f of
bipyramid b is glued to the bottom face f of bipyramid ``pi_f[b]``.

This is the same data as a 4-coloured gem (see Sec. 3 of the paper): a pair of
gem vertices joined by a colour-0 edge is a pair of tetrahedra glued along
face 0, i.e. one bipyramid, and the remaining three colours give the face
gluings pi_1, pi_2, pi_3.  ``cycle3_scan.py`` uses this to turn each gem it
generates into a triple of permutations.

Each bipyramid is subdivided into 36 tetrahedra.  The cell labels used for the
subdivision come from ``_build_vertex_dict``: bipyramid centres, face centres,
edge midpoints and the original vertices.  The tetrahedra are glued with
identity face permutations, and Regina then simplifies the result.

Main entry point: :func:`build_regina_manifold`.
"""
from typing import List, Dict, Tuple
import numpy as np
import regina
from regina import Triangulation3


def get_cycles(perm) -> Tuple[np.ndarray, int]:
    # Returns a list mapping each element to the cycle it belongs to
    n = len(perm)
    cycles = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    n_cycles = 0
    for i in range(n):
        if visited[i]:
            continue
        j = i
        while not visited[j]:
            visited[j] = True
            cycles[j] = n_cycles
            j = perm[j]
        n_cycles += 1
    return cycles, n_cycles


def get_orbits(perms_list) -> Tuple[np.ndarray, int]:
    # Returns a list mapping each element to the orbit it belongs to
    # under the action of the list of permutations
    n = len(perms_list[0])
    orbits = np.zeros(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    n_orbits = 0
    path = []
    for i in range(n):
        if visited[i]:
            continue
        n_orbits += 1
        visited[i] = True
        orbits[i] = n_orbits-1
        path.append(i)
        while len(path) > 0:
            j = path[-1]
            visited[j] = True
            orbits[j] = n_orbits-1
            to_continue = False
            for p in perms_list:
                if not visited[p[j]]:
                    path.append(p[j])
                    to_continue = True
            if to_continue:
                continue

            path.pop()

    return orbits, n_orbits


def inv_perm(perm: np.array) -> np.ndarray:
    # Returns the inverse of a permutation
    n = len(perm)
    inv = np.zeros(n, dtype=int)
    for i in range(n):
        inv[perm[i]] = i
    return inv


def mul_perm(perm1: np.array, perm2: np.array) -> np.ndarray:
    # Multiplies two permutations
    n = len(perm1)
    perm = np.zeros(n, dtype=int)
    for i in range(n):
        perm[i] = perm2[perm1[i]]
    return perm


def pow_perm(perm: np.array, k: int) -> np.ndarray:
    # Raises a permutation to the power k
    n = len(perm)
    if k == 0:
        return np.arange(n)
    res = np.copy(perm)
    for _ in range(k-1):
        res = mul_perm(res, perm)
    return res


def _build_vertex_dict(perms: List[List[int]]) -> Dict:
    """Label the cells of the bipyramid complex. These labels become the
    vertices of the subdivided triangulation.

    Returns a dict with the following keys:
      'balls'  : one label per bipyramid (its centre).
      'faces'  : faces[f][b] = centre of the top face f of bipyramid b
                 (which is also the bottom face f of bipyramid pi_f[b]).
      'edges'  : edges[f][b] = the "outer" (equatorial) edge of face f.
                 Edges are identified along the cycles of pi_f.
                 edges[(i, j)][b] = the "inner" edge between faces i and j.
                 Edges are identified along the cycles of pi_i pi_j^{-1}.
      'points' : points[0][b] = the apex shared by the three top faces.
                 Points are identified along the orbits of
                 <pi_0 pi_1^{-1}, pi_0 pi_2^{-1}>.
                 points[(i, j)][b] = the equatorial vertex between faces i
                 and j. Points are identified along the orbits of
                 <pi_i, pi_j>.
    All labels are distinct integers.
    """
    vertex_dict = {}
    n_vertices = 0
    n = len(perms[0])

    # Add 3-cell vertices
    balls_list = np.arange(n)
    n_vertices += n
    vertex_dict['balls'] = balls_list

    # Add 2-cell vertices
    face_dict = {}
    for i in range(3):
        face_dict[i] = np.arange(n) + n_vertices
        n_vertices += n
    vertex_dict['faces'] = face_dict

    # Add 1-cell vertices
    edge_dict = {}
    # Outer edges
    for i in range(3):
        cycles, n_cycles = get_cycles(perms[i])
        edge_dict[i] = cycles + n_vertices
        n_vertices += n_cycles

    # Inner edges
    for ij in [(0, 1), (1, 2), (0, 2)]:
        perm = mul_perm(perms[ij[0]], inv_perm(perms[ij[1]]))
        cycles, n_cycles = get_orbits([perm])
        edge_dict[ij] = cycles + n_vertices
        n_vertices += n_cycles

    vertex_dict['edges'] = edge_dict

    # Add 0-cell vertices
    vertex_dict['points'] = {}
    for pt in [0, (0, 1), (0, 2), (1, 2)]:
        if pt == 0:
            perm1 = mul_perm(perms[0], inv_perm(perms[1]))
            perm2 = mul_perm(perms[0], inv_perm(perms[2]))
        else:
            perm1 = perms[pt[0]]
            perm2 = perms[pt[1]]
        cycles, n_cycles = get_orbits([perm1, perm2])
        vertex_dict['points'][pt] = cycles + n_vertices
        n_vertices += n_cycles

    return vertex_dict


def tup(i, j):
    # Returns a tuple with the smaller element first
    return (min(i, j), max(i, j))


def build_triangulation(perms: List[List[int]]) -> List:
    """
    Builds a triangulation from the given three permutations.
    Each permutation corresponds to a face of the double pyramid.

    Each tetrahedron is a tuple (ball, face, edge, point, to_inv), with one
    cell label from ``_build_vertex_dict`` per vertex.  Vertex k of the
    tetrahedron is entry k of the tuple.  ``to_inv`` marks whether the
    tetrahedron lies under a top face (False) or a bottom face (True).
    A gluing (t1, k, t2) glues face k of t1 (the face opposite vertex k) to
    face k of t2 with the identity vertex map.  In each gluing exactly one
    entry of the tuple changes.

    Args:
        perms (List[List[int]]): A list of three permutations, each of length n.
    Returns:
        Tuple[int, List[Tuple[int, int, int]]]: The number of tetrahedra (36 n)
        and a list of gluings (t1, face, t2).
    """

    assert len(perms) == 3, "There should be exactly three permutations"
    assert all(len(perm) == len(
        perms[0]) for perm in perms), "All permutations must have the same length"

    tetrahedra_list = []
    ind_to_tet_dict = {}
    tet_to_ind_dict = {}
    gluings = {}
    n = len(perms[0])
    vertex_dict = _build_vertex_dict(perms)
    inv_perms = [inv_perm(perm) for perm in perms]

    def add_tet(tet):
        ind_to_tet_dict[len(tetrahedra_list)] = tet
        tet_to_ind_dict[tet] = len(tetrahedra_list)
        if tet not in gluings:
            gluings[tet] = []
        tetrahedra_list.append(tet)

    # Iterate over balls
    for b in range(n):
        # Iterate over faces
        for f in range(3):
            # Top and bottom tetrahedra
            # For bottom tetrahedra, use inverse permutation
            # to obtain the ball the faces and edges belong to
            for to_inv in [True, False]:
                b2 = b if not to_inv else inv_perms[f][b]
                # Iterate over edges
                # Outer edges: one edge and two options for neighbours
                for adj_f in range(3):
                    if f == adj_f:
                        continue
                    tet = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                           vertex_dict['edges'][f][b2], vertex_dict['points'][tup(f, adj_f)][b2], to_inv)
                    add_tet(tet)
                    # Gluings for the top tetrahedra
                    if not to_inv:
                        # Glue to opposite tetrahedron, change only the ball
                        tet2 = (vertex_dict['balls'][perms[f][b]], vertex_dict['faces'][f][b],
                                vertex_dict['edges'][f][b], vertex_dict['points'][tup(f, adj_f)][b], not to_inv)
                        assert tet2 != tet
                        gluings[tet].append((0, tet2))
                        # Glue to tetrahedron below change only the face
                        tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][f][inv_perms[f][b]],
                                vertex_dict['edges'][f][b2], vertex_dict['points'][tup(f, adj_f)][b2], not to_inv)
                        assert tet2 != tet
                        gluings[tet].append((1, tet2))
                    # Glue to tetrahedron on the left, change only the point
                    if (adj_f-f) % 3 == 1:
                        adj2 = (adj_f+1) % 3
                        tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                                vertex_dict['edges'][f][b2], vertex_dict['points'][tup(f, adj2)][b2], to_inv)
                        assert tet2 != tet
                        gluings[tet].append((3, tet2))

                # Inner edges: two edges, two points for each edge
                for adj_f in range(3):
                    if f == adj_f:
                        continue

                    for pt in [0, (tup(f, adj_f))]:
                        tet = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                               vertex_dict['edges'][tup(f, adj_f)][b2], vertex_dict['points'][pt][b2], to_inv)
                        add_tet(tet)
                        # Glue to opposite tetrahedron, change only the ball
                        if not to_inv:
                            tet2 = (vertex_dict['balls'][perms[f][b]], vertex_dict['faces'][f][b],
                                    vertex_dict['edges'][tup(f, adj_f)][b2], vertex_dict['points'][pt][b2], not to_inv)
                            assert tet2 != tet
                            gluings[tet].append((0, tet2))
                        # Glue to tetrahedron below
                        # For the tetrahedron adjacent to the point 0, change only the point
                        if pt == 0:
                            tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                                    vertex_dict['edges'][tup(f, adj_f)][b2], vertex_dict['points'][tup(f, adj_f)][b2], to_inv)
                            assert tet2 != tet
                            gluings[tet].append((3, tet2))
                        # For the tetrahedron adjacent to the point (f, adj_f), change only the edge
                        else:
                            tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                                    vertex_dict['edges'][f][b2], vertex_dict['points'][pt][b2], to_inv)
                            assert tet2 != tet
                            gluings[tet].append((2, tet2))

                        # For tetrahedra on the right of the face, glue to tetrahedra on the left
                        if (adj_f-f) % 3 == 1:
                            # Gluing that changes only the face
                            face_b = b if not to_inv else inv_perms[adj_f][b]
                            tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][adj_f][face_b],
                                    vertex_dict['edges'][tup(f, adj_f)][b2], vertex_dict['points'][pt][b2], to_inv)
                            assert tet2 != tet
                            gluings[tet].append((1, tet2))
                            # For top tetrahedra, glue that changes only the edge
                            adj2 = (adj_f+1) % 3
                            if pt == 0:
                                tet2 = (vertex_dict['balls'][b], vertex_dict['faces'][f][b2],
                                        vertex_dict['edges'][tup(f, adj2)][b2], vertex_dict['points'][pt][b2], to_inv)
                                gluings[tet].append((2, tet2))

    # Simplify gluings
    simp_gluings_list = [(tet_to_ind_dict[tet1], f, tet_to_ind_dict[tet2])
                         for tet1, gluings_list in gluings.items() for f, tet2 in gluings_list]
    simp_gluings_set = set(simp_gluings_list)
    # Check that all gluings are unique
    assert len(set(simp_gluings_list)) == len(simp_gluings_list)
    # Check that all faces are glued
    simp_and_faces = set([(tet, f) for tet, f, _ in simp_gluings_list] +
                         [(tet, f) for _, f, tet in simp_gluings_list])
    for tet in range(len(tetrahedra_list)):
        for f in range(4):
            assert (
                tet, f) in simp_and_faces, f"Face {f} of tetrahedron {tet} is not glued"

    # Check that no face is glued twice
    inv_gluing_set = set([(tet2, f, tet1)
                         for tet1, f, tet2 in simp_gluings_list])
    assert len(simp_gluings_set & inv_gluing_set) == 0

    return len(tetrahedra_list), simp_gluings_list


def build_regina_manifold(perms):
    """ Builds a Regina manifold from the given permutations.

    The 36 n-tetrahedron triangulation from :func:`build_triangulation` is
    passed to Regina and reduced with ``simplifyToLocalMinimum``.

    Args:
        perms (List[List[int]]): A list of three permutations, each of length n.
    Returns:
        Triangulation3: A Regina manifold triangulation.
    """
    n, triang = build_triangulation(perms)
    triang = [(t1, f, t2, regina.Perm4([0, 1, 2, 3])) for t1, f, t2 in triang]
    manifold = Triangulation3.fromGluings(n, triang)
    manifold.simplifyToLocalMinimum()
    return manifold


if __name__ == "__main__":
    # Minimal smoke test: a 4-ball example, printed with its Regina isoSig.
    perms = [[0, 3, 2, 1], [2, 1, 0, 3], [1, 2, 3, 0]]
    manifold = build_regina_manifold(perms)
    print(perms)
    print(f"Is sphere: {manifold.isSphere()}")
    print(f"Is closed: {manifold.isClosed()}")
    manifold.intelligentSimplify()
    print(manifold.isoSig())
