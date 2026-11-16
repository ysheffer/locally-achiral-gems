"""Search for locally-achiral 3-gems with a 3-cycle residue (paper, Sec. 3.3).

Reference: Y. Sheffer, "Universal entanglement probes of topological order and
locally-achiral manifolds", arXiv:2606.18361, Sec. 3.3 ("Construction of
locally-achiral 3-gems").

What the script does
--------------------
A 3-gem is a 4-edge-coloured bipartite graph.  Colours here are 0..3; the
paper uses 1..4, so code colour c is paper colour c+1.  The search runs
as follows.

1. The colour {0,1} residue consists of exactly three bicoloured cycles of
   lengths n1 <= n2 <= n3, with n1 + n2 + n3 = 4k (paper: g_12 = 3).
2. By Lemma 3.6 the colour-2 edges are then fixed up to colouring: m_ij of
   them join cycle i to cycle j, with m_ij given by Eq. (10),
       m_12 = (n1+n2-n3)/2,  m_13 = (n1+n3-n2)/2,  m_23 = (n2+n3-n1)/2.
   The resulting 3-coloured graph is a gem of S^2.  Because the number of
   vertices is 4k, it is reflection-positive, which makes the gem
   locally-achiral.
3. The colour-3 edges copy the colour-2 pattern, with the endpoints on
   cycle i shifted by 2*q_i + par steps.  Only choices with
   g_23 = 3 are kept (paper: g_34 = 3).  With this condition all
   complementary residue counts match (g_01 = g_23, g_02 = g_13,
   g_03 = g_12), so the graph is a valid 3-gem (Theorem 3.3).
4. Each gem becomes three permutations (``_as_perms``) and then a Regina
   triangulation (``regina_triangulations.build_regina_manifold``).
   The script prints one representative per distinct hyperbolic volume,
   with its SnapPy/Regina isoSig, volume and H_1.

Usage
-----
    python cycle3_scan.py --min-n 14 --max-n 20

The isoSigs that are printed can be looked up in Regina's census, or loaded
into SnapPy, e.g. ``snappy.Manifold(isosig).identify()``.  They can also be
compared with Dehn fillings of the links in ``link_pd``.
"""
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import regina
from regina_triangulations import build_regina_manifold
from regina_orient_rev_scan import has_orientation_reversing_automorphism
import snappy


@dataclass
class FourGraph:
    """A 4-coloured gem found by the search.

    G0..G3 are the colour-c edges, stored as symmetric dicts vertex -> vertex.
    A vertex (i, j) is the j-th vertex along the i-th cycle of the {0,1}
    residue (i = 0, 1, 2 for the cycles of length n1, n2, n3).
    par, q1, q2, q3 are the parameters that fix the colour-3 edges.
    pi1, pi2, pi3 are the face-gluing permutations used by
    ``build_regina_manifold`` (see ``_as_perms``).
    """
    G0:Dict[Tuple[int, int], Tuple[int, int]]
    G1: Dict[Tuple[int, int], Tuple[int, int]]
    G2: Dict[Tuple[int, int], Tuple[int, int]]
    G3: Dict[Tuple[int, int], Tuple[int, int]]
    par: int
    q1: int
    q2: int
    q3: int
    pi1: list[int]
    pi2: list[int]
    pi3: list[int]


def _bidirect(d: dict):
    """Given a dict mapping (a,b) to (c,d), add the reverse mapping (c,d) to (a,b)."""
    for k, v in list(d.items()):
        if v not in d:
            d[v] = k
    return d


def classify_hyperbolic_candidate(manifold, seen_isosigs: set[str]) -> Tuple[bool, str, Optional[str], Optional[float]]:
    """Classify whether a Regina triangulation is hyperbolic using staged filters.

    Returns (is_hyperbolic, status, isoSig, volume).
    """
    manifold.intelligentSimplify()

    if not manifold.isClosed():
        return False, "skip-open", None, None
    if manifold.isSphere():
        return False, "non-hyperbolic-sphere", None, None
    if not manifold.isIrreducible():
        return False, "non-hyperbolic-reducible", None, None

    try:
        if manifold.fundamentalGroup().identifyAbelian():
            return False, "non-hyperbolic-abelian-pi1", None, None
    except Exception:
        # Keep candidate when group identification is unavailable.
        pass

    iso = manifold.isoSig()
    if iso in seen_isosigs:
        return False, "duplicate-isosig", iso, None
    seen_isosigs.add(iso)

    try:
        snappy_manifold = snappy.Manifold(iso)
    except Exception as exc:
        return False, f"snappy-init-failed: {exc}", iso, None

    try:
        solution_type = str(snappy_manifold.solution_type()).lower()
    except Exception:
        solution_type = ""

    is_geometric = (
        "geometric" in solution_type
        or "all tetrahedra positively oriented" in solution_type
        or "contains negatively oriented tetrahedra" in solution_type
    )
    if not is_geometric:
        return False, f"no-geometric-solution ({solution_type})", iso, None

    try:
        verified = snappy_manifold.verify_hyperbolicity()
        if verified:
            return True, "hyperbolic-verified", iso, float(snappy_manifold.volume())
    except Exception:
        pass

    try:
        volume = float(snappy_manifold.volume())
    except Exception:
        return False, "volume-failed", iso, None

    if volume > 0:
        return True, "hyperbolic-numerical", iso, volume
    return False, f"nonpositive-volume ({volume})", iso, volume


def approximate_min_heegaard_genus(manifold, mode: str = "fast") -> str:
    """Return a bounds-style estimate for the minimal Heegaard genus.

    Output format examples:
    - "0 (exact)"
    - "1 (exact)"
    - "2 (exact via matching bounds)"
    - "1..3 (bounds)"
    - ">=2 (lower bound only)"
    - "unknown"
    """
    try:
        if not manifold.isClosed() or not manifold.isOrientable() or not manifold.isConnected():
            return "unknown"

        lower_bound = 0
        if manifold.isSphere():
            return "0 (exact)"

        try:
            group = manifold.fundamentalGroup()
            lower_bound = max(lower_bound, group.abelianRank())
            # Closed orientable non-spherical manifolds with abelian pi1
            # have Heegaard genus 1 (lens spaces and S^2 x S^1).
            if group.identifyAbelian() and lower_bound <= 1:
                return "1 (exact)"
        except Exception:
            pass

        upper_bound: Optional[int] = None
        list_types = (regina.NS_VERTEX,) if mode == "fast" else (
            regina.NS_VERTEX,
            regina.NS_FUNDAMENTAL,
        )

        search_triangulations = [manifold]
        if mode == "deeper":
            # Additional pass on a simplified copy for tighter upper bounds.
            simpler = type(manifold)(manifold)
            simpler.intelligentSimplify()
            search_triangulations.append(simpler)

        for tri in search_triangulations:
            for list_type in list_types:
                surfaces = regina.NormalSurfaces(
                    tri,
                    regina.NS_STANDARD,
                    list_type | regina.NS_EMBEDDED_ONLY,
                )
                for surface in surfaces:
                    if not surface.isConnected() or not surface.isOrientable():
                        continue
                    if surface.hasRealBoundary() or not surface.isSplitting():
                        continue

                    cut = surface.cutAlong()
                    components = cut.triangulateComponents()
                    if len(components) != 2:
                        continue

                    genera = [component.recogniseHandlebody()
                              for component in components]
                    if genera[0] < 0 or genera[1] < 0 or genera[0] != genera[1]:
                        continue
                    split_genus = int(genera[0])
                    if upper_bound is None or split_genus < upper_bound:
                        upper_bound = split_genus

        if upper_bound is None:
            return f">={lower_bound} (lower bound only)" if lower_bound > 0 else "unknown"
        if upper_bound == lower_bound:
            return f"{upper_bound} (exact via matching bounds)"
        return f"{lower_bound}..{upper_bound} (bounds)"
    except Exception:
        return "unknown"


def hyperbolic_invariants(manifold, iso_sig: str, cache: dict[str, Tuple[str, Optional[bool], str]], heegaard_mode: str) -> Tuple[str, Optional[bool], str]:
    """Return (H1, orient-rev automorphism, Heegaard genus estimate)."""
    if iso_sig in cache:
        return cache[iso_sig]

    try:
        h1 = str(snappy.Manifold(iso_sig).homology())
    except Exception:
        try:
            h1 = str(manifold.homology())
        except Exception as exc:
            h1 = f"unknown ({exc})"

    try:
        has_orient_rev = has_orientation_reversing_automorphism(manifold)
    except Exception:
        has_orient_rev = None

    heegaard_genus_estimate = approximate_min_heegaard_genus(
        manifold, mode=heegaard_mode)

    result = (h1, has_orient_rev, heegaard_genus_estimate)
    cache[iso_sig] = result
    return result


def get_n_2cycles(g1, g2):
    """Count the bicoloured cycles (connected components) of the residue
    spanned by the edge colours g1 and g2. This is the paper's g_ij."""
    vertices = set(g1.keys())
    checked_vertices = set()
    n_cycles = 0
    for v in vertices:
        if v in checked_vertices:
            continue
        n_cycles += 1
        w = v
        while True:
            w = g1[w]
            checked_vertices.add(w)
            w = g2[w]
            checked_vertices.add(w)
            if w == v:
                break
    return n_cycles


def _as_perms(lengths, g0, g1, g2, g3):
    """Convert a gem into the triple of permutations used by
    ``build_regina_manifold``.

    The gem is bipartite.  ``vertices1`` and ``vertices2`` are its two
    classes, ordered so that colour 0 pairs vertices1[k] with vertices2[k]
    (this is asserted).  Each colour-0 pair is one double pyramid.
    pi_c[k] is the index of the vertex that colour c joins to vertices1[k].
    """
    # Convert the bidirectional dicts into permutations of the vertices
    vertices1 = [(i, j) for i in range(3)
                 for j in range(0 if i == 0 else 1, lengths[i], 2)]
    vertices2 = [(i, j) for i in range(3)
                 for j in range(1 if i == 0 else 0, lengths[i], 2)]
    inds1 = {v: idx for idx, v in enumerate(vertices1)}
    inds2 = {v: idx for idx, v in enumerate(vertices2)}
    assert [inds2[g0[v]] for v in vertices1] == list(range(len(vertices1)))
    pi1 = [inds2[g1[v]] for v in vertices1]
    pi2 = [inds2[g2[v]] for v in vertices1]
    pi3 = [inds2[g3[v]] for v in vertices1]
    return pi1, pi2, pi3


def get_all_3cycle_graphs(n1, n2, n3):
    """Enumerate all 3-gems whose {0,1} residue has three cycles of lengths
    n1 <= n2 <= n3 and whose {2,3} residue also has three cycles.

    Raises AssertionError when (n1, n2, n3) is not admissible: the lengths
    must be even, their sum must be divisible by 4, and every m_ij > 0.
    """
    assert n1 % 2 == 0 and n2 % 2 == 0 and n3 % 2 == 0
    assert n3 >= n2 and n2 >= n1, "Require n3 > n2 > n1 for a 3-cycle graph"
    N = n1 + n2 + n3
    assert N % 4 == 0, "# edges must be even, and total must be divisible by 4 for a 3-cycle graph"
    m12 = (n1 + n2 - n3) // 2
    m13 = (n1 + n3 - n2) // 2
    m23 = (n2 + n3 - n1) // 2
    lengths = [n1, n2, n3]
    results = []

    assert m12 > 0 and m13 > 0 and m23 > 0

    # Colours 0 and 1 alternate around each of the three cycles.
    G0 = _bidirect(
        {(i, j): (i, (j+1) % lengths[i]) for i in range(3) for j in range(0, lengths[i], 2)})
    G1 = _bidirect(
        {(i, j): (i, (j+1) % lengths[i]) for i in range(3) for j in range(1, lengths[i], 2)})
    # Colour 2: the unique planar matching between the three cycles (Lemma 3.6),
    # with m12, m13, m23 edges between the corresponding pairs of cycles.
    G2 = _bidirect({(0, j): (1, j) for j in range(m12)} |
                   {(0, m12 + j): (2, m23 + j) for j in range(m13)} |
                   {(1, m12 + j): (2, m23 - j - 1) for j in range(m23)})

    # {0,1,2} is a gem of S^2:  g01 + g02 + g12 = 2 + N/2.
    assert get_n_2cycles(G0, G1) == 3
    assert get_n_2cycles(G0, G2) == N//4
    assert get_n_2cycles(G1, G2) == N//4-1

    # Now we run over all possible choices of G3: the colour-2 pattern with
    # the endpoints on cycle i shifted by 2*q_i + par.  Keep those with
    # g23 = 3.  The gem conditions g03 = g12 and g13 = g02 then hold too
    # (asserted below).
    for par in (0, 1):
        for q1 in range(1, n1//2+1):
            for q2 in range(1, n2//2+1):
                for q3 in range(1, n3//2+1):
                    G3 = _bidirect({(0, (q1*2+par+j) % n1): (1, (q2*2+par+j) % n2) for j in range(m12)} |
                                   {(0, (q1*2+par+j+m12) % n1): (2, (q3*2+par+j+m23) % n3) for j in range(m13)} |
                                   {(1, (q2*2+par+j+m12) % n2): (2, (q3*2+par-j-1+m23) % n3)
                                    for j in range(m23)})
                    if get_n_2cycles(G2, G3) == 3:
                        assert get_n_2cycles(G0, G3) == N//4-1
                        assert get_n_2cycles(G1, G3) == N//4
                        pi1, pi2, pi3 = _as_perms(lengths, G0, G1, G2, G3)
                        res = FourGraph(G0, G1, G2, G3, par,
                                        q1, q2, q3, pi1, pi2, pi3)
                        results.append(res)
    print(
        f"Found {len(results)}/{n1*n2*n3//2} 3-cycle graphs with n1={n1}, n2={n2}, n3={n3}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n", type=int, default=14)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument(
        "--heegaard-mode",
        choices=["fast", "deeper"],
        default="fast",
        help="Heegaard estimate depth: fast=vertex normal surfaces, deeper=vertex+fundamental on original+simplified triangulations.",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also print whether the triangulation has an orientation-reversing automorphism, and the Heegaard genus estimate.",
    )
    args = parser.parse_args()

    # Scan all n1 <= n2 <= n3 in [min_n, max_n] (even).  Manifolds are
    # deduplicated by isoSig, and printed results by volume (rounded to 4
    # decimals).
    min_n = args.min_n
    max_n = args.max_n
    heegaard_mode = args.heegaard_mode
    seen_isosigs = set()
    seen_vols = set()
    invariant_cache: dict[str, Tuple[str, Optional[bool], str]] = {}

    for n1 in range(min_n, max_n+1, 2):
        for n2 in range(n1, max_n+1, 2):
            for n3 in range(n2, max_n+1, 2):
                try:
                    results = get_all_3cycle_graphs(n1, n2, n3)
                except AssertionError:
                    # Inadmissible (n1, n2, n3), e.g. sum not divisible by 4.
                    results = []
                for res in results:
                    perms = [res.pi1, res.pi2, res.pi3]
                    manifold = build_regina_manifold(perms)
                    is_hyperbolic, status, iso, volume = classify_hyperbolic_candidate(
                        manifold, seen_isosigs)

                    if is_hyperbolic and round(volume, 4) not in seen_vols:
                        seen_vols.add(round(volume, 4))
                        h1, has_orient_rev, heegaard_genus_estimate = hyperbolic_invariants(
                            manifold, iso, invariant_cache, heegaard_mode)
                        print("n1, n2, n3 =", n1, n2, n3)
                        print("q1, q2, q3, par=", res.q1,
                              res.q2, res.q3, res.par)
                        print(perms)
                        print(iso)
                        print(f"{status} volume={volume:.12g}")
                        print(f"H1={h1}")
                        if args.extra:
                            print(
                                "has_orientation_reversing_automorphism="
                                f"{has_orient_rev}"
                            )
                            print(
                                "heegaard_genus_estimate="
                                f"{heegaard_genus_estimate}"
                            )
