"""Scan closed orientable 3-manifold triangulations in Regina.

The script enumerates closed orientable triangulations up to a tetrahedron
bound and reports those whose triangulation has no orientation-reversing
automorphism while each vertex link does.

Enumeration is done through Regina's pairing/permutation search API (which
is how Regina census data is generated), deduplicated by isomorphism
signature.

Role in this repository: ``cycle3_scan.py`` imports only
:func:`has_orientation_reversing_automorphism`, and prints its result
when run with ``--extra``.  The census scan below (``python
regina_orient_rev_scan.py --max-tetrahedra N``) is a standalone tool.
It is not needed to reproduce Sec. 3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List


def _require_regina() -> Any:
    try:
        import regina  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Regina Python module is required. Install with: pip install regina"
        ) from exc
    return regina


def has_orientation_reversing_automorphism(triangulation: Any) -> bool:
    """Return True iff triangulation admits an orientation-reversing automorphism.

    For orientable triangulations, we orient first so parity is measured
    against a consistent global orientation. For each self-isomorphism, the
    map is orientation-reversing iff the induced permutation on every simplex
    has sign -1.
    """
    work = triangulation
    if work.isOrientable() and not work.isOriented():
        work = type(triangulation)(triangulation)
        work.orient()

    for iso in work.findAllIsomorphisms(work):
        signs = [iso.facetPerm(i).sign() for i in range(work.size())]
        if signs and all(s == -1 for s in signs):
            return True
    return False


def vertex_link_orientation_reversal_flags(triangulation3: Any) -> List[bool]:
    """Return one bool per vertex indicating link orientation-reversal symmetry."""
    flags: List[bool] = []
    for i in range(triangulation3.countVertices()):
        link = triangulation3.vertex(i).buildLink()
        flags.append(has_orientation_reversing_automorphism(link))
    return flags


def manifold_names_for_isosig(regina_module: Any, iso_sig: str) -> List[str]:
    """Return closed-manifold names from Regina census lookup for an isoSig."""
    names: List[str] = []
    for hit in regina_module.Census.lookup(iso_sig):
        db_desc = hit.db().desc().lower()
        # Restrict to closed census sources to avoid boundary/cusped aliases.
        if "closed" not in db_desc:
            continue
        full = hit.name()
        # Regina census names typically end in " : #k"; split from the right
        # so manifold descriptors containing colons stay intact.
        base = full.rsplit(" : #", 1)[0].strip(
        ) if " : #" in full else full.strip()
        names.append(base)
    # Stable de-duplication while preserving first appearance.
    seen = set()
    uniq: List[str] = []
    for nm in names:
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq


def _purge_flag(regina_module: Any) -> Any:
    # Prime+minimal filter keeps output manageable and closer to census usage.
    return regina_module.PURGE_NON_MINIMAL_PRIME


def iter_closed_orientable_triangulations(max_tetrahedra: int) -> Iterable[Any]:
    """Yield unique closed orientable triangulations up to max_tetrahedra."""
    regina = _require_regina()
    seen_iso: set[str] = set()

    def _on_gluing(gluing_perms: Any) -> None:
        tri = gluing_perms.triangulate()
        iso = tri.isoSig()
        if iso in seen_iso:
            return
        seen_iso.add(iso)
        triangulations.append(tri)

    def _on_pairing(pairing: Any, automorphisms: Any) -> None:
        regina.GluingPermSearcher3.findAllPerms(
            pairing,
            automorphisms,
            True,   # orientableOnly
            False,  # finiteOnly
            _purge_flag(regina),
            _on_gluing,
        )

    for n_tets in range(1, max_tetrahedra + 1):
        triangulations: List[Any] = []
        regina.FacetPairing3.findAllPairings(
            n_tets,
            regina.BoolSet(False),  # closed only
            -1,
            _on_pairing,
        )
        for tri in triangulations:
            yield tri


@dataclass
class TriangulationResult:
    manifold_names: List[str]
    iso_sig: str
    tetrahedra: int
    vertices: int
    tri_has_orientation_reversal: bool
    link_has_orientation_reversal: List[bool]


def scan(max_tetrahedra: int) -> Dict[str, Any]:
    """Run the scan and return raw + grouped results."""
    regina = _require_regina()
    passing: List[TriangulationResult] = []
    amphichiral_names: set[str] = set()

    for tri in iter_closed_orientable_triangulations(max_tetrahedra):
        iso = tri.isoSig()
        names = manifold_names_for_isosig(regina, iso)
        tri_has_rev = has_orientation_reversing_automorphism(tri)

        if tri_has_rev:
            # Practical manifold-level amphichirality filter: if a manifold
            # has an orientation-reversing triangulation in the scanned range,
            # exclude it from output.
            amphichiral_names.update(names)

        if tri_has_rev:
            continue

        link_flags = vertex_link_orientation_reversal_flags(tri)
        if not link_flags or not all(link_flags):
            continue

        passing.append(
            TriangulationResult(
                manifold_names=names,
                iso_sig=iso,
                tetrahedra=tri.size(),
                vertices=tri.countVertices(),
                tri_has_orientation_reversal=tri_has_rev,
                link_has_orientation_reversal=link_flags,
            )
        )

    filtered: List[TriangulationResult] = []
    for item in passing:
        names = [nm for nm in item.manifold_names if nm not in amphichiral_names]
        if not names:
            continue
        filtered.append(
            TriangulationResult(
                manifold_names=names,
                iso_sig=item.iso_sig,
                tetrahedra=item.tetrahedra,
                vertices=item.vertices,
                tri_has_orientation_reversal=item.tri_has_orientation_reversal,
                link_has_orientation_reversal=item.link_has_orientation_reversal,
            )
        )

    by_manifold: Dict[str, List[str]] = {}
    for item in filtered:
        for nm in item.manifold_names:
            by_manifold.setdefault(nm, []).append(item.iso_sig)

    # De-duplicate isoSigs per manifold.
    for nm, sigs in list(by_manifold.items()):
        uniq = []
        seen = set()
        for sig in sigs:
            if sig not in seen:
                seen.add(sig)
                uniq.append(sig)
        by_manifold[nm] = uniq

    return {
        "max_tetrahedra": max_tetrahedra,
        "matches": [asdict(r) for r in filtered],
        "manifolds": by_manifold,
        "excluded_amphichiral_manifolds": sorted(amphichiral_names),
    }


def _example_report() -> Dict[str, Dict[str, Any]]:
    regina = _require_regina()

    def _line(tri: Any) -> Dict[str, Any]:
        return {
            "iso_sig": tri.isoSig(),
            "tetrahedra": tri.size(),
            "vertices": tri.countVertices(),
            "tri_has_orientation_reversal": has_orientation_reversing_automorphism(tri),
            "link_has_orientation_reversal": vertex_link_orientation_reversal_flags(tri),
        }

    return {
        "S3": _line(regina.Example3.sphere()),
        "L(3,1)": _line(regina.Example3.lens(3, 1)),
        "Poincare(minimal)": _line(regina.Example3.poincare()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan closed orientable 3-manifold triangulations up to n tetrahedra "
            "for orientation-reversal behavior."
        )
    )
    parser.add_argument(
        "--max-tetrahedra",
        type=int,
        default=4,
        help="Upper tetrahedron bound for census scan (default: 4).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output (includes grouped manifolds and per-triangulation data).",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Print only S^3 / L(3,1) / minimal Poincare example diagnostics.",
    )
    args = parser.parse_args()

    if args.examples_only:
        payload = _example_report()
    else:
        payload = {
            "scan": scan(args.max_tetrahedra),
            "examples": _example_report(),
        }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.examples_only:
        print("Example checks:")
        for name, details in payload.items():
            print(f"- {name}: {details}")
        return

    scan_data = payload["scan"]
    print(
        "Scan complete for closed orientable triangulations up to"
        f" {scan_data['max_tetrahedra']} tetrahedra."
    )
    print(f"Passing triangulations: {len(scan_data['matches'])}")
    print("Manifolds with at least one passing triangulation:")
    for manifold in sorted(scan_data["manifolds"]):
        print(f"- {manifold}: {scan_data['manifolds'][manifold]}")


if __name__ == "__main__":
    main()
