# Locally-achiral 3-gems

Code for the computer search in **Sec. 3** of

> Y. Sheffer, *Universal entanglement probes of topological order and
> locally-achiral manifolds*, [arXiv:2606.18361](https://arxiv.org/abs/2606.18361).

Sec. 3.3 builds locally-achiral 3-gems. In these 4-coloured graphs the
residue of two colours has exactly three cycles. The code enumerates all of
them up to a given size, turns each one into a closed 3-manifold
triangulation, and identifies the hyperbolic manifolds it finds. These are
the manifolds that can be probed with multi-entropy measures.

## Contents

| File | What it is |
| --- | --- |
| [`cycle3_scan.py`](cycle3_scan.py) | **Main script.** Enumerates 3-gems with a 3-cycle {0,1} residue and a 3-cycle {2,3} residue (Lemma 3.6, Eq. (10)). It builds each manifold in Regina, filters for closed irreducible hyperbolic manifolds (Regina + SnapPy), and prints one representative per hyperbolic volume: its isoSig, volume and H₁. |
| [`regina_triangulations.py`](regina_triangulations.py) | Dependency. Builds a Regina `Triangulation3` from three permutations ("double pyramid" gluings). This is the same data as a gem with colour 0 fixed. |
| [`regina_orient_rev_scan.py`](regina_orient_rev_scan.py) | Dependency. Provides `has_orientation_reversing_automorphism`, used by `cycle3_scan.py --extra`. Run on its own, it is a census scan for orientation-reversing triangulation symmetries. That scan is not needed for Sec. 3. |
| [`link_pd`](link_pd) | PD codes of the Whitehead link and the figure-eight knot. Use them in SnapPy to build Dehn fillings (e.g. F8ₙ, Sec. 3.4) and compare them with the scan output. |
| [`magma_group_iso_test.txt`](magma_group_iso_test.txt) | Magma script. Tests whether π₁ of a manifold found by the search is isomorphic to π₁ of the Whitehead link filling W(5/1, 11/t), for t = 1…10. |

## Requirements

* Python ≥ 3.9
* [Regina](https://regina-normal.github.io/) ≥ 7.3 (`pip install regina`)
* [SnapPy](https://snappy.computop.org/) (`pip install snappy`)
* NumPy
* [Magma](http://magma.maths.usyd.edu.au/) (commercial) for `magma_group_iso_test.txt` only

```bash
pip install -r requirements.txt
```

Tested with Regina 7.4, SnapPy 3.3.2 and NumPy 2.5.5.

## Usage

```bash
python cycle3_scan.py --min-n 14 --max-n 20
```

The script scans all even cycle lengths `min_n ≤ n1 ≤ n2 ≤ n3 ≤ max_n` with
`n1 + n2 + n3 ≡ 0 (mod 4)`. The gem has `n1 + n2 + n3` vertices, i.e.
`(n1 + n2 + n3)/2` replicas. Options:

* `--extra` also prints whether the triangulation has an orientation-reversing
  automorphism, and a Heegaard-genus estimate.
* `--heegaard-mode {fast,deeper}` sets how much work the Heegaard-genus
  estimate does.

For each hyperbolic manifold with a new volume, the script prints:

```
n1, n2, n3 = 16 16 16
q1, q2, q3, par= 1 6 6 1
[[...], [...], [...]]            # the three permutations (pi1, pi2, pi3)
kLLLLPQkccefihhijjjlnxcsnwnqae   # isoSig
hyperbolic-numerical volume=1.58316666062
H1=Z/3 + Z/9
```

The isoSig labels one particular simplified triangulation. Regina's
simplification is not canonical, so the same manifold can print a different
isoSig in a different scan range. Volume and `M.identify()` /
`is_isometric_to` are the reliable identifiers.

This example is the only hyperbolic manifold found with
`--min-n 6 --max-n 16`. That run takes about 2 minutes on a laptop. The
default range (14–20) takes considerably longer.

To identify an output manifold:

```python
import snappy
M = snappy.Manifold("kLLLLPQkccefihhijjjlnxcsnwnqae")
M.identify()
```

## Conventions

* Colours are 0–3 in the code and 1–4 in the paper (code colour *c* = paper
  colour *c*+1). The paper's conditions g₁₂ = 3 and g₃₄ = 3 are
  `get_n_2cycles(G0, G1) == 3` and `get_n_2cycles(G2, G3) == 3` in the code.
* The colour-3 edges are the colour-2 matching with the endpoints on cycle *i*
  shifted by `2*q_i + par` steps.
