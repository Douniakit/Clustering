## 
# Box Clustering Methods 

## Problem

Boxes inside a container need to be grouped into "walls" (rows/layers) for
picking. Each box has a 3D position (`x`, `y`, `z`) and dimensions
(`x_dim`, `y_dim`, `z_dim`), where in this container's convention:

- **X** = depth into the container (increases with distance from the opening)
- **Y** = left/right position across the container width
- **Z** = height above the container floor

A "wall" is a group of boxes that sit at roughly the same depth (X) and
roughly the same height (Z), regardless of their spread in Y. Clustering is
done with DBSCAN, driven by a **neighbor test** that decides whether two
boxes are "close enough" on X and Z to belong to the same wall.

Three different neighbor tests were implemented and compared, all sharing
the same underlying DBSCAN traversal logic. Only the definition of
"neighbor" changes between them.

---

## Shared DBSCAN core (`modules/dbscans_core.py`)

`run_dbscan(neighbors, min_samples, n)` is a direct port of the production
C++ `ClusterBoxesDBSCAN` main loop:

- Each box starts unvisited.
- A box with fewer than `min_samples` neighbors is marked as noise.
- A box with enough neighbors starts a new cluster, which then expands by
  visiting each neighbor's own neighbors (density-reachability), exactly
  like standard DBSCAN.
- `finalize_clusters()` optionally gives every noise point its own
  singleton "cluster" (matching the C++ code's optional block), so every
  box ends up assigned to *some* wall.

This function is identical across all three methods below — it only
consumes a precomputed `neighbors[i] = [list of neighbor indices]`
structure, so swapping the neighbor definition doesn't touch the
clustering algorithm itself.

---

## Method 1 — AxisGap distance (`cluster_axisgap`)

**Definition:** two boxes are neighbors if the edge-to-edge gap between
their extents is within `eps_x` on the X axis **and** within `eps_z` on
the Z axis. This is a direct Python port of the production C++
`AxisGap` / `AreNeighbors` functions. A gap of `0` means the boxes'
extents already overlap or touch.

**Parameters:** `eps_x`, `eps_z`, `min_samples` (matches the production
values: `eps_x = eps_z = 0.05`, `min_samples = 5`).

**Known issue — chaining:** DBSCAN's density-reachability is transitive.
If box A is within `eps` of box B, and B is within `eps` of C, then A and
C end up in the same cluster even if A and C are far apart. On real
container data, this caused **all boxes to collapse into a single
cluster** spanning the container's full height and depth, because the
gap between consecutive rows/layers is small and roughly uniform — there
is no clean density "valley" for a fixed `eps` to exploit. This is a
structural limitation of distance-based DBSCAN on this kind of
data, not a parameter-tuning mistake — no `eps_x`/`eps_z` value fixes it
without also risking under-clustering or excessive noise elsewhere.

---

## Method 2 — Overlap fraction (`cluster_overlap_fraction`)

**Definition:** instead of gap distance, this checks how much two boxes'
footprints actually **overlap**, normalized by the **smaller** box's
extent on that axis:

```
overlap_length = max(0, min(a_max, b_max) - max(a_min, b_min))
fraction = overlap_length / min(dim_a, dim_b)
```

A fraction of `1.0` means the smaller box's entire extent on that axis is
covered by the other box. Checked independently for X and Z, matching the
original two-axis-AND structure of `AreNeighbors`.

**Why not plain area IoU:** an earlier version used intersection-over-union
of the two boxes' footprint areas. This under-counts overlap when box
sizes differ a lot — e.g. a 10cm box fully contained inside a 40cm box
scores IoU ≈ 0.06 (dominated by the union, which is mostly the big box's
area) even though the small box is 100% covered. The per-axis overlap
fraction fixes this by normalizing against the *smaller* box instead of
the union, so full containment always scores 1.0 regardless of the size
ratio.

**Parameters:** `frac_x_thresh`, `frac_z_thresh`, `min_samples`.
Manually chosen starting values: `frac_x = 0.70`, `frac_z = 0.50`.

**Known trade-off:** normalizing by the smaller box makes it *easier* for
a single oversized or misdetected box (unusually tall or unusually deep)
to "bridge" two otherwise-separate rows or layers, because the small
row-boxes on either side of it are evaluated against their own (small)
extent, not the bridging box's extent. This method does not eliminate
chaining — it only changes which boxes are prone to causing it.

---

## Method 3 — Overlap fraction, auto-searched (`cluster_overlap_fraction_auto`)

**Definition:** same neighbor test as Method 2, but `frac_x`/`frac_z`
are chosen automatically via grid search instead of set by hand.

**Selection rule:** among all `(frac_x, frac_z)` combinations tested,
keep only those that leave **zero boxes unclassified as noise**, then
pick the one that produces the **most clusters**. This is the tightest
(least merge-prone) threshold that still assigns every box to a wall —
the loosest threshold trivially achieves zero noise by collapsing
everything into one giant cluster, so cluster count is used as a
tie-breaker to reject that degenerate case.

**Important caveat:** this is a heuristic proxy, not a verified-correct
answer. There's no ground-truth row/layer labeling in the test data to
check the result against (the test file is randomly generated
specifically to *not* have artificial structure), so "most clusters at
zero noise" is a reasonable default, not a guarantee that the resulting
walls match the physically correct grouping.

**Cost:** runs a full DBSCAN pass for every grid point (e.g. 21×21 = 441
combinations at a 0.05 step), so it is significantly slower than Methods
1 or 2, which run DBSCAN exactly once.

---

## Corner cases identified during testing

1. **Chaining collapse:** confirmed on both random test data and real
   production logs — a set of boxes with small, roughly uniform gaps
   between adjacent rows/layers can merge into one enormous cluster
   regardless of how tight `eps`/`frac` thresholds are set, because
   DBSCAN's reachability is transitive. Worth checking whether downstream
   consumers of `Wall` objects assume a bounded wall size — the
   production code explicitly logs "no size cap enforced."
2. **Size-mismatch bridging:** a single very tall or very deep box can
   satisfy the neighbor test against two otherwise-unrelated rows/layers,
   merging them. Present in all three methods to varying degrees; not
   fully solved by any of them alone.
3. **Floating-point near-zero gaps:** boxes that are meant to exactly
   touch sometimes produce gaps like `2.22e-16` instead of exact `0.0`
   due to floating-point rounding. Harmless for `<=` comparisons against
   any reasonable `eps`, but would break a strict `== 0` check if one
   were ever added.
4. **Random test data has no true row/layer structure:** confirmed via
   histogram — Z positions are smoothly, continuously distributed with
   no empty bins, meaning there is no genuine density gap for
   distance-based clustering to find. This is expected for adversarial/
   corner-case testing but means clustering results on this file cannot
   be checked against a "correct" answer, only compared to each other.

---

## Code structure

```
Clustering/
├── main.py                        # entry point; loads data, runs all
│                                   # three methods, prints summaries,
│                                   # shows one 3D plot per method
├── containers/
│   └── random_container_28042026.json
└── modules/
    ├── box_loader.py               # load_boxes(): JSON -> numpy arrays
    ├── dbscans_core.py             # run_dbscan(), finalize_clusters(),
    │                               # print_summary() -- shared by all
    │                               # three methods
    ├── clustering_methods.py       # cluster_axisgap(),
    │                               # cluster_overlap_fraction(),
    │                               # cluster_overlap_fraction_auto()
    └── visualize_compraison.py     # build_3d_figure(), plot_3d_single(),
                                     # plot_comparison() -- 3D cuboid and
                                     # 2D top/side rendering, colored by
                                     # final cluster assignment
```

### Function reference

| Function | File | Purpose |
|---|---|---|
| `load_boxes(path)` | `box_loader.py` | Loads a container JSON file into numpy arrays (`x_pos`, `x_dim`, etc.) |
| `run_dbscan(neighbors, min_samples, n)` | `dbscans_core.py` | Core DBSCAN traversal, independent of neighbor definition |
| `finalize_clusters(cluster_id, n)` | `dbscans_core.py` | Relabels noise as singleton clusters, groups indices by final cluster id |
| `print_summary(...)` | `dbscans_core.py` | Prints cluster counts/sizes for a given result |
| `cluster_axisgap(...)` | `clustering_methods.py` | Method 1 — AxisGap distance neighbor test |
| `cluster_overlap_fraction(...)` | `clustering_methods.py` | Method 2 — per-axis overlap fraction neighbor test |
| `cluster_overlap_fraction_auto(...)` | `clustering_methods.py` | Method 3 — auto-searched overlap fraction thresholds |
| `build_3d_figure(...)` | `visualize_compraison.py` | Builds (without showing) a 3D cuboid plot for one method's result |
| `plot_3d_single(...)` | `visualize_compraison.py` | Builds and immediately shows one 3D plot |
| `plot_comparison(...)` | `visualize_compraison.py` | 2D top/side view grid, one column per method, for side-by-side comparison |

### Running

```bash
cd Clustering
python main.py
```

Update `JSON_PATH`, `EPS_X`/`EPS_Z`, `FRAC_X`/`FRAC_Z`, and `MIN_SAMPLES`
at the top of `main.py` to test different containers or parameters.
