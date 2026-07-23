

# Box-to-Wall Clustering (branch `feat/wall_similarity_clustering_ff`)

This document describes how `Prioritizer::SortWalls()` groups detected boxes into `Wall` clusters
on this branch. Source: [prioritizer.cpp](../src/prioritizer.cpp) (`SortWalls`,
`FitClusterAndRecompute`), [prioritizer_clustering_utilities.cpp](../src/prioritizer_clustering_utilities.cpp),
and the `Wall` struct in [prioritizer_utilities.hpp](../include/async_planner/prioritizer_utilities.hpp).

A `Wall` is a rectangular (X/Z) group of boxes that share roughly the same tilting-conveyor angle —
i.e. boxes that can be picked from the same depth/height "slice" of the container.

## Pipeline overview

`Prioritizer::SortWalls()` runs these steps in order:

1. Sort all boxes.
2. Greedy initial assignment into walls.
3. Iterative refinement (K-means-like reassignment).
4. Cleanup: drop boxes with no valid pick segment, drop empty walls.
5. Merge away isolated, single-sided, small clusters into a neighboring wall.
6. Prune stale wall references from the wall order.

### 1. Sorting boxes (`SortWalls`, `prioritizer.cpp:830`)

Boxes are copied from `container_boxes_copy_` and sorted by:

- **Primary key**: X position of the box's **front face center** (`Box::GetBoxFrontFaceCenter()`),
  ascending. Front faces within 5 cm (`0.05`) of each other are treated as tied.
- **Secondary key** (tie-break): top Z (`position.z + dimensions.z/2`), ascending.

This is a recent change on this branch — an older commit sorted by the front face computed from
`position.x - dimensions.x/2` (axis-aligned bounding box), left commented out in the code, in favor of
the true (possibly rotated) front-face-center geometry. The intent is the same either way: process
boxes roughly front-to-back, then bottom-to-top, so the greedy pass below builds walls depth-first.

### 2 & 3. Clustering: `ClusterBoxesKMeans` (`prioritizer_clustering_utilities.cpp:281`)

Despite the name, this is not classic K-means (no fixed K, no centroid recomputation) — it's a
**greedy bin-packing pass followed by iterative local-search refinement**, both driven by the same
fit test.

#### Fit test — `FitClusterAndRecompute` (`prioritizer.cpp:764`)

For a candidate `(wall, box)` pair, computes what the wall's bounding box would become if `box` were
added:

```
new_front_x = min(wall.wall_front_x, box.front_x)
new_back_x  = max(wall.wall_back_x,  box.back_x)
new_bottom_z = min(wall.wall_bottom_z, box.bottom_z)
new_top_z    = max(wall.wall_top_z,    box.top_z)
```

The box fits iff the resulting span stays within fixed limits (`constants.hpp`):

- `new_back_x - new_front_x <= kWallMaxDepthDefinition` (**0.65 m**)
- `new_top_z - new_bottom_z <= kWallMaxHeightDefinition` (**0.55 m**)

If it fits, it also returns the resulting `area_increase = new_area - old_area` (area = span_x ×
span_z), used as the cost signal for placement/reassignment decisions.

There used to be an additional check tying a box's front face to the wall's front face within
`kWallFrontFaceTolerance` — it's commented out on this branch, so depth alignment is enforced only
indirectly via the max-depth span check above.

#### Blocking-safety test

Every candidate placement is also checked against `is_blocking_safe_fn`
(`Prioritizer::IsBoxPlacementBlockingSafe`): a box can only be assigned/moved to a wall if doing so
would not create a cycle in the wall-blocking DAG (walls that block each other via boxes stacked on
top of / in front of one another — see `prioritizer_dag_utilities.cpp`). This DAG safety check is
orthogonal to geometric fit and is applied everywhere a box changes wall membership during
clustering.

#### Step A — Greedy initial assignment

Boxes are walked in sorted order. For each box (skipping impossible/non-detection/perma-blocked/
ignored boxes, and boxes already fixed to a wall from a previous solution via `previous_sol_boxes`):

- Evaluate every existing wall; keep those where `FitClusterAndRecompute` succeeds **and**
  `is_blocking_safe_fn` allows it.
- Among those, pick the wall with the **smallest area increase**.
- If no existing wall qualifies, start a brand-new wall (UUID from a high-resolution clock tick)
  containing just this box.

This is a first-fit/best-fit bin-packing heuristic, not globally optimal, but cheap and gives a
reasonable starting partition since boxes are processed front-to-back.

#### Step B — Iterative refinement

Runs up to `kWallClusteringMaxIteration` (**10**) passes over all boxes. Each pass, for every
(non-impossible, non-fixed) box currently in some wall:

1. Compute `area_before` = current wall's area.
2. Compute `area_after_removal` = area of the current wall **without** this box (0 if it's the only
   box in the wall).
3. `area_freed = area_before - area_after_removal`.
4. For every *other* wall, if the box fits there (`FitClusterAndRecompute`) and it's blocking-safe,
   compute `net_gain = area_freed - area_inc_at_target`.
5. Move the box to whichever wall gives the largest strictly-positive `net_gain` (i.e. only move if
   it actually shrinks total wall area).

If a full pass produces no moves, refinement stops early. This is the "K-means-like" part: boxes
migrate toward whichever cluster they tighten the most, converging to a local optimum on total wall
area rather than reassigning around explicit centroids.

Boxes belonging to walls from the **previous** solution (`previous_sol_boxes`) are skipped in both
steps A and B — once a box has settled into a wall in an earlier prioritization cycle, clustering
leaves it there rather than reshuffling it, for solution stability across cycles.

### 4. Post-clustering cleanup (`SortWalls`, `prioritizer.cpp:912`)

For each resulting wall:

- Boxes are re-sorted (axis-aligned front-x, then top-z) purely for deterministic ordering.
- Each box is checked against `priority_segments_` for a valid (non-blocked, non-fully-pose-blocked,
  finite-cost) segment that can actually pick it. If none exists, the box is queued for removal from
  the wall (and its DAG supports removed).
- Walls left with no boxes or no segments after this are dropped entirely.

### 5. Merging isolated single-sided clusters — `RemoveIsolatedSingleSidedSmallClusters`

(`prioritizer_clustering_utilities.cpp:78`)

Small, lopsided walls are folded into a nearby wall rather than left as their own cluster:

- **`IsOneSided(wall)`**: true if ≥ `kClusterOneSidedRatio` (**90%**) of a wall's segments are graspable
  from the same arm side (LEFT or RIGHT).
- A wall is a merge candidate if it has `<= kMinBoxesCluster` (**3**) boxes **and** is one-sided.
- For each candidate wall `i`, every other wall `j` is considered a merge target if:
  - `ClustersFitMerged(i, j, is_merging_isolated_clusters=true)` — the merged X-span stays within
    `kWallMaxDepthDefinition`, and the merged Z-span stays within **1.5×** `kWallMaxHeightDefinition`
    (a relaxed height bound just for this isolated-cluster case; normal merges use the strict bound).
  - Every box in wall `i` is blocking-safe to move into wall `j`. This is checked as a set: the whole
    cluster moves as one unit, so a single unsafe box vetoes merging into that target (splitting the
    cluster across multiple targets is not attempted).
  - Among valid targets, the one with the smallest `VerticalGapBetweenWalls(i, j)` is chosen (0 if the
    two walls' Z-ranges already overlap).
- On merge: every box and segment moves from wall `i` into the target wall (with DAG supports
  updated per box), and wall `i` is emptied and excluded from the result.

Walls that don't get merged, and merge targets themselves, are all what get returned as the final
`sorted_walls`.

> Note: `Prioritizer::CheckMergeOnesided()` — an older, different one-sided-wall-merge step that ran
> later in `ComputePriorityChain` (after wall ordering, merging only the first two walls in
> `walls_order`) — is currently **commented out** at the call site (`prioritizer.cpp:3701`), left
> disabled so wall-order validation could be tested in isolation. It is not part of the active
> clustering path on this branch.

### 6. Pruning stale wall references

Any UUID left in `optimal_solution_.walls_order` that no longer corresponds to a surviving wall
(because it was cleaned up in step 4 or merged away in step 5) is removed, so later lookups by UUID
don't fail.

## Key constants (`include/async_planner/constants.hpp`)

| Constant | Value | Meaning |
|---|---|---|
| `kWallMaxDepthDefinition` | 0.65 m | Max X-span (front-to-back) of a wall |
| `kWallMaxHeightDefinition` | 0.55 m | Max Z-span (bottom-to-top) of a wall |
| `kWallClusteringMaxIteration` | 10 | Max refinement passes in `ClusterBoxesKMeans` |
| `kMinBoxesCluster` | 3 | Max box count for a wall to be considered "small" |
| `kClusterOneSidedRatio` | 0.9 | Fraction of same-arm-side segments to call a wall "one-sided" |

## Wall geometry bookkeeping (`Wall`, `prioritizer_utilities.hpp`)

Each `Wall` incrementally maintains `wall_front_x` / `wall_back_x` / `wall_bottom_z` / `wall_top_z`
(box-extent bounds, recomputed in `RecomputeBoundsAndCenter()` on every `AddBox`/`RemoveBox`), plus
`center_x`/`center_z` (box-position centroid) and `higher_boxes`/`closer_boxes` (used by downstream
wall-ordering heuristics, not by clustering itself). `Wall::Area()` — used throughout clustering as
the cost signal — is `SpanX() * SpanZ()` from these bounds, i.e. bounding-box area, not the sum of
individual box footprints.

## Not part of clustering itself

`ComputeWallIoUXZAligned`, `ComputeUnknownVolumeRatioAligned`, and `EvaluateWallScores`
(`prioritizer_clustering_utilities.cpp:169-279`) compute per-wall quality metrics (box-packing IoU,
unknown-volume ratio) but are not called from `SortWalls`/`ClusterBoxesKMeans` on this branch — they
exist as scoring helpers for other parts of prioritization (wall ordering/selection), not for forming
the clusters.
