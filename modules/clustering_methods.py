import numpy as np
from collections import Counter

from modules.dbscans_core import run_dbscan
from modules.cluster_utils import finalize_clusters


# =========================================================================
# METHOD 1 -- Hierarchical interval-merge (NOT DBSCAN). Boxes are grouped
# by sorting along an axis and walking through once: a new group starts
# only when the gap to the current group's reach exceeds eps. Unlike
# DBSCAN, there is no density-reachability, so a group can never grow by
# hopping through a chain of intermediate boxes -- it only ever extends
# by direct, local, consecutive-gap decisions. Applied hierarchically:
# first split by X into layers, then within each layer, split by Z into
# rows.
# =========================================================================
def interval_merge_groups(pos, dim, eps):
    """Groups n items into contiguous 1D clusters along one axis.

    Sorts by each item's min edge, then walks through in order, tracking
    the running max edge ("reach") of the current group. If the next
    item's min edge is within eps of the current reach, it joins the
    group and extends the reach (max(reach, this item's max edge) --
    this correctly handles a box that fully contains, or is contained by,
    the group so far). Otherwise a new group starts.

    Returns a list of group ids (0, 1, 2, ...), one per item, in the
    ORIGINAL (unsorted) order.
    """
    n = len(pos)
    if n == 0:
        return []

    half = dim / 2.0
    item_min = pos - half
    item_max = pos + half

    order = np.argsort(item_min)
    group_id = np.empty(n, dtype=int)

    # Groups start at 1, NOT 0 -- finalize_clusters() (shared with the
    # DBSCAN-based methods) treats cluster_id == 0 as its noise sentinel.
    # If groups were numbered starting at 0, the first real group found
    # by this sweep would collide with that sentinel and get incorrectly
    # shredded into singletons by finalize_clusters, regardless of its
    # actual size. Confirmed bug in the previous version -- this is the
    # fix.
    current_group = 1
    current_reach = item_max[order[0]]
    group_id[order[0]] = current_group

    for k in range(1, n):
        idx = order[k]
        if item_min[idx] <= current_reach + eps:
            # Joins the current group; extend the reach if this box's max
            # edge goes further (handles a small box fully inside a big
            # one, or a big box that extends past everything so far).
            current_reach = max(current_reach, item_max[idx])
        else:
            current_group += 1
            current_reach = item_max[idx]
        group_id[idx] = current_group

    return group_id.tolist()


def _apply_min_size(group_id, min_size):
    """Marks any group smaller than min_size as noise (id 0), leaving
    real groups unchanged. Mirrors DBSCAN's min_samples filtering so
    results stay comparable across methods."""
    counts = Counter(group_id)
    return [gid if counts[gid] >= min_size else 0 for gid in group_id]


def fixed_bin_groups(pos, bin_size, origin=0.0):
    """Quantizes positions into fixed-width bins along one axis, REGARDLESS
    of whether the underlying data actually has a gap there. Unlike
    interval_merge_groups (which only splits where a real gap exists),
    this forcibly imposes a split every `bin_size` units, starting from
    `origin`.

    This is the right tool when you know (or want to assume) a real,
    fixed row/shelf height, even if the raw positions are continuously
    distributed with no natural gap to detect. The trade-off: a box whose
    center sits right at a bin boundary can be assigned to either
    neighboring bin somewhat arbitrarily, since the cut is imposed rather
    than found.

    `origin` should normally be a GLOBAL reference (e.g. the container's
    floor, or the overall minimum Z across every layer) rather than each
    layer's own local minimum -- shelf heights are typically consistent
    across the whole container, not relative to wherever a given layer
    happens to start.

    Returns group ids starting at 1 (not 0, to avoid colliding with
    finalize_clusters' noise sentinel), one per item, in the ORIGINAL
    (unsorted) order.
    """
    bin_index = np.floor((pos - origin) / bin_size).astype(int)
    return (bin_index + 1 - bin_index.min()).tolist()  # shift so ids start at 1


def cluster_hierarchical_interval(x_pos, x_dim, z_pos, z_dim, n, eps_x, row_height,
                                   min_samples_x=1, min_samples_z=1, z_origin=None):
    """Hierarchical, non-DBSCAN replacement for the old AxisGap method.

    Stage 1: group all boxes by X alone (interval merge, gap-based) -> layers.
    Stage 2: within each layer, split by Z into FIXED-HEIGHT bins of size
    `row_height` -> rows. This is a forced quantization, not a gap search --
    it will split a layer into rows of that height even if the Z positions
    are perfectly continuous with no real gap anywhere, which gap-based
    interval-merge cannot do.

    z_origin anchors the bin boundaries; defaults to the global minimum Z
    across ALL boxes (not per-layer), so row boundaries line up
    consistently across every layer -- e.g. "row 1" means the same
    physical height band everywhere in the container, not a height
    relative to wherever each layer happens to start.

    min_samples_x / min_samples_z filter out groups smaller than that
    size, marking their members as noise.
    """
    if z_origin is None:
        z_origin = float(z_pos.min())

    # --- Stage 1: layers from X (gap-based, unchanged) ---
    layer_id = interval_merge_groups(x_pos, x_dim, eps_x)
    layer_id = _apply_min_size(layer_id, min_samples_x)
    layer_id, layers = finalize_clusters(layer_id, n)  # noise -> singleton, else grouped by layer

    # --- Stage 2: rows from Z, fixed-height bins, independently within each layer ---
    final_cluster_id = [None] * n
    next_final_id = 0   # counts UP for real clusters (always positive)
    next_noise_id = 0   # counts DOWN for noise singletons (always negative)
    n_raw_total = 0

    for layer_key, indices in layers.items():
        if layer_key < 0:
            # Stage-1 noise singleton: stays its own final cluster, not
            # split further by Z. Uses the SEPARATE noise counter --
            # sharing one counter between real-cluster increments and
            # noise decrements was a confirmed bug: decrementing an
            # already-large positive counter does not reliably produce a
            # negative id, it can land exactly on an id already assigned
            # to an unrelated real cluster from a previous layer, merging
            # two spatially distant groups purely by numeric coincidence.
            next_noise_id -= 1
            for idx in indices:
                final_cluster_id[idx] = next_noise_id
            continue

        idx_arr = np.array(indices)
        sub_z_pos = z_pos[idx_arr]

        sub_row_id = fixed_bin_groups(sub_z_pos, row_height, origin=z_origin)
        sub_row_id = _apply_min_size(sub_row_id, min_samples_z)
        sub_row_id, sub_rows = finalize_clusters(sub_row_id, len(idx_arr))
        n_raw_total += sum(1 for k in sub_rows if k > 0)

        local_to_final = {}
        for local_id in sub_rows:
            if local_id > 0:
                next_final_id += 1
                local_to_final[local_id] = next_final_id
            else:
                next_noise_id -= 1
                local_to_final[local_id] = next_noise_id

        for local_pos, global_idx in enumerate(idx_arr):
            final_cluster_id[global_idx] = local_to_final[sub_row_id[local_pos]]

    final_cluster_id, final_clusters = finalize_clusters(final_cluster_id, n, give_noise_own_cluster=False)
    return final_cluster_id, final_clusters, n_raw_total


# =========================================================================
# METHOD 2 -- Overlap fraction (per-axis overlap length normalized by the
# SMALLER box's extent on that axis; avoids the area-IoU size-mismatch
# distortion, still checks X and Z independently like AreNeighbors).
# =========================================================================
def overlap_fraction_matrix(pos, dim):
    half = dim / 2.0
    a_min = pos[:, None] - half[:, None]
    a_max = pos[:, None] + half[:, None]
    b_min = pos[None, :] - half[None, :]
    b_max = pos[None, :] + half[None, :]

    overlap_len = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    min_dim = np.minimum(dim[:, None], dim[None, :])
    min_dim = np.where(min_dim <= 0, np.inf, min_dim)

    frac = overlap_len / min_dim
    np.fill_diagonal(frac, 1.0)
    return frac


def cluster_overlap_fraction(x_pos, x_dim, z_pos, z_dim, n, frac_x_thresh, frac_z_thresh, min_samples):
    frac_x = overlap_fraction_matrix(x_pos, x_dim)
    frac_z = overlap_fraction_matrix(z_pos, z_dim)
    nm = (frac_x >= frac_x_thresh) & (frac_z >= frac_z_thresh)
    np.fill_diagonal(nm, False)
    neighbors = [np.nonzero(nm[i])[0].tolist() for i in range(n)]
    cluster_id, n_raw = run_dbscan(neighbors, min_samples, n)
    cluster_id, clusters = finalize_clusters(cluster_id, n)
    return cluster_id, clusters, n_raw


# =========================================================================
# METHOD 3 -- Auto-searched overlap fraction: grid search over frac_x/frac_z,
# keep only zero-noise combinations, pick the one with the MOST clusters
# (tightest threshold that still classifies every box). Heuristic, not a
# verified-correct answer -- there's no ground truth in random test data
# to check against.
# =========================================================================
def cluster_overlap_fraction_auto(x_pos, x_dim, z_pos, z_dim, n, min_samples, search_grid=None):
    if search_grid is None:
        search_grid = np.round(np.arange(0.0, 1.01, 0.05), 2)

    frac_x = overlap_fraction_matrix(x_pos, x_dim)
    frac_z = overlap_fraction_matrix(z_pos, z_dim)

    best = None  # (n_raw, fx, fz, cluster_id)
    for fx in search_grid:
        for fz in search_grid:
            nm = (frac_x >= fx) & (frac_z >= fz)
            np.fill_diagonal(nm, False)
            neighbors = [np.nonzero(nm[i])[0].tolist() for i in range(n)]
            cid, n_raw = run_dbscan(neighbors, min_samples, n)
            n_noise = cid.count(0)
            if n_noise == 0 and (best is None or n_raw > best[0]):
                best = (n_raw, fx, fz, cid)

    if best is None:
        raise RuntimeError("No zero-noise threshold found in search_grid; try lowering min_samples.")

    n_raw, best_fx, best_fz, cluster_id = best
    cluster_id, clusters = finalize_clusters(cluster_id, n)
    print(f"  [auto] chosen thresholds: frac_x={best_fx:.2f}, frac_z={best_fz:.2f}")
    return cluster_id, clusters, n_raw