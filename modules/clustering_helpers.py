import numpy as np
from collections import Counter


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


def apply_min_size(group_id, min_size):
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
