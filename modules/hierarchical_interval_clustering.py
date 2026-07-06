import numpy as np

from modules.cluster_utils import finalize_clusters
from modules.base_clustering import ClusteringMethod
from modules.clustering_helpers import interval_merge_groups, apply_min_size, fixed_bin_groups


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
class HierarchicalIntervalClustering(ClusteringMethod):
    """Stage 1: group all boxes by X alone (interval merge, gap-based) ->
    layers. Stage 2: within each layer, split by Z into FIXED-HEIGHT bins
    of size `row_height` -> rows. This is a forced quantization, not a
    gap search -- it will split a layer into rows of that height even if
    the Z positions are perfectly continuous with no real gap anywhere,
    which gap-based interval-merge cannot do.

    z_origin anchors the bin boundaries; defaults to the global minimum Z
    across ALL boxes (not per-layer), so row boundaries line up
    consistently across every layer -- e.g. "row 1" means the same
    physical height band everywhere in the container, not a height
    relative to wherever each layer happens to start.

    min_samples_x / min_samples_z filter out groups smaller than that
    size, marking their members as noise.
    """

    name = "hierarchical_interval"

    def __init__(self, eps_x, row_height, min_samples_x=1, min_samples_z=1, z_origin=None):
        self.eps_x = eps_x
        self.row_height = row_height
        self.min_samples_x = min_samples_x
        self.min_samples_z = min_samples_z
        self.z_origin = z_origin

    def _cluster(self, x_pos, x_dim, z_pos, z_dim, n):
        z_origin = self.z_origin
        if z_origin is None:
            z_origin = float(z_pos.min())

        # --- Stage 1: layers from X (gap-based, unchanged) ---
        layer_id = interval_merge_groups(x_pos, x_dim, self.eps_x)
        layer_id = apply_min_size(layer_id, self.min_samples_x)
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

            sub_row_id = fixed_bin_groups(sub_z_pos, self.row_height, origin=z_origin)
            sub_row_id = apply_min_size(sub_row_id, self.min_samples_z)
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
