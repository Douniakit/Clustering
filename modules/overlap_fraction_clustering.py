import numpy as np

from modules.dbscans_core import run_dbscan
from modules.cluster_utils import finalize_clusters
from modules.base_clustering import ClusteringMethod
from modules.clustering_helpers import overlap_fraction_matrix


# =========================================================================
# METHOD 2 -- Overlap fraction (per-axis overlap length normalized by the
# SMALLER box's extent on that axis; avoids the area-IoU size-mismatch
# distortion, still checks X and Z independently like AreNeighbors).
# =========================================================================
class OverlapFractionClustering(ClusteringMethod):
    name = "overlap_fraction"

    def __init__(self, frac_x_thresh, frac_z_thresh, min_samples):
        self.frac_x_thresh = frac_x_thresh
        self.frac_z_thresh = frac_z_thresh
        self.min_samples = min_samples

    def _cluster(self, x_pos, x_dim, z_pos, z_dim, n):
        frac_x = overlap_fraction_matrix(x_pos, x_dim)
        frac_z = overlap_fraction_matrix(z_pos, z_dim)
        nm = (frac_x >= self.frac_x_thresh) & (frac_z >= self.frac_z_thresh)
        np.fill_diagonal(nm, False)
        neighbors = [np.nonzero(nm[i])[0].tolist() for i in range(n)]
        cluster_id, n_raw = run_dbscan(neighbors, self.min_samples, n)
        cluster_id, clusters = finalize_clusters(cluster_id, n)
        return cluster_id, clusters, n_raw
