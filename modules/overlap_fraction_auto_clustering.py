import numpy as np

from modules.dbscans_core import run_dbscan
from modules.cluster_utils import finalize_clusters
from modules.base_clustering import ClusteringMethod
from modules.clustering_helpers import overlap_fraction_matrix


# =========================================================================
# METHOD 3 -- Auto-searched overlap fraction: grid search over frac_x/frac_z,
# keep only zero-noise combinations, pick the one with the MOST clusters
# (tightest threshold that still classifies every box). Heuristic, not a
# verified-correct answer -- there's no ground truth in random test data
# to check against.
# =========================================================================
class OverlapFractionAutoClustering(ClusteringMethod):
    name = "overlap_fraction_auto"

    def __init__(self, min_samples, search_grid=None):
        self.min_samples = min_samples
        self.search_grid = search_grid if search_grid is not None else np.round(np.arange(0.0, 1.01, 0.05), 2)

    def _cluster(self, x_pos, x_dim, z_pos, z_dim, n):
        frac_x = overlap_fraction_matrix(x_pos, x_dim)
        frac_z = overlap_fraction_matrix(z_pos, z_dim)

        best = None  # (n_raw, fx, fz, cluster_id)
        for fx in self.search_grid:
            for fz in self.search_grid:
                nm = (frac_x >= fx) & (frac_z >= fz)
                np.fill_diagonal(nm, False)
                neighbors = [np.nonzero(nm[i])[0].tolist() for i in range(n)]
                cid, n_raw = run_dbscan(neighbors, self.min_samples, n)
                n_noise = cid.count(0)
                if n_noise == 0 and (best is None or n_raw > best[0]):
                    best = (n_raw, fx, fz, cid)

        if best is None:
            raise RuntimeError("No zero-noise threshold found in search_grid; try lowering min_samples.")

        n_raw, best_fx, best_fz, cluster_id = best
        cluster_id, clusters = finalize_clusters(cluster_id, n)
        print(f"  [auto] chosen thresholds: frac_x={best_fx:.2f}, frac_z={best_fz:.2f}")
        return cluster_id, clusters, n_raw
