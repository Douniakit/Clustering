from modules.box_loader import load_boxes
from modules.cluster_utils import print_summary
from modules.hierarchical_interval_clustering import HierarchicalIntervalClustering
from modules.overlap_fraction_clustering import OverlapFractionClustering
from modules.overlap_fraction_auto_clustering import OverlapFractionAutoClustering
from modules.visualize_compraison import build_3d_figure, plot_clusters_one_by_one
import matplotlib.pyplot as plt

# =========================================================================
# PARAMETERS
# =========================================================================
JSON_PATH = "containers/random_container_28042026.json"

EPS_X = 0.005                                   # gap tolerance for X layers (unchanged, gap-based)
ROW_HEIGHT = 0.50                              # fixed row height for Z splitting -- SET THIS to your real shelf spacing
FRAC_X, FRAC_Z = 0.70, 0.50                   # manually chosen overlap-fraction values
MIN_SAMPLES = 5                               # shared across all methods for a fair comparison

# =========================================================================
# Load data
# =========================================================================
boxes = load_boxes(JSON_PATH)
n = boxes["n"]
print(f"Loaded {n} boxes")

x_pos, y_pos, z_pos = boxes["x_pos"], boxes["y_pos"], boxes["z_pos"]
x_dim, y_dim, z_dim = boxes["x_dim"], boxes["y_dim"], boxes["z_dim"]

# =========================================================================
# Run all three methods -- every method inherits from ClusteringMethod and
# .run() always returns a ClusterResult with the same fields
# (method_name, cluster_id, clusters, n_raw), regardless of the algorithm.
# =========================================================================
methods = [
    HierarchicalIntervalClustering(EPS_X, ROW_HEIGHT, min_samples_x=MIN_SAMPLES, min_samples_z=MIN_SAMPLES),
    OverlapFractionClustering(FRAC_X, FRAC_Z, MIN_SAMPLES),
    OverlapFractionAutoClustering(MIN_SAMPLES),
]

method_titles = {
    "hierarchical_interval": f"Hierarchical interval-merge (eps_x={EPS_X}, row_height={ROW_HEIGHT})",
    "overlap_fraction": f"Overlap frac ({FRAC_X}/{FRAC_Z})",
    "overlap_fraction_auto": "Overlap frac (auto)",
}

results = {}
for method in methods:
    result = method.run(x_pos, x_dim, z_pos, z_dim, n)
    results[result.method_name] = result
    print_summary(result, n, title=method_titles[result.method_name])

# =========================================================================
# One 3D figure per method, all shown simultaneously (built first, then
# a single blocking plt.show() displays every open figure at once)
# =========================================================================
for method_name, result in results.items():
    build_3d_figure(boxes, result.cluster_id, result.clusters, method_titles[method_name])

plt.show()

# =========================================================================
# Optional: save one figure PER CLUSTER for a chosen method to figs/,
# instead of opening a window per cluster. Uncomment and pick a method:
# =========================================================================
#result = results["hierarchical_interval"]
#plot_clusters_one_by_one(boxes, result.cluster_id, result.clusters, title_prefix="method1", output_dir="figs")
