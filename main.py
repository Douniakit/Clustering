from modules.box_loader import load_boxes
from modules.cluster_utils import print_summary
from modules.clustering_methods import (
    cluster_hierarchical_interval,
    cluster_overlap_fraction,
    cluster_overlap_fraction_auto,
)
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
# Run all three methods
# =========================================================================
results = {}

cid, clusters, n_raw = cluster_hierarchical_interval(x_pos, x_dim, z_pos, z_dim, n, EPS_X, ROW_HEIGHT,
                                                      min_samples_x=MIN_SAMPLES, min_samples_z=MIN_SAMPLES)
results["hierarchical_interval"] = (cid, clusters, n_raw)
print_summary(f"Method 1: Hierarchical interval-merge (eps_x={EPS_X} for layers, row_height={ROW_HEIGHT} for rows)", cid, clusters, n_raw, n)

cid, clusters, n_raw = cluster_overlap_fraction(x_pos, x_dim, z_pos, z_dim, n, FRAC_X, FRAC_Z, MIN_SAMPLES)
results["overlap_fraction"] = (cid, clusters, n_raw)
print_summary(f"Method 2: Overlap fraction (frac_x={FRAC_X}, frac_z={FRAC_Z})", cid, clusters, n_raw, n)

cid, clusters, n_raw = cluster_overlap_fraction_auto(x_pos, x_dim, z_pos, z_dim, n, MIN_SAMPLES)
results["overlap_fraction_auto"] = (cid, clusters, n_raw)
print_summary("Method 3: Overlap fraction (auto-searched, max clusters at zero noise)", cid, clusters, n_raw, n)

# =========================================================================
# Side-by-side comparison plot
# =========================================================================
method_order = ["hierarchical_interval", "overlap_fraction", "overlap_fraction_auto"]
method_titles = {
    "hierarchical_interval": f"Hierarchical interval-merge (eps_x={EPS_X}, row_height={ROW_HEIGHT})",
    "overlap_fraction": f"Overlap frac ({FRAC_X}/{FRAC_Z})",
    "overlap_fraction_auto": "Overlap frac (auto)",
}

# =========================================================================
# One 3D figure per method, all shown simultaneously (built first, then
# a single blocking plt.show() displays every open figure at once)
# =========================================================================
for method in method_order:
    cid, clusters, n_raw = results[method]
    build_3d_figure(boxes, cid, clusters, method_titles[method])

plt.show()

# =========================================================================
# Optional: save one figure PER CLUSTER for a chosen method to figs/,
# instead of opening a window per cluster. Uncomment and pick a method:
# =========================================================================
cid, clusters, n_raw = results["hierarchical_interval"]
plot_clusters_one_by_one(boxes, cid, clusters, title_prefix="method1", output_dir="figs")