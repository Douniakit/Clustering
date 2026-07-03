import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 7 alpha levels -- 7 is coprime with the 20-color tab20 palette, so the
# combined (color, alpha) pairing only repeats every LCM(20, 7) = 140
# clusters instead of every 20. Two clusters sharing a color due to the
# % 20 wraparound will very likely still render at different opacities.
ALPHA_LEVELS = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15]


def cuboid_faces(cx, cy, cz, dx, dy, dz):
    """Return the 6 faces of a box centered at (cx,cy,cz) with full
    extents (dx,dy,dz), as a list of 4-point polygons for Poly3DCollection."""
    x0, x1 = cx - dx / 2, cx + dx / 2
    y0, y1 = cy - dy / 2, cy + dy / 2
    z0, z1 = cz - dz / 2, cz + dz / 2
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),  # bottom
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),  # top
    ]
    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[0], v[3], v[7], v[4]],  # left
    ]


def build_3d_figure(boxes, cluster_id, clusters, title):
    """Builds a standalone 3D figure for a single method's clustering
    result but does NOT call plt.show() -- lets the caller create several
    figures first and show them all at once. Rotation (yaw) is ignored,
    consistent with the neighbor tests themselves (all axis-aligned,
    unrotated half-extent models)."""
    n = boxes["n"]
    x_pos, y_pos, z_pos = boxes["x_pos"], boxes["y_pos"], boxes["z_pos"]
    x_dim, y_dim, z_dim = boxes["x_dim"], boxes["y_dim"], boxes["z_dim"]

    cmap = plt.get_cmap("tab20")
    unique_ids = sorted(set(cluster_id))
    color_map = {c: cmap(k % 20) for k, c in enumerate(unique_ids)}
    alpha_map = {c: ALPHA_LEVELS[k % len(ALPHA_LEVELS)] for k, c in enumerate(unique_ids)}

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n):
        c = cluster_id[i]
        if c < 0:
            color = "lightgray"
        else:
            r, g, b, _ = color_map[c]
            color = (r, g, b, alpha_map[c])  # bake this cluster's alpha into the facecolor
        faces = cuboid_faces(x_pos[i], y_pos[i], z_pos[i], x_dim[i], y_dim[i], z_dim[i])
        box_poly = Poly3DCollection(faces, facecolor=color, edgecolor="black", linewidths=0.3)
        ax.add_collection3d(box_poly)

    ax.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
    ax.set_ylim(y_pos.min() - 0.5, y_pos.max() + 0.5)
    ax.set_zlim(0, z_pos.max() + z_dim.max())
    ax.set_xlabel("X (depth into container)")
    ax.set_ylabel("Y (left <-> right)")
    ax.set_zlabel("Z (height)")
    ax.set_title(f"{title}\n{len(clusters)} walls")

    ax.set_box_aspect((
        x_pos.max() - x_pos.min() + 1,
        y_pos.max() - y_pos.min() + 1,
        z_pos.max() + z_dim.max()
    ))
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    return fig


def plot_clusters_one_by_one(boxes, cluster_id, clusters, title_prefix="Cluster", output_dir="figs"):
    """Saves one 3D figure PER CLUSTER for a single method's result to
    disk, highlighting that cluster's boxes while dimming every other box
    as light gray context. No windows are opened -- each figure is
    rendered and saved directly, so this works the same whether you have
    5 clusters or 500.

    Files are written to `output_dir` (created if it doesn't exist) as
    `{title_prefix}_wall_{cluster_id:03d}.png`, one per real cluster.

    Iterates over real clusters (id > 0) in ascending id order. Noise
    singletons (id < 0) are skipped -- they're already visually distinct
    as gray in the other plot functions and there could be dozens of them.
    """
    import os

    n = boxes["n"]
    x_pos, y_pos, z_pos = boxes["x_pos"], boxes["y_pos"], boxes["z_pos"]
    x_dim, y_dim, z_dim = boxes["x_dim"], boxes["y_dim"], boxes["z_dim"]

    os.makedirs(output_dir, exist_ok=True)

    real_ids = sorted(cid for cid in clusters if cid > 0)
    print(f"Saving {len(real_ids)} cluster figures to '{output_dir}/'...")

    for cid in real_ids:
        member_idxs = set(clusters[cid])

        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(n):
            is_member = i in member_idxs
            faces = cuboid_faces(x_pos[i], y_pos[i], z_pos[i], x_dim[i], y_dim[i], z_dim[i])
            if is_member:
                facecolor = (1.0, 0.3, 0.1, 0.9)  # bright orange-red, mostly opaque
                edgecolor = "black"
                linewidth = 0.6
            else:
                facecolor = (0.7, 0.7, 0.7, 0.08)  # faint gray context
                edgecolor = (0.5, 0.5, 0.5, 0.3)
                linewidth = 0.15
            box_poly = Poly3DCollection(faces, facecolor=facecolor, edgecolor=edgecolor, linewidths=linewidth)
            ax.add_collection3d(box_poly)

        ax.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
        ax.set_ylim(y_pos.min() - 0.5, y_pos.max() + 0.5)
        ax.set_zlim(0, z_pos.max() + z_dim.max())
        ax.set_xlabel("X (depth into container)")
        ax.set_ylabel("Y (left <-> right)")
        ax.set_zlabel("Z (height)")
        ax.set_title(f"{title_prefix} -- wall {cid} ({len(member_idxs)} boxes)")

        ax.set_box_aspect((
            x_pos.max() - x_pos.min() + 1,
            y_pos.max() - y_pos.min() + 1,
            z_pos.max() + z_dim.max()
        ))
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()

        safe_prefix = title_prefix.replace(" ", "_")
        filepath = os.path.join(output_dir, f"{safe_prefix}_wall_{cid:03d}.png")
        plt.savefig(filepath, dpi=120)
        plt.close(fig)  # free memory immediately, don't accumulate open figures

    print(f"Done. {len(real_ids)} files written to '{output_dir}/'.")


def plot_3d_single(boxes, cluster_id, clusters, title):
    """Same as build_3d_figure but shows it immediately (blocks until
    closed). Kept for single-method use; for showing several methods at
    once, use build_3d_figure() per method then a single plt.show() call."""
    build_3d_figure(boxes, cluster_id, clusters, title)
    plt.show()


def plot_comparison(boxes, results, method_order, method_titles):
    """boxes: dict from box_loader.load_boxes
    results: dict of method_name -> (cluster_id, clusters, n_raw)
    method_order: list of method_name keys, left-to-right column order
    method_titles: dict of method_name -> display title

    Colors are arbitrary per column and NOT comparable across methods --
    only cluster membership/shape should be compared visually."""
    n = boxes["n"]
    x_pos, y_pos, z_pos = boxes["x_pos"], boxes["y_pos"], boxes["z_pos"]
    x_dim, y_dim, z_dim = boxes["x_dim"], boxes["y_dim"], boxes["z_dim"]

    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, len(method_order), figsize=(7 * len(method_order), 10))
    if len(method_order) == 1:
        axes = axes.reshape(2, 1)  # keep 2D indexing consistent for a single method

    for col, method in enumerate(method_order):
        cid, clusters, n_raw = results[method]
        unique_ids = sorted(set(cid))
        color_map = {c: cmap(k % 20) for k, c in enumerate(unique_ids)}
        alpha_map = {c: ALPHA_LEVELS[k % len(ALPHA_LEVELS)] for k, c in enumerate(unique_ids)}

        ax_top = axes[0, col]
        ax_side = axes[1, col]

        for i in range(n):
            c = cid[i]
            if c < 0:
                color = "lightgray"
            else:
                r, g, b, _ = color_map[c]
                color = (r, g, b, alpha_map[c])
            ax_top.add_patch(patches.Rectangle(
                (x_pos[i] - x_dim[i] / 2, y_pos[i] - y_dim[i] / 2),
                x_dim[i], y_dim[i], facecolor=color, edgecolor="black", linewidth=0.4
            ))
            ax_side.add_patch(patches.Rectangle(
                (x_pos[i] - x_dim[i] / 2, z_pos[i] - z_dim[i] / 2),
                x_dim[i], z_dim[i], facecolor=color, edgecolor="black", linewidth=0.4
            ))

        ax_top.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
        ax_top.set_ylim(y_pos.min() - 0.5, y_pos.max() + 0.5)
        ax_top.set_title(f"{method_titles[method]}\ntop view - {len(clusters)} walls")
        ax_top.set_aspect("equal")
        ax_top.grid(True, alpha=0.3)

        ax_side.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
        ax_side.set_ylim(0, z_pos.max() + z_dim.max())
        ax_side.set_title("side view")
        ax_side.set_aspect("equal")
        ax_side.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Y (left <-> right)")
    axes[1, 0].set_ylabel("Z (height)")
    for ax in axes[1, :]:
        ax.set_xlabel("X (depth)")

    plt.tight_layout()
    plt.show()