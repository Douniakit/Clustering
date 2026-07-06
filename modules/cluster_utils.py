def finalize_clusters(cluster_id, n, give_noise_own_cluster=True):
    """Relabels noise as singleton negative ids and groups indices by
    final cluster id. Independent of clustering algorithm -- used by all
    methods (DBSCAN-based and non-DBSCAN alike)."""
    cluster_id = list(cluster_id)
    if give_noise_own_cluster:
        next_noise_id = 0
        for i in range(n):
            if cluster_id[i] == 0:
                next_noise_id -= 1
                cluster_id[i] = next_noise_id

    clusters = {}
    for i in range(n):
        clusters.setdefault(cluster_id[i], []).append(i)
    return cluster_id, clusters


def print_summary(result, n, title=None):
    """Independent of clustering algorithm -- 'result' is a ClusterResult
    (same shape for every ClusteringMethod subclass), and 'n_raw' is just
    the count of clusters found before noise-singleton relabeling,
    regardless of whether the method used DBSCAN or not."""
    cluster_id, clusters, n_raw = result.cluster_id, result.clusters, result.n_raw
    n_noise = cluster_id.count(0) if 0 in cluster_id else sum(1 for c in cluster_id if c < 0)
    print(f"\n--- {title or result.method_name} ---")
    print(f"Produced {n_raw} raw clusters from {n} boxes ({n_noise} noise points)")
    print(f"{len(clusters)} final walls")
    sizes = sorted((len(idxs) for cid, idxs in clusters.items() if cid > 0), reverse=True)
    print(f"cluster sizes (top 10): {sizes[:10]}")