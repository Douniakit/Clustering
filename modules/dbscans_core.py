def run_dbscan(neighbors, min_samples, n):
    """Direct port of ClusterBoxesDBSCAN's main loop. Takes a precomputed
    neighbor list (neighbors[i] = list of indices considered neighbors of
    box i) so it works identically regardless of which distance/overlap
    definition produced that list.

    Used only by the DBSCAN-based methods (overlap-fraction and its
    auto-searched variant). Method 1 (hierarchical interval-merge) does
    NOT use this -- it doesn't use DBSCAN at all."""
    cluster_id = [-1] * n   # -1 = unvisited, 0 = noise (pre-relabel), >0 = cluster label
    visited = [False] * n
    next_cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        if len(neighbors[i]) < min_samples:
            cluster_id[i] = 0  # noise
            continue

        next_cluster_id += 1
        current_cluster_id = next_cluster_id
        cluster_id[i] = current_cluster_id

        seed_set = list(neighbors[i])
        s = 0
        while s < len(seed_set):
            q = seed_set[s]
            s += 1
            if not visited[q]:
                visited[q] = True
                if len(neighbors[q]) + 1 >= min_samples:
                    seed_set.extend(neighbors[q])
            if cluster_id[q] <= 0:
                cluster_id[q] = current_cluster_id

    return cluster_id, next_cluster_id