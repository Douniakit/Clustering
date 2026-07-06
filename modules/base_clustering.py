from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClusterResult:
    """Uniform output shape for every clustering method, regardless of
    the algorithm used internally (DBSCAN-based or not)."""
    method_name: str
    cluster_id: list
    clusters: dict
    n_raw: int


class ClusteringMethod(ABC):
    """Base class for all box-clustering methods. Subclasses implement
    `_cluster()` with whatever algorithm they like; `run()` is the single
    entry point everyone calls, and it always returns a ClusterResult
    with the same fields so callers (main.py, visualization, summaries)
    never need to know which subclass produced it."""

    name = "base"

    def run(self, x_pos, x_dim, z_pos, z_dim, n):
        cluster_id, clusters, n_raw = self._cluster(x_pos, x_dim, z_pos, z_dim, n)
        return ClusterResult(method_name=self.name, cluster_id=cluster_id, clusters=clusters, n_raw=n_raw)

    @abstractmethod
    def _cluster(self, x_pos, x_dim, z_pos, z_dim, n):
        """Returns (cluster_id, clusters, n_raw) -- same contract
        finalize_clusters() has always produced. run() wraps this into
        a ClusterResult."""
        raise NotImplementedError
