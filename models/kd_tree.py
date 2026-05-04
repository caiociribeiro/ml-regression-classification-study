import heapq
import numpy as np


class _KDNode:
    __slots__ = ("idx", "axis", "left", "right")

    def __init__(self, idx=None, axis=None, left=None, right=None):
        self.idx = idx
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim == 1:
            self.points = self.points.reshape(-1, 1)
        self.n_samples, self.dim = self.points.shape
        idxs = list(range(self.n_samples))
        self.root = self._build(idxs)

    def _build(self, idxs, depth=0):
        if not idxs:
            return None
        axis = depth % self.dim
        idxs.sort(key=lambda i: self.points[i, axis])
        mid = len(idxs) // 2
        node = _KDNode(
            idx=idxs[mid],
            axis=axis,
            left=self._build(idxs[:mid], depth + 1),
            right=self._build(idxs[mid + 1 :], depth + 1),
        )
        return node

    def _dist(self, x, idx, p):
        if p == 2:
            return np.sqrt(np.sum((x - self.points[idx]) ** 2))
        return np.sum(np.abs(x - self.points[idx]))

    def _query_single(self, node, x, k, heap, p):
        if node is None:
            return

        idx = node.idx
        axis = node.axis

        go_left = x[axis] <= self.points[idx, axis]
        first = node.left if go_left else node.right
        second = node.right if go_left else node.left

        if first is not None:
            self._query_single(first, x, k, heap, p)

        dist = self._dist(x, idx, p)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, idx))
        else:
            if dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, idx))

        if second is not None:
            diff = abs(x[axis] - self.points[idx, axis])
            if len(heap) < k or diff < -heap[0][0]:
                self._query_single(second, x, k, heap, p)

    def query(self, X, k=1, p=2):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        k_eff = min(k, max(1, self.n_samples))

        dists = np.zeros((n, k_eff), dtype=float)
        idxs = np.zeros((n, k_eff), dtype=int)

        for i, x in enumerate(X):
            heap = [] 
            self._query_single(self.root, x, k_eff, heap, p)

            heap_sorted = sorted([(-d, idx) for (d, idx) in heap])
            # pad if necessary
            for j in range(k_eff):
                if j < len(heap_sorted):
                    d, idx = heap_sorted[j]
                    dists[i, j] = d
                    idxs[i, j] = idx
                else:
                    dists[i, j] = np.inf
                    idxs[i, j] = -1

        return dists, idxs
