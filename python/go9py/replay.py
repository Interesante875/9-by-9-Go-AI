"""Prioritized experience replay (Schaul et al. 2015) with a sum tree, stored as numpy arrays."""
from __future__ import annotations

import os

import numpy as np

from .board import HISTORY, NN, NUM_ACTIONS

FIELDS = {
    "boards": ((HISTORY, NN), np.int8),
    "legal": ((NUM_ACTIONS,), np.uint8),
    "to_move": ((), np.int8),
    "last_move": ((), np.int16),
    "komi": ((), np.float32),
    "policy": ((NUM_ACTIONS,), np.float32),
    "ownership": ((NN,), np.int8),
    "z": ((), np.float32),
    "score": ((), np.float32),
    "root_q": ((), np.float32),
    "old_logp": ((), np.float32),
    "old_value": ((), np.float32),
    "advantage": ((), np.float32),
    "policy_weight": ((), np.float32),
    "aux_weight": ((), np.float32),
    "action": ((), np.int16),
}


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.leaf0 = 1
        while self.leaf0 < capacity:
            self.leaf0 <<= 1
        self.tree = np.zeros(2 * self.leaf0, dtype=np.float64)

    def set(self, i: int, p: float) -> None:
        k = self.leaf0 + i
        delta = p - self.tree[k]
        while k >= 1:
            self.tree[k] += delta
            k >>= 1

    def get(self, i: int) -> float:
        return float(self.tree[self.leaf0 + i])

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def find(self, mass: float) -> int:
        k = 1
        tree = self.tree
        while k < self.leaf0:
            left = 2 * k
            if mass < tree[left] or tree[left + 1] <= 0:
                k = left
            else:
                mass -= tree[left]
                k = left + 1
        return min(k - self.leaf0, self.capacity - 1)


class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = {k: np.zeros((capacity,) + shape, dtype=dt) for k, (shape, dt) in FIELDS.items()}
        self.tree = SumTree(capacity)
        self.next = 0
        self.size = 0
        self.total_added = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, sample: dict) -> None:
        i = self.next
        for k, v in sample.items():
            self.data[k][i] = v
        self.tree.set(i, self.max_priority ** self.alpha)
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator):
        """Stratified proportional sampling. Returns (indices, normalised importance weights)."""
        total = self.tree.total
        seg = total / batch_size
        masses = np.minimum(total * (1 - 1e-12), seg * (np.arange(batch_size) + rng.random(batch_size)))
        idx = np.array([self.tree.find(m) for m in masses], dtype=np.int64)
        bad = idx >= self.size
        if bad.any():
            idx[bad] = rng.integers(0, self.size, bad.sum())
        p = np.array([self.tree.get(i) for i in idx]) / total
        w = (self.size * np.maximum(p, 1e-12)) ** (-beta)
        return idx, (w / w.max()).astype(np.float32)

    def update_priorities(self, idx, errors) -> None:
        for i, e in zip(idx, errors):
            p = abs(float(e)) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.set(int(i), p ** self.alpha)

    def priority(self, i: int) -> float:
        return self.tree.get(i)

    def batch(self, idx) -> dict:
        return {k: v[idx] for k, v in self.data.items()}

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        n = self.size
        # Chronological order (oldest first) so loading into a smaller buffer keeps the newest data.
        order = np.arange(n) if n < self.capacity else (np.arange(n) + self.next) % self.capacity
        arrays = {k: v[order] for k, v in self.data.items()}
        arrays["priorities"] = np.array([self.tree.get(int(i)) for i in order])
        arrays["meta"] = np.array([self.total_added, self.max_priority], dtype=np.float64)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            f = np.load(path)
            keys = set(f.files)
            if not set(FIELDS) <= keys:
                return False
            n = len(f["z"])
            start = max(0, n - self.capacity)
            self.__init__(self.capacity, self.alpha, self.eps)
            for k in FIELDS:
                arr = f[k][start:]
                self.data[k][: len(arr)] = arr
            pr = f["priorities"][start:]
            for i, p in enumerate(pr):
                self.tree.set(i, float(p))
            self.size = n - start
            self.next = self.size % self.capacity
            self.total_added = int(f["meta"][0])
            self.max_priority = float(f["meta"][1])
            return True
        except Exception:  # corrupt / incompatible file
            return False
