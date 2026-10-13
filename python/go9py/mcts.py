"""PUCT Monte-Carlo Tree Search: batched leaf evaluation (virtual loss), FPU, root Dirichlet noise,
subtree reuse, exact Tromp-Taylor evaluation of finished games, and the "safe pass" rule."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .board import NN, PASS, WHITE, Board
from .features import snapshot


@dataclass
class MCTSConfig:
    c_puct_init: float = 1.1
    c_puct_base: float = 19652.0
    fpu_reduction: float = 0.2
    root_fpu_reduction: float = 0.0
    root_noise: bool = False
    noise_eps: float = 0.25
    noise_alpha_total: float = 10.83
    keep_ownership: bool = False


def terminal_eval(board: Board):
    """(value, score) of a finished game for the player to move."""
    s = board.score()
    if board.to_move == WHITE:
        s = -s
    return (1.0 if s > 0 else -1.0 if s < 0 else 0.0), s


class Node:
    __slots__ = ("board", "moves", "prior", "prior_raw", "children", "cn", "cw", "cs", "cvl",
                 "n", "w", "score_sum", "vl", "nn_value", "nn_score", "ownership", "expanded", "pending", "terminal")

    def __init__(self, board: Board):
        self.board = board
        self.moves = None
        self.prior = self.prior_raw = None
        self.children = None
        self.cn = self.cw = self.cs = self.cvl = None  # per-child visits / value sum / score sum / virtual loss
        self.n = 0
        self.w = 0.0          # value sum, perspective of the player to move here
        self.score_sum = 0.0
        self.vl = 0
        self.nn_value = 0.0
        self.nn_score = 0.0
        self.ownership = None
        self.expanded = self.pending = self.terminal = False

    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


@dataclass
class MoveStat:
    move: int
    visits: int
    winrate: float      # for the player to move at the root, 0..1
    score_lead: float
    prior: float
    lcb: float
    pv: list = field(default_factory=list)


class Search:
    def __init__(self, board: Board, cfg: MCTSConfig | None = None, seed: int = 1):
        self.cfg = cfg or MCTSConfig()
        self.rng = np.random.default_rng(seed)
        self.root = Node(board.copy())

    @property
    def root_board(self) -> Board:
        return self.root.board

    def reset(self, board: Board) -> None:
        self.root = Node(board.copy())

    def advance(self, action: int) -> None:
        """Moves the root to the child reached by ``action`` (reusing its subtree)."""
        r = self.root
        child = None
        if r.expanded:
            idx = np.nonzero(r.moves == action)[0]
            if len(idx):
                child = r.children[idx[0]]
        if child is None:
            b = r.board.copy()
            if not b.play(action):
                return
            child = Node(b)
        child.pending = False
        self.root = child
        if child.expanded:
            if self.cfg.root_noise:
                self._apply_noise()
            else:
                child.prior = child.prior_raw.copy()

    def set_root_noise(self, on: bool) -> None:
        self.cfg.root_noise = on
        if self.root.expanded:
            if on:
                self._apply_noise()
            else:
                self.root.prior = self.root.prior_raw.copy()

    def _apply_noise(self) -> None:
        r = self.root
        k = len(r.moves)
        if k == 0:
            return
        alpha = max(0.03, self.cfg.noise_alpha_total / k)
        noise = self.rng.dirichlet([alpha] * k)
        r.prior = (1 - self.cfg.noise_eps) * r.prior_raw + self.cfg.noise_eps * noise

    # ------------------------------------------------------------------ search
    def _select(self, node: Node, is_root: bool) -> int:
        n_eff = node.cn + node.cvl
        parent_n = node.n + node.vl
        c = self.cfg.c_puct_init + math.log((1 + parent_n + self.cfg.c_puct_base) / self.cfg.c_puct_base)
        visited = n_eff > 0
        red = self.cfg.root_fpu_reduction if is_root else self.cfg.fpu_reduction
        fpu = node.q() - red * math.sqrt(float(node.prior[visited].sum()))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.where(visited, (node.cw - node.cvl) / np.maximum(n_eff, 1), fpu)
        u = c * node.prior * math.sqrt(max(1.0, parent_n)) / (1.0 + n_eff)
        return int(np.argmax(q + u))

    def gather(self, k: int):
        """Selects up to k leaves needing a NN evaluation. Terminal leaves are backed up immediately."""
        leaves = []
        for _ in range(k):
            node, path, is_root = self.root, [], True
            while node.expanded and not node.terminal:
                i = self._select(node, is_root)
                child = node.children[i]
                if child is None:
                    b = node.board.copy()
                    b.play(int(node.moves[i]))
                    child = node.children[i] = Node(b)
                path.append((node, i))
                node, is_root = child, False
            if not node.expanded and not node.terminal and node.board.is_over():
                node.terminal = True
            if node.terminal:
                v, s = terminal_eval(node.board)
                node.nn_value, node.nn_score = v, s
                self._backup(path, node, v, s)
                continue
            if node.pending:
                break
            node.pending = True
            for parent, i in path:
                parent.cvl[i] += 1
                parent.vl += 1
            node.vl += 1
            leaves.append((path, node))
        return leaves

    def _expand(self, node: Node, e) -> None:
        legal = np.array(node.board.legal_mask(), dtype=bool)
        moves = np.nonzero(legal)[0]
        p = e.policy[moves].astype(np.float64)
        total = p.sum()
        p = p / total if total > 0 else np.full(len(moves), 1.0 / len(moves))
        order = np.argsort(-p, kind="stable")
        node.moves = moves[order]
        node.prior_raw = p[order]
        node.prior = node.prior_raw.copy()
        k = len(moves)
        node.children = [None] * k
        node.cn = np.zeros(k)
        node.cw = np.zeros(k)
        node.cs = np.zeros(k)
        node.cvl = np.zeros(k)
        node.nn_value, node.nn_score = float(e.value), float(e.score)
        if self.cfg.keep_ownership:
            node.ownership = e.ownership
        node.expanded = True
        if node is self.root and self.cfg.root_noise:
            self._apply_noise()

    def _backup(self, path, leaf: Node, value: float, score: float) -> None:
        leaf.n += 1
        leaf.w += value
        leaf.score_sum += score
        v, s = value, score
        for parent, i in reversed(path):
            v, s = -v, -s  # parent's perspective
            parent.cn[i] += 1
            parent.cw[i] += v
            parent.cs[i] += s
            parent.n += 1
            parent.w += v
            parent.score_sum += s

    def apply(self, leaves, evals) -> None:
        for (path, node), e in zip(leaves, evals):
            for parent, i in path:
                parent.cvl[i] -= 1
                parent.vl -= 1
            node.vl -= 1
            node.pending = False
            if not node.expanded:
                self._expand(node, e)
            self._backup(path, node, e.value, e.score)

    def run(self, evaluator, visits: int, batch: int = 8) -> None:
        guard = 0
        while self.root.n < visits:
            leaves = self.gather(1 if self.root.n == 0 else batch)
            if not leaves:
                guard += 1
                if guard > 100000:
                    break
                continue
            self.apply(leaves, evaluator.evaluate([snapshot(n.board) for _, n in leaves]))

    # ------------------------------------------------------------------ results
    @property
    def root_visits(self) -> int:
        return self.root.n

    def root_value(self) -> float:
        return self.root.q()

    def root_score(self) -> float:
        return self.root.score_sum / self.root.n if self.root.n else 0.0

    def visit_policy(self) -> np.ndarray:
        pi = np.zeros(NN + 1, dtype=np.float32)
        r = self.root
        if not r.expanded:
            return pi
        total = r.cn.sum()
        if total > 0:
            pi[r.moves] = r.cn / total
        else:
            pi[r.moves] = r.prior_raw
        return pi

    def best_move(self) -> int:
        r = self.root
        if not r.expanded:
            return PASS
        q = np.where(r.cn > 0, r.cw / np.maximum(r.cn, 1), -2.0)
        best = np.lexsort((-q, -r.cn))[0]  # most visits, then best Q
        return int(r.moves[best])

    def sample_move(self, temperature: float) -> int:
        r = self.root
        if temperature <= 1e-3 or not r.expanded or r.cn.sum() == 0:
            return self.best_move()
        w = np.where(r.cn > 0, r.cn, 0.0) ** (1.0 / temperature)
        return int(r.moves[self.rng.choice(len(w), p=w / w.sum())])

    def move_stats(self, max_pv: int = 10) -> list:
        r = self.root
        out = []
        if not r.expanded:
            return out
        for i, mv in enumerate(r.moves):
            n = int(r.cn[i])
            st = MoveStat(int(mv), n, 0.5, 0.0, float(r.prior_raw[i]), 0.0)
            if n > 0:
                q = r.cw[i] / n
                st.winrate = (q + 1) / 2
                st.score_lead = r.cs[i] / n
                st.lcb = max(0.0, st.winrate - 0.5 / math.sqrt(n))
                st.pv = [int(mv)]
                node = r.children[i]
                while node is not None and node.expanded and len(st.pv) < max_pv and node.cn.sum() > 0:
                    j = int(np.argmax(node.cn))
                    st.pv.append(int(node.moves[j]))
                    node = node.children[j]
            out.append(st)
        out.sort(key=lambda m: (-m.visits, -m.prior))
        return out


def tt_safe_move(search: Search, chosen: int) -> int:
    """If the opponent just passed, passing ends the game: when the current area count loses,
    play the most-visited non-pass move instead (never filling our own single-point eye)."""
    if chosen != PASS:
        return chosen
    b = search.root_board
    if b.passes != 1:
        return PASS
    _, s = terminal_eval(b)
    if s > 0:
        return PASS
    r = search.root
    if not r.expanded:
        return PASS
    me = b.to_move
    best, best_n = PASS, 0
    for i, mv in enumerate(r.moves):
        mv = int(mv)
        if mv == PASS or r.cn[i] <= best_n:
            continue
        x, y = mv % 9, mv // 9
        nbrs = [(x + dx, y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= x + dx < 9 and 0 <= y + dy < 9]
        if all(b.at(ny * 9 + nx) == me for nx, ny in nbrs):
            continue
        best, best_n = mv, r.cn[i]
    return best
