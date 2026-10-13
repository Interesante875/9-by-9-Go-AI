"""Batched self-play: many games advance together so the network sees large batches."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .board import BLACK, EMPTY, NN, PASS, SUPERKO, WHITE, Board, opponent, score_to_result
from .env import make_sgf
from .features import snapshot
from .mcts import MCTSConfig, Search, tt_safe_move


@dataclass
class MoveRecord:
    snap: object
    pi: np.ndarray
    root_q: float
    nn_value: float
    nn_logp: float
    action: int
    train: bool


@dataclass
class GameResult:
    samples: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    score: float = 0.0
    winner: int = EMPTY
    resigned: bool = False
    resign_allowed: bool = True
    false_resign: bool = False
    sgf: str = ""


@dataclass
class SelfPlayStats:
    games: int = 0
    moves: int = 0
    samples: int = 0
    black_wins: int = 0
    resigned: int = 0
    noresign_games: int = 0
    false_resigns: int = 0
    seconds: float = 0.0
    nn_evals: int = 0


def finalize_game(recs, final_board: Board, winner: int, resigned: bool, gae_lambda: float) -> list:
    """Training samples (outcome, ownership, score, GAE advantages) for a finished game."""
    own_black = np.array(final_board.area_ownership(), dtype=np.int8)
    score_black = 0.0 if resigned else final_board.score()
    T = len(recs)
    adv = np.zeros(T, dtype=np.float32)
    nxt = 0.0
    for t in range(T - 1, -1, -1):
        mover = recs[t].snap.to_move
        r = 0.0
        if t == T - 1:
            r = 0.0 if winner == EMPTY else (1.0 if winner == mover else -1.0)
        v_next = recs[t + 1].nn_value if t + 1 < T else 0.0
        delta = r - v_next - recs[t].nn_value  # V_{t+1} is from the opponent's point of view
        adv[t] = delta - gae_lambda * nxt
        nxt = adv[t]
    samples = []
    for t, rec in enumerate(recs):
        if not rec.train:
            continue
        mover = rec.snap.to_move
        sign = 1 if mover == BLACK else -1
        samples.append({
            "boards": rec.snap.boards, "legal": rec.snap.legal, "to_move": mover,
            "last_move": rec.snap.last_move, "komi": rec.snap.komi, "policy": rec.pi,
            "ownership": own_black * sign,
            "z": 0.0 if winner == EMPTY else (1.0 if winner == mover else -1.0),
            "score": score_black * sign, "root_q": rec.root_q, "old_logp": rec.nn_logp,
            "old_value": rec.nn_value, "advantage": adv[t], "policy_weight": 1.0,
            "aux_weight": 0.0 if resigned else 1.0, "action": rec.action,
        })
    return samples


class _Slot:
    __slots__ = ("active", "board", "search", "target", "full", "allow_resign", "would_resign", "below", "recs",
                 "moves")


class SelfPlay:
    def __init__(self, cfg, evaluator):
        self.cfg = cfg
        self.ev = evaluator
        self.resign_enabled = True

    def play(self, n: int, seed: int, stop=None):
        t0 = time.time()
        e0 = self.ev.evaluations
        games = self._play_policy(n, seed, stop) if self.cfg.algo == "ppo" else self._play_mcts(n, seed, stop)
        st = SelfPlayStats()
        for g in games:
            st.games += 1
            st.moves += len(g.moves)
            st.samples += len(g.samples)
            st.black_wins += g.winner == BLACK
            st.resigned += g.resigned
            if not g.resign_allowed:
                st.noresign_games += 1
                st.false_resigns += g.false_resign
        st.seconds = time.time() - t0
        st.nn_evals = self.ev.evaluations - e0
        return games, st

    def _new_board(self) -> Board:
        return Board(self.cfg.komi, SUPERKO, self.cfg.max_moves)

    def _play_mcts(self, n, seed, stop):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        out, started = [], 0
        slots = []

        def begin_search(s):
            s.full = rng.random() < cfg.full_prob  # playout cap randomisation
            s.target = cfg.full_sims if s.full else cfg.fast_sims
            s.search.set_root_noise(s.full)

        def start(s):
            mc = MCTSConfig(c_puct_init=cfg.c_puct, fpu_reduction=cfg.fpu_reduction, noise_eps=cfg.noise_eps,
                            noise_alpha_total=cfg.noise_alpha_total)
            s.active = True
            s.board = self._new_board()
            s.search = Search(s.board, mc, seed=int(rng.integers(1 << 31)))
            s.allow_resign = self.resign_enabled and cfg.resign_threshold != 0 and rng.random() >= cfg.noresign_frac
            s.would_resign = {BLACK: False, WHITE: False}
            s.below = {BLACK: 0, WHITE: 0}
            s.recs, s.moves = [], []
            begin_search(s)

        def finish(s, winner, resigned):
            g = GameResult(moves=s.moves, winner=winner, resigned=resigned, resign_allowed=s.allow_resign)
            g.false_resign = (not s.allow_resign) and winner != EMPTY and s.would_resign[winner]
            g.score = 0.0 if resigned else s.board.score()
            g.samples = finalize_game(s.recs, s.board, winner, resigned, cfg.gae_lambda)
            res = ("B+R" if winner == BLACK else "W+R") if resigned else score_to_result(g.score)
            g.sgf = make_sgf(s.moves, cfg.komi, res, "go9py self-play", "go9py self-play")
            out.append(g)
            s.active = False

        for _ in range(max(1, cfg.parallel_games)):
            s = _Slot()
            s.active = False
            slots.append(s)
            if started < n:
                start(s)
                started += 1

        k = max(1, cfg.leaves_per_search)
        while not (stop and stop.is_set()):
            active = [s for s in slots if s.active]
            if not active:
                break
            groups, snaps = [], []
            for s in active:
                leaves = s.search.gather(1 if s.search.root_visits == 0 else k)
                groups.append((s, leaves))
                snaps.extend(snapshot(node.board) for _, node in leaves)
            evals = self.ev.evaluate(snaps)
            off = 0
            for s, leaves in groups:
                if leaves:
                    s.search.apply(leaves, evals[off:off + len(leaves)])
                    off += len(leaves)
            for s in active:
                if s.search.root_visits < s.target:
                    continue
                root = s.search.root
                mover = s.board.to_move
                rv = s.search.root_value()
                s.below[mover] = s.below[mover] + 1 if rv < cfg.resign_threshold else 0
                if (self.resign_enabled and cfg.resign_threshold != 0 and s.board.move_number >= cfg.resign_min_moves
                        and s.below[mover] >= max(1, cfg.resign_consecutive)):
                    s.would_resign[mover] = True
                    if s.allow_resign:
                        finish(s, opponent(mover), True)
                        if started < n:
                            start(s)
                            started += 1
                        continue
                temp = cfg.temperature if s.board.move_number < cfg.temp_moves else 0.0
                action = tt_safe_move(s.search, s.search.sample_move(temp))
                prior = 1e-6
                if root.expanded:
                    idx = np.nonzero(root.moves == action)[0]
                    if len(idx):
                        prior = max(prior, float(root.prior_raw[idx[0]]))
                s.recs.append(MoveRecord(snapshot(s.board), s.search.visit_policy(), rv, root.nn_value,
                                         math.log(prior), action, s.full))
                s.board.play(action)
                s.moves.append(action)
                s.search.advance(action)
                if s.board.is_over():
                    finish(s, s.board.winner(), False)
                    if started < n:
                        start(s)
                        started += 1
                    continue
                begin_search(s)
        return out

    def _play_policy(self, n, seed, stop):
        """Pure PPO rollouts: every move is sampled from the raw network policy."""
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        out, started = [], 0
        slots = []
        for _ in range(max(1, cfg.parallel_games)):
            if started < n:
                slots.append({"board": self._new_board(), "recs": [], "moves": []})
                started += 1
        while slots and not (stop and stop.is_set()):
            snaps = [snapshot(s["board"]) for s in slots]
            evals = self.ev.evaluate(snaps)
            nxt = []
            for s, snap, e in zip(slots, snaps, evals):
                pol = e.policy.astype(np.float64)
                a = int(rng.choice(NN + 1, p=pol / pol.sum()))
                if not snap.legal[a]:
                    a = PASS
                s["recs"].append(MoveRecord(snap, e.policy, e.value, e.value, math.log(max(1e-6, e.policy[a])), a,
                                            True))
                s["board"].play(a)
                s["moves"].append(a)
                if s["board"].is_over():
                    b = s["board"]
                    g = GameResult(moves=s["moves"], winner=b.winner(), score=b.score(), resign_allowed=False)
                    g.samples = finalize_game(s["recs"], b, g.winner, False, cfg.gae_lambda)
                    for smp in g.samples:
                        smp["policy_weight"] = 0.0  # no search targets in pure PPO
                    g.sgf = make_sgf(g.moves, cfg.komi, score_to_result(g.score), "go9py ppo", "go9py ppo")
                    out.append(g)
                    if started < n:
                        nxt.append({"board": self._new_board(), "recs": [], "moves": []})
                        started += 1
                else:
                    nxt.append(s)
            slots = nxt
        return out
