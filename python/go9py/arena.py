"""Head-to-head matches between two networks (or a network and a random player)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .board import BLACK, EMPTY, PASS, SUPERKO, WHITE, Board, score_to_result
from .env import make_sgf
from .features import snapshot
from .mcts import MCTSConfig, Search, tt_safe_move


@dataclass
class MatchResult:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_black_wins: int = 0
    a_white_wins: int = 0
    avg_length: float = 0.0
    sgfs: list = field(default_factory=list)

    @property
    def games(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    @property
    def a_score(self) -> float:
        return (self.a_wins + 0.5 * self.draws) / self.games if self.games else 0.5


def elo_from_score(s: float) -> float:
    s = min(0.99, max(0.01, s))
    return 400.0 * math.log10(s / (1 - s))


def random_move(board: Board, rng) -> int:
    """Uniformly random legal move that does not fill one of our own single-point eyes."""
    me = board.to_move
    cands = []
    for a in board.legal_actions():
        if a == PASS:
            continue
        x, y = a % 9, a // 9
        nbrs = [(x + dx, y + dy) for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 0 <= x + dx < 9 and 0 <= y + dy < 9]
        if all(board.at(ny * 9 + nx) == me for nx, ny in nbrs):
            continue
        cands.append(a)
    return int(rng.choice(cands)) if cands else PASS


def play_match(eval_a, eval_b, games=40, sims=64, parallel=20, komi=7.0, max_moves=200, temp_moves=4,
               temperature=0.5, seed=1, stop=None, name_a="A", name_b="B") -> MatchResult:
    """eval_* is an Evaluator or None (random player). Player A is Black in even games."""
    rng = np.random.default_rng(seed)
    evs = [eval_a, eval_b]
    res = MatchResult()
    total_len = 0
    next_game = 0
    slots = []

    def start():
        nonlocal next_game
        b = Board(komi, SUPERKO, max_moves)
        s = {"game": next_game, "board": b, "moves": [],
             "search": [Search(b, MCTSConfig(), seed=int(rng.integers(1 << 31))) for _ in range(2)]}
        next_game += 1
        return s

    def player(s):
        a_black = s["game"] % 2 == 0
        return 0 if a_black == (s["board"].to_move == BLACK) else 1

    while len(slots) < max(1, min(parallel, games)) and next_game < games:
        slots.append(start())

    while slots and not (stop and stop.is_set()):
        ready, batches = [], {0: [], 1: []}
        for s in slots:
            p = player(s)
            search = s["search"][p]
            if evs[p] is None or sims <= 0 or search.root_visits >= sims:
                ready.append(s)
                continue
            leaves = search.gather(1 if search.root_visits == 0 else 4)
            batches[p].append((s, leaves))
        for p in (0, 1):
            snaps = [snapshot(n.board) for _, leaves in batches[p] for _, n in leaves]
            if not snaps:
                continue
            evals = evs[p].evaluate(snaps)
            off = 0
            for s, leaves in batches[p]:
                if leaves:
                    s["search"][p].apply(leaves, evals[off:off + len(leaves)])
                    off += len(leaves)
        ready_ids = {id(s) for s in ready}
        nxt = [s for s in slots if id(s) not in ready_ids]
        for s in ready:
            p = player(s)
            b = s["board"]
            if evs[p] is None:
                a = random_move(b, rng)
            elif sims <= 0:
                a = int(np.argmax(evs[p].evaluate([snapshot(b)])[0].policy))
            else:
                t = temperature if len(s["moves"]) < temp_moves else 0.0
                a = tt_safe_move(s["search"][p], s["search"][p].sample_move(t))
            if not b.play(a):
                a = PASS
                b.play(PASS)
            s["moves"].append(a)
            for q in (0, 1):
                s["search"][q].advance(a)
            if b.is_over():
                w = b.winner()
                a_black = s["game"] % 2 == 0
                a_color = BLACK if a_black else WHITE
                if w == EMPTY:
                    res.draws += 1
                elif w == a_color:
                    res.a_wins += 1
                    if a_black:
                        res.a_black_wins += 1
                    else:
                        res.a_white_wins += 1
                else:
                    res.b_wins += 1
                total_len += len(s["moves"])
                res.sgfs.append(make_sgf(s["moves"], komi, score_to_result(b.score()),
                                         name_a if a_black else name_b, name_b if a_black else name_a))
                if next_game < games:
                    nxt.append(start())
            else:
                nxt.append(s)
        slots = nxt
    res.avg_length = total_len / res.games if res.games else 0.0
    return res
