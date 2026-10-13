"""Game + search wrapper shared by the web GUI, the terminal mode and GTP."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from .board import BLACK, NN, NUM_ACTIONS, SUPERKO
from .env import GoEnv
from .features import snapshot
from .mcts import MCTSConfig, Search, terminal_eval, tt_safe_move
from .model import load_model

NAN = float("nan")


@dataclass
class Analysis:
    visits: int = 0
    to_move: int = BLACK
    winrate_black: float = 0.5
    score_black: float = 0.0
    nn_value_black: float = 0.0
    moves: list = field(default_factory=list)  # MoveStat, perspective of the player to move
    policy: np.ndarray = field(default_factory=lambda: np.zeros(NUM_ACTIONS))
    ownership_black: np.ndarray = field(default_factory=lambda: np.zeros(NN))


class PlayEngine:
    def __init__(self, evaluator, model_path: str = "", komi: float = 7.0):
        self.ev = evaluator
        self.model_path = model_path or ""
        self.lock = threading.RLock()
        self.env = GoEnv(komi, SUPERKO, 0)
        self.search = Search(self.env.board, MCTSConfig(keep_ownership=True), seed=42)
        self.winrate_hist: list = []
        self.score_hist: list = []
        self._sync_history()

    # ------------------------------------------------------------------ history (win-rate graph)
    def _sync_history(self):
        n = len(self.env.moves) + 1
        del self.winrate_hist[n:]
        del self.score_hist[n:]
        while len(self.winrate_hist) < n:
            self.winrate_hist.append(NAN)
            self.score_hist.append(NAN)
        i = n - 1
        if math.isnan(self.winrate_hist[i]):
            b = self.env.board
            if b.is_over():
                v, s = terminal_eval(b)
            else:
                e = self.ev.evaluate([snapshot(b)])[0]
                v, s = e.value, e.score
            sign = 1 if b.to_move == BLACK else -1
            self.winrate_hist[i] = (sign * v + 1) / 2
            self.score_hist[i] = sign * s

    # ------------------------------------------------------------------ game control
    def new_game(self, komi: float):
        self.env = GoEnv(komi, SUPERKO, 0)
        self.search.reset(self.env.board)
        self.winrate_hist, self.score_hist = [], []
        self._sync_history()

    def play(self, action: int) -> bool:
        if self.env.done:
            return False
        _, _, _, info = self.env.step(action)
        if info["illegal"]:
            return False
        self.search.advance(action)
        if self.search.root_board.hash != self.env.board.hash or \
                self.search.root_board.move_number != self.env.board.move_number:
            self.search.reset(self.env.board)
        self._sync_history()
        return True

    def undo(self) -> bool:
        if self.env.resigned:
            moves = list(self.env.moves)
            self.env = GoEnv(self.env.komi, SUPERKO, 0)
            for a in moves:
                self.env.step(a)
        elif not self.env.undo():
            return False
        self.search.reset(self.env.board)
        self._sync_history()
        return True

    def resign(self, who: int):
        self.env.resign(who)

    def load_sgf(self, text: str):
        env = GoEnv(self.env.komi, SUPERKO, 0)
        env.load_sgf(text)
        self.env = env
        self.search.reset(env.board)
        self.winrate_hist, self.score_hist = [], []
        # Quick network evaluation for every position so the win-rate graph is complete.
        replay = GoEnv(env.komi, SUPERKO, 0)
        for a in env.moves:
            self.winrate_hist.append(NAN)
            self.score_hist.append(NAN)
            b = replay.board
            e = self.ev.evaluate([snapshot(b)])[0]
            sign = 1 if b.to_move == BLACK else -1
            self.winrate_hist[-1] = (sign * e.value + 1) / 2
            self.score_hist[-1] = sign * e.score
            replay.step(a)
        self._sync_history()

    def reload(self, path: str):
        self.ev.set_net(load_model(path, self.ev.device))
        self.model_path = path
        self.search.reset(self.env.board)
        self.winrate_hist = [NAN] * len(self.winrate_hist)
        self.score_hist = [NAN] * len(self.score_hist)
        self._sync_history()

    # ------------------------------------------------------------------ search
    def _analysis(self) -> Analysis:
        b = self.env.board
        r = self.search.root
        a = Analysis(visits=self.search.root_visits, to_move=b.to_move)
        sign = 1 if b.to_move == BLACK else -1
        a.moves = self.search.move_stats(12)
        if a.visits:
            a.winrate_black = (sign * self.search.root_value() + 1) / 2
            a.score_black = sign * self.search.root_score()
        a.nn_value_black = sign * r.nn_value
        if r.expanded:
            a.policy[r.moves] = r.prior_raw
        if r.ownership is not None:
            a.ownership_black = sign * np.asarray(r.ownership)
        i = len(self.env.moves)
        if a.visits and i < len(self.winrate_hist):
            self.winrate_hist[i] = a.winrate_black
            self.score_hist[i] = a.score_black
        return a

    def analyze(self, visits: int, max_total: int = 200000) -> Analysis:
        if not self.env.done:
            self.search.set_root_noise(False)
            self.search.run(self.ev, min(max_total, self.search.root_visits + max(1, visits)), batch=16)
        return self._analysis()

    def choose_move(self, visits: int, allow_resign: bool = True):
        """Returns (action or -2 for resign, Analysis). Does not play the move."""
        self.search.set_root_noise(False)
        self.search.run(self.ev, max(1, visits), batch=16)
        a = self._analysis()
        move = tt_safe_move(self.search, self.search.best_move())
        my_wr = a.winrate_black if a.to_move == BLACK else 1 - a.winrate_black
        if allow_resign and self.env.board.move_number > 30 and a.visits >= 50 and my_wr < 0.03:
            move = -2
        return move, a
