"""Gym-style 9x9 Go environment with SGF import/export.

    from go9py.env import GoEnv
    env = GoEnv()
    obs = env.reset()                    # (20, 9, 9) float32 planes for the player to move
    while not env.done:
        obs, reward, done, info = env.step(random.choice(env.legal_actions()))

``reward`` is +1 / -1 (0 for jigo) for the player who just moved, and only non-zero at the end.
If ``gymnasium`` is installed, :class:`GymGoEnv` is a ``gymnasium.Env`` version of the same thing.
"""
from __future__ import annotations

import re

import numpy as np

from .board import (BLACK, EMPTY, NN, PASS, SUPERKO, WHITE, Board, action_to_gtp, action_to_sgf,
                    opponent, score_to_result, sgf_to_action)
from .features import encode


class GoEnv:
    def __init__(self, komi: float = 7.0, ko_rule: str = SUPERKO, max_moves: int = 3 * NN):
        self.komi = komi
        self.ko_rule = ko_rule
        self.max_moves = max_moves
        self.board = Board(komi, ko_rule, max_moves)
        self.moves: list[int] = []
        self.resigned = EMPTY

    def reset(self) -> np.ndarray:
        self.board = Board(self.komi, self.ko_rule, self.max_moves)
        self.moves = []
        self.resigned = EMPTY
        return self.observation()

    def observation(self) -> np.ndarray:
        return encode(self.board)

    @property
    def done(self) -> bool:
        return self.board.is_over() or self.resigned != EMPTY

    @property
    def to_move(self) -> int:
        return self.board.to_move

    def legal_actions(self) -> list:
        return self.board.legal_actions()

    def step(self, action: int):
        """Returns (observation, reward, done, info). Illegal actions leave the state unchanged."""
        if self.done or not self.board.play(action):
            return self.observation(), 0.0, self.done, {"illegal": True}
        self.moves.append(action)
        mover = opponent(self.board.to_move)
        reward = 0.0
        if self.board.is_over():
            w = self.board.winner()
            reward = 0.0 if w == EMPTY else (1.0 if w == mover else -1.0)
        return self.observation(), reward, self.done, {"illegal": False}

    def undo(self) -> bool:
        if not self.moves:
            return False
        moves = self.moves[:-1]
        self.reset()
        for a in moves:
            self.step(a)
        return True

    def resign(self, who: int) -> None:
        self.resigned = who

    def winner(self) -> int:
        if self.resigned != EMPTY:
            return opponent(self.resigned)
        return self.board.winner()

    def result_string(self) -> str:
        if self.resigned != EMPTY:
            return "W+R" if self.resigned == BLACK else "B+R"
        if not self.board.is_over():
            return "?"
        return score_to_result(self.board.score())

    def set_komi(self, komi: float) -> None:
        self.komi = komi
        self.board.komi = komi

    def render(self) -> str:
        return str(self.board)

    def to_sgf(self, black: str = "Black", white: str = "White") -> str:
        return make_sgf(self.moves, self.komi, self.result_string(), black, white)

    def load_sgf(self, sgf: str) -> None:
        """Loads the main line of a 9x9 SGF record. Raises ValueError on problems."""
        # Main line = everything up to the first ')' that closes the first variation.
        body = sgf[: sgf.index(")")] if ")" in sgf else sgf
        props = re.findall(r"([A-Z]+)((?:\s*\[(?:\\.|[^\]])*\])+)", body)
        komi = self.komi
        seq = []
        for key, vals in props:
            values = re.findall(r"\[((?:\\.|[^\]])*)\]", vals)
            if key == "SZ" and values[0] != "9":
                raise ValueError(f"only 9x9 games are supported (SZ[{values[0]}])")
            if key == "KM":
                try:
                    komi = float(values[0])
                except ValueError:
                    pass
            if key in ("AB", "AW"):
                raise ValueError("setup stones (AB/AW) are not supported")
            if key in ("B", "W"):
                a = sgf_to_action(values[0])
                if a < 0:
                    raise ValueError(f"bad move {values[0]}")
                seq.append((BLACK if key == "B" else WHITE, a))
        self.komi = komi
        self.reset()
        for color, a in seq:
            if color != self.board.to_move:
                self.step(PASS)
            _, _, _, info = self.step(a)
            if info["illegal"]:
                raise ValueError(f"illegal move {action_to_gtp(a)} at move {len(self.moves) + 1}")


def make_sgf(moves, komi, result, black="Black", white="White", comment="") -> str:
    out = [f"(;GM[1]FF[4]CA[UTF-8]AP[go9py]SZ[9]KM[{komi:g}]RU[Chinese]PB[{black}]PW[{white}]"]
    if result and result != "?":
        out.append(f"RE[{result}]")
    if comment:
        out.append(f"C[{comment}]")
    c = BLACK
    for a in moves:
        out.append(f";{'B' if c == BLACK else 'W'}[{action_to_sgf(a)}]")
        c = opponent(c)
    return "".join(out) + ")\n"


try:  # Optional gymnasium wrapper
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    class GymGoEnv(_gym.Env):
        """gymnasium.Env wrapper. Action space Discrete(82); illegal actions end the episode with -1."""

        metadata = {"render_modes": ["ansi"]}

        def __init__(self, komi: float = 7.0, max_moves: int = 3 * NN):
            self.env = GoEnv(komi, SUPERKO, max_moves)
            from .features import INPUT_PLANES
            self.observation_space = _spaces.Box(-1.0, 1.0, (INPUT_PLANES, 9, 9), np.float32)
            self.action_space = _spaces.Discrete(NN + 1)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.env.reset()
            return obs, {"action_mask": np.array(self.env.board.legal_mask(), dtype=np.int8)}

        def step(self, action):
            obs, reward, done, info = self.env.step(int(action))
            if info["illegal"]:
                return obs, -1.0, True, False, {"illegal": True}
            info["action_mask"] = np.array(self.env.board.legal_mask(), dtype=np.int8)
            return obs, reward, done, False, info

        def render(self):
            return self.env.render()
except ImportError:  # pragma: no cover
    GymGoEnv = None
