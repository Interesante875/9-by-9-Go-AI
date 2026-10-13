"""9x9 Go rules engine (pure Python).

Rules (same as the C++ engine):
  * area scoring (Tromp-Taylor / Chinese): stones + empty regions reaching only one colour,
    komi added to White (default 7.0, so jigo is possible);
  * suicide is illegal;
  * positional superko (default) or simple ko;
  * the game ends after two consecutive passes or when ``max_moves`` is reached.

Coordinates: action ``a`` in [0, 81) is the point ``x = a % 9`` (column, left to right),
``y = a // 9`` (row, top to bottom); action 81 is PASS. GTP vertex: column letter
"ABCDEFGHJ"[x] (no "I") and row number ``9 - y``.
"""
from __future__ import annotations

import random

N = 9
NN = N * N
W = N + 2
PADDED = W * W
PASS = NN
NUM_ACTIONS = NN + 1
HISTORY = 4  # current board + 3 previous boards for NN features

EMPTY, BLACK, WHITE, BORDER = 0, 1, 2, 3
SIMPLE_KO, SUPERKO = "simple", "superko"

COLUMNS = "ABCDEFGHJ"
TO_PAD = [(a // N + 1) * W + (a % N + 1) for a in range(NN)]
TO_ACTION = [-1] * PADDED
for _a, _p in enumerate(TO_PAD):
    TO_ACTION[_p] = _a
DIRS = (-W, W, -1, 1)
NEIGHBORS = [tuple(p + d for d in DIRS) for p in range(PADDED)]

_rng = random.Random(0x9E3779B97F4A7C15)
ZOBRIST = [[_rng.getrandbits(64) for _ in range(3)] for _ in range(PADDED)]


def opponent(c: int) -> int:
    return WHITE if c == BLACK else BLACK


def action_to_gtp(a: int) -> str:
    if a == PASS:
        return "pass"
    if not 0 <= a < NN:
        return "invalid"
    return COLUMNS[a % N] + str(N - a // N)


def gtp_to_action(v: str) -> int:
    v = v.strip().upper()
    if v == "PASS":
        return PASS
    if len(v) < 2 or v[0] not in COLUMNS or not v[1:].isdigit():
        return -1
    row = int(v[1:])
    if not 1 <= row <= N:
        return -1
    return (N - row) * N + COLUMNS.index(v[0])


def action_to_sgf(a: int) -> str:
    if a == PASS or not 0 <= a < NN:
        return ""
    return chr(ord("a") + a % N) + chr(ord("a") + a // N)


def sgf_to_action(s: str) -> int:
    if s in ("", "tt"):
        return PASS
    if len(s) != 2:
        return -1
    x, y = ord(s[0]) - ord("a"), ord(s[1]) - ord("a")
    if not (0 <= x < N and 0 <= y < N):
        return -1
    return y * N + x


def score_to_result(score: float) -> str:
    if score == 0:
        return "0"
    return f"{'B' if score > 0 else 'W'}+{abs(score):.1f}"


class Board:
    """Mutable Go position. ``copy()`` is cheap enough to keep one board per MCTS node."""

    __slots__ = ("cells", "to_move", "ko_point", "passes", "move_number", "last_move", "captures",
                 "hash", "history", "prev", "komi", "ko_rule", "max_moves", "_legal")

    def __init__(self, komi: float = 7.0, ko_rule: str = SUPERKO, max_moves: int = 0):
        self.komi = float(komi)
        self.ko_rule = ko_rule
        self.max_moves = max_moves
        self.reset()

    def reset(self) -> None:
        cells = [BORDER] * PADDED
        for p in TO_PAD:
            cells[p] = EMPTY
        self.cells = cells
        self.to_move = BLACK
        self.ko_point = -1
        self.passes = 0
        self.move_number = 0
        self.last_move = -1
        self.captures = [0, 0, 0]  # stones captured BY colour
        self.hash = 0
        self.history = {0}
        empty = (EMPTY,) * NN
        self.prev = (empty,) * (HISTORY - 1)  # previous on-board positions, most recent first
        self._legal = None

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.cells = self.cells[:]
        b.to_move = self.to_move
        b.ko_point = self.ko_point
        b.passes = self.passes
        b.move_number = self.move_number
        b.last_move = self.last_move
        b.captures = self.captures[:]
        b.hash = self.hash
        b.history = self.history  # copy-on-write: replaced (not mutated) in play()
        b.prev = self.prev
        b.komi = self.komi
        b.ko_rule = self.ko_rule
        b.max_moves = self.max_moves
        b._legal = self._legal
        return b

    # ------------------------------------------------------------------ groups
    def _liberties(self, p: int, limit: int = 1000) -> int:
        cells = self.cells
        color = cells[p]
        seen = {p}
        stack = [p]
        libs = set()
        while stack:
            q = stack.pop()
            for r in NEIGHBORS[q]:
                c = cells[r]
                if c == EMPTY:
                    if r not in libs:
                        libs.add(r)
                        if len(libs) >= limit:
                            return len(libs)
                elif c == color and r not in seen:
                    seen.add(r)
                    stack.append(r)
        return len(libs)

    def _group(self, p: int) -> list:
        cells = self.cells
        color = cells[p]
        seen = {p}
        stack = [p]
        while stack:
            q = stack.pop()
            for r in NEIGHBORS[q]:
                if cells[r] == color and r not in seen:
                    seen.add(r)
                    stack.append(r)
        return list(seen)

    def group_liberties(self, a: int) -> int:
        p = TO_PAD[a]
        if self.cells[p] not in (BLACK, WHITE):
            return 0
        return self._liberties(p)

    # ------------------------------------------------------------------ legality
    def is_over(self) -> bool:
        return self.passes >= 2 or (self.max_moves > 0 and self.move_number >= self.max_moves)

    def is_legal(self, a: int) -> bool:
        if self.is_over():
            return False
        if a == PASS:
            return True
        if not 0 <= a < NN:
            return False
        p = TO_PAD[a]
        cells = self.cells
        if cells[p] != EMPTY:
            return False
        if self.ko_rule == SIMPLE_KO and p == self.ko_point:
            return False
        c = self.to_move
        o = opponent(c)
        has_lib = False
        captured = []
        for n in NEIGHBORS[p]:
            cn = cells[n]
            if cn == EMPTY:
                has_lib = True
            elif cn == c:
                if not has_lib and self._liberties(n, 2) >= 2:
                    has_lib = True
            elif cn == o and self._liberties(n, 2) == 1:
                captured.append(n)
        if not has_lib and not captured:
            return False  # suicide
        if self.ko_rule == SUPERKO:
            h = self.hash ^ ZOBRIST[p][c]
            removed = set()
            for n in captured:
                if n in removed:
                    continue
                for q in self._group(n):
                    removed.add(q)
                    h ^= ZOBRIST[q][o]
            if h in self.history:
                return False
        return True

    def legal_mask(self) -> list:
        """List of NUM_ACTIONS 0/1 values (cached until the position changes)."""
        if self._legal is None:
            self._legal = [1 if self.is_legal(a) else 0 for a in range(NUM_ACTIONS)]
        return self._legal

    def legal_actions(self) -> list:
        return [a for a, ok in enumerate(self.legal_mask()) if ok]

    # ------------------------------------------------------------------ play
    def play(self, a: int) -> bool:
        """Plays ``a`` for the side to move. Returns False (state unchanged) if illegal."""
        if not self.is_legal(a):
            return False
        cells = self.cells
        self.prev = (tuple(cells[p] for p in TO_PAD),) + self.prev[: HISTORY - 2]
        c = self.to_move
        o = opponent(c)
        if a == PASS:
            self.passes += 1
            self.ko_point = -1
        else:
            self.passes = 0
            p = TO_PAD[a]
            cells[p] = c
            h = self.hash ^ ZOBRIST[p][c]
            total, cap_point = 0, -1
            for n in NEIGHBORS[p]:
                if cells[n] == o and self._liberties(n, 1) == 0:
                    for q in self._group(n):
                        cells[q] = EMPTY
                        h ^= ZOBRIST[q][o]
                        total += 1
                    cap_point = n
            self.captures[c] += total
            self.hash = h
            self.ko_point = -1
            if total == 1 and all(cells[n] != c for n in NEIGHBORS[p]) and self._liberties(p, 2) == 1:
                self.ko_point = cap_point
        self.history = self.history | {self.hash}
        self.last_move = a
        self.move_number += 1
        self.to_move = o
        self._legal = None
        return True

    # ------------------------------------------------------------------ scoring
    def at(self, a: int) -> int:
        return self.cells[TO_PAD[a]]

    def board_list(self) -> list:
        return [self.cells[p] for p in TO_PAD]

    def stone_count(self, color: int) -> int:
        return sum(1 for p in TO_PAD if self.cells[p] == color)

    def area_ownership(self) -> list:
        """Per point: +1 Black area, -1 White area, 0 neutral."""
        cells = self.cells
        out = [0] * NN
        seen = set()
        for a, p in enumerate(TO_PAD):
            c = cells[p]
            if c == BLACK:
                out[a] = 1
            elif c == WHITE:
                out[a] = -1
            elif p not in seen:
                region, stack = [], [p]
                seen.add(p)
                tb = tw = False
                while stack:
                    q = stack.pop()
                    region.append(q)
                    for r in NEIGHBORS[q]:
                        cr = cells[r]
                        if cr == BLACK:
                            tb = True
                        elif cr == WHITE:
                            tw = True
                        elif cr == EMPTY and r not in seen:
                            seen.add(r)
                            stack.append(r)
                owner = 1 if (tb and not tw) else (-1 if (tw and not tb) else 0)
                for q in region:
                    out[TO_ACTION[q]] = owner
        return out

    def score(self) -> float:
        """Area score Black minus White minus komi (positive = Black wins)."""
        return float(sum(self.area_ownership())) - self.komi

    def winner(self) -> int:
        s = self.score()
        return BLACK if s > 0 else WHITE if s < 0 else EMPTY

    def simple_ko_action(self) -> int:
        return -1 if self.ko_point < 0 else TO_ACTION[self.ko_point]

    def compute_hash(self) -> int:
        h = 0
        for p in TO_PAD:
            c = self.cells[p]
            if c in (BLACK, WHITE):
                h ^= ZOBRIST[p][c]
        return h

    def liberty_map(self, cap: int = 3) -> list:
        out = [0] * NN
        done = set()
        cells = self.cells
        for a, p in enumerate(TO_PAD):
            if p in done or cells[p] not in (BLACK, WHITE):
                continue
            group = self._group(p)
            libs = set()
            for q in group:
                for r in NEIGHBORS[q]:
                    if cells[r] == EMPTY:
                        libs.add(r)
            v = min(len(libs), cap)
            for q in group:
                done.add(q)
                out[TO_ACTION[q]] = v
        return out

    def __str__(self) -> str:
        lines = ["   " + " ".join(COLUMNS)]
        for y in range(N):
            row = []
            for x in range(N):
                a = y * N + x
                c = self.at(a)
                ch = "X" if c == BLACK else "O" if c == WHITE else "."
                if c == EMPTY and (x, y) in ((2, 2), (6, 2), (2, 6), (6, 6), (4, 4)):
                    ch = "+"
                row.append(ch + (")" if a == self.last_move else " "))
            lines.append(f"{N - y:2d} " + "".join(row) + f"{N - y}")
        lines.append("   " + " ".join(COLUMNS))
        lines.append(f"To move: {'Black (X)' if self.to_move == BLACK else 'White (O)'}  Move: {self.move_number}  "
                     f"Captures B:{self.captures[BLACK]} W:{self.captures[WHITE]}  Komi: {self.komi:g}")
        return "\n".join(lines) + "\n"
