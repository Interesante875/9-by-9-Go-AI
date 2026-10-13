"""Neural-network input planes and the 8 board symmetries (identical encoding to the C++ version).

Planes, from the perspective of the player to move:
   0..7   own / opponent stones for the current and 3 previous positions
   8..10  own groups with 1, 2, >=3 liberties
   11..13 opponent groups with 1, 2, >=3 liberties
   14     legal moves
   15     last move
   16     ones
   17     1 if Black to move
   18     komi from the mover's point of view / 15
   19     1 if the opponent's last move was a pass
"""
from __future__ import annotations

import numpy as np

from .board import BLACK, HISTORY, N, NN, NUM_ACTIONS, PASS, WHITE, Board

INPUT_PLANES = 20


def _sym_tables():
    fwd = np.zeros((8, NUM_ACTIONS), dtype=np.int64)
    for s in range(8):
        for a in range(NN):
            x, y = a % N, a // N
            if s & 1:
                x = N - 1 - x
            if s & 2:
                y = N - 1 - y
            if s & 4:
                x, y = y, x
            fwd[s, a] = y * N + x
        fwd[s, PASS] = PASS
    return fwd


SYM = _sym_tables()  # SYM[s, a] = action a seen through symmetry s


def inverse_sym(s: int) -> int:
    if not s & 4:
        return s
    return 4 | ((s >> 1) & 1) | ((s & 1) << 1)


class Snapshot:
    """History-free description of a position: enough for features and the replay buffer."""

    __slots__ = ("boards", "legal", "to_move", "last_move", "komi")

    def __init__(self, boards, legal, to_move, last_move, komi):
        self.boards = boards        # int8 array (HISTORY, 81), [0] = current
        self.legal = legal          # uint8 array (82,)
        self.to_move = to_move
        self.last_move = last_move
        self.komi = komi


def snapshot(board: Board) -> Snapshot:
    boards = np.empty((HISTORY, NN), dtype=np.int8)
    boards[0] = board.board_list()
    for i, prev in enumerate(board.prev):
        boards[i + 1] = prev
    return Snapshot(boards, np.array(board.legal_mask(), dtype=np.uint8), board.to_move, board.last_move,
                    board.komi)


_NEIGH81 = []
for _a in range(NN):
    _x, _y = _a % N, _a // N
    _NEIGH81.append([b for b, ok in ((_a - 1, _x > 0), (_a + 1, _x < N - 1), (_a - N, _y > 0), (_a + N, _y < N - 1))
                     if ok])


def liberty_map(cells, cap: int = 3) -> np.ndarray:
    """Liberties (capped) of the group at each point, from a flat 81-cell board."""
    out = np.zeros(NN, dtype=np.int8)
    done = [False] * NN
    for a in range(NN):
        c = cells[a]
        if done[a] or c not in (BLACK, WHITE):
            continue
        group, stack, libs = [a], [a], set()
        done[a] = True
        while stack:
            q = stack.pop()
            for r in _NEIGH81[q]:
                cr = cells[r]
                if cr == 0:
                    libs.add(r)
                elif cr == c and not done[r]:
                    done[r] = True
                    group.append(r)
                    stack.append(r)
        out[group] = min(len(libs), cap)
    return out


def encode_snapshot(snap: Snapshot, sym: int = 0, out: np.ndarray | None = None) -> np.ndarray:
    """Returns (INPUT_PLANES, 9, 9) float32 planes, transformed by symmetry ``sym``."""
    if out is None:
        out = np.zeros((INPUT_PLANES, NN), dtype=np.float32)
    else:
        out = out.reshape(INPUT_PLANES, NN)
        out.fill(0.0)
    m = SYM[sym, :NN]
    own = snap.to_move
    opp = WHITE if own == BLACK else BLACK
    for h in range(HISTORY):
        b = snap.boards[h]
        out[2 * h, m[b == own]] = 1.0
        out[2 * h + 1, m[b == opp]] = 1.0
    cur = snap.boards[0]
    libs = liberty_map(cur.tolist())
    for k in (1, 2, 3):
        out[7 + k, m[(cur == own) & (libs == k)]] = 1.0
        out[10 + k, m[(cur == opp) & (libs == k)]] = 1.0
    out[14, m[snap.legal[:NN] == 1]] = 1.0
    if 0 <= snap.last_move < NN:
        out[15, m[snap.last_move]] = 1.0
    out[16] = 1.0
    out[17] = 1.0 if own == BLACK else 0.0
    out[18] = (snap.komi if own == WHITE else -snap.komi) / 15.0
    out[19] = 1.0 if snap.last_move == PASS else 0.0
    return out.reshape(INPUT_PLANES, N, N)


def encode(board: Board, sym: int = 0) -> np.ndarray:
    return encode_snapshot(snapshot(board), sym)
