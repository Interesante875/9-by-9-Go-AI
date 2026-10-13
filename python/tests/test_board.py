import random

import numpy as np
import pytest

from go9py.board import (BLACK, EMPTY, NN, NUM_ACTIONS, PASS, SIMPLE_KO, SUPERKO, WHITE, Board, action_to_gtp,
                         action_to_sgf, gtp_to_action, sgf_to_action)
from go9py.features import INPUT_PLANES, SYM, encode, encode_snapshot, inverse_sym, liberty_map, snapshot

A = gtp_to_action


def play_all(b: Board, moves):
    for m in moves:
        a = A(m)
        assert a >= 0
        assert b.play(a), f"illegal setup move {m}\n{b}"


def test_coordinates_roundtrip():
    for a in range(NUM_ACTIONS):
        assert gtp_to_action(action_to_gtp(a)) == a
        if a != PASS:
            assert sgf_to_action(action_to_sgf(a)) == a
    assert action_to_gtp(0) == "A9" and action_to_gtp(80) == "J1"
    assert gtp_to_action("e5") == 40
    assert gtp_to_action("I5") == -1 and gtp_to_action("A10") == -1


def test_empty_board():
    b = Board()
    assert len(b.legal_actions()) == NUM_ACTIONS
    assert b.to_move == BLACK
    assert b.score() == -7.0 and b.winner() == WHITE and not b.is_over()


def test_capture_single_stone():
    b = Board()
    play_all(b, ["E5", "E4", "A1", "E6", "A2", "D5", "A3"])
    assert b.group_liberties(A("E5")) == 1
    play_all(b, ["F5"])
    assert b.at(A("E5")) == EMPTY
    assert b.captures[WHITE] == 1 and b.captures[BLACK] == 0
    assert b.hash == b.compute_hash()


def test_capture_group_and_two_groups_at_once():
    b = Board()
    play_all(b, ["A9", "C9", "B9", "A8", "J1", "B8"])
    assert b.at(A("A9")) == EMPTY and b.at(A("B9")) == EMPTY and b.captures[WHITE] == 2
    c = Board()
    play_all(c, ["A9", "A8", "C9", "C8", "J1", "D9", "J2", "B9"])
    assert c.at(A("A9")) == EMPTY and c.at(A("C9")) == EMPTY and c.captures[WHITE] == 2
    assert c.hash == c.compute_hash()


def test_suicide_is_illegal():
    b = Board()
    play_all(b, ["E5", "A2", "E6", "B1"])
    assert not b.is_legal(A("A1"))
    assert not b.play(A("A1")) and b.to_move == BLACK
    c = Board()
    play_all(c, ["A8", "A9", "B8", "pass", "C9"])
    assert not c.is_legal(A("B9"))  # multi-stone suicide


def test_suicide_that_captures_is_legal():
    b = Board()
    play_all(b, ["A2", "A1", "E5", "C1", "D1", "B2"])
    assert b.is_legal(A("B1"))
    assert b.play(A("B1")) and b.at(A("A1")) == EMPTY


def make_ko(rule):
    b = Board(7.0, rule)
    play_all(b, ["B9", "C9", "A8", "D8", "B7", "C7", "J1", "B8", "C8"])
    return b


@pytest.mark.parametrize("rule", [SIMPLE_KO, SUPERKO])
def test_ko(rule):
    b = make_ko(rule)
    assert b.at(A("B8")) == EMPTY and b.at(A("C8")) == BLACK and b.to_move == WHITE
    assert not b.is_legal(A("B8"))
    if rule == SIMPLE_KO:
        assert b.simple_ko_action() == A("B8")
    play_all(b, ["J5", "J6"])
    assert b.play(A("B8")) and b.at(A("C8")) == EMPTY
    assert not b.is_legal(A("C8"))


def test_two_passes_end_game_and_max_moves():
    b = make_ko(SIMPLE_KO)
    assert b.play(PASS) and not b.is_over()
    assert b.play(PASS) and b.is_over()
    assert not b.is_legal(PASS) and b.legal_actions() == []
    m = Board(7.0, SUPERKO, 4)
    play_all(m, ["A1", "B1", "C1", "D1"])
    assert m.is_over()


def test_area_scoring_and_dame():
    b = Board()
    for row in range(1, 10):
        play_all(b, [f"E{row}", f"F{row}"])
    assert b.score() == 45 - 36 - 7.0 and b.winner() == BLACK
    own = b.area_ownership()
    assert own[A("A1")] == 1 and own[A("J9")] == -1
    b.komi = 9.5
    assert b.winner() == WHITE
    d = Board()
    for row in range(1, 10):
        play_all(d, [f"D{row}", f"F{row}"])
    own = d.area_ownership()
    assert all(own[A(f"E{r}")] == 0 for r in range(1, 10))
    assert d.score() == -7.0


def test_liberty_maps():
    b = Board()
    play_all(b, ["E5", "A1", "E6"])
    libs = b.liberty_map(10)
    assert libs[A("E5")] == 6 and libs[A("E6")] == 6 and libs[A("A1")] == 2 and libs[A("D4")] == 0
    assert list(liberty_map(b.board_list(), 10)) == libs


def test_random_games_invariants():
    rng = random.Random(7)
    for _ in range(25):
        b = Board(7.0, SUPERKO, 400)
        seen = {b.hash}
        while not b.is_over():
            legal = b.legal_actions()
            a = rng.choice(legal)
            if a == PASS and len(legal) > 1 and b.move_number < 250:
                a = rng.choice(legal[:-1])
            assert b.play(a)
            assert b.hash == b.compute_hash()
            if a != PASS:
                assert b.hash not in seen  # positional superko
                seen.add(b.hash)
            for p in range(NN):
                if b.at(p) != EMPTY:
                    assert b.group_liberties(p) > 0


def test_superko_is_stricter_than_simple_ko():
    rng = random.Random(99)
    for _ in range(10):
        sk, simple = Board(7.0, SUPERKO, 300), Board(7.0, SIMPLE_KO, 300)
        while not sk.is_over():
            legal = sk.legal_actions()
            assert all(simple.is_legal(a) for a in legal)
            a = rng.choice(legal)
            if a == PASS and len(legal) > 1:
                a = rng.choice(legal[:-1])
            assert sk.play(a) and simple.play(a)


def test_copy_is_independent():
    b = Board()
    play_all(b, ["E5"])
    c = b.copy()
    play_all(c, ["D4"])
    assert b.at(A("D4")) == EMPTY and b.move_number == 1 and c.move_number == 2
    assert c.hash in c.history and c.hash not in b.history


def test_symmetries():
    for s in range(8):
        img = set(SYM[s, :NN].tolist())
        assert len(img) == NN
        for a in range(NN):
            assert SYM[inverse_sym(s), SYM[s, a]] == a
        assert SYM[s, PASS] == PASS and SYM[s, 40] == 40


def test_feature_encoding():
    b = Board()
    play_all(b, ["E5", "D4", "C3"])
    f0 = encode(b).reshape(INPUT_PLANES, NN)
    assert f0[0, A("D4")] == 1 and f0[1, A("E5")] == 1 and f0[1, A("C3")] == 1
    assert f0[15, A("C3")] == 1 and f0[17, 0] == 0 and abs(f0[18, 0] - 7 / 15) < 1e-6 and f0[19, 0] == 0
    snap = snapshot(b)
    for s in range(8):
        fs = encode_snapshot(snap, s).reshape(INPUT_PLANES, NN)
        assert np.array_equal(fs[:, SYM[s, :NN]], f0)
    b.play(PASS)
    assert encode(b).reshape(INPUT_PLANES, NN)[19, 40] == 1


def test_snapshot_history():
    b = Board()
    play_all(b, ["E5", "D4", "C3"])
    s = snapshot(b)
    assert s.boards[0][A("C3")] == BLACK and s.boards[1][A("C3")] == EMPTY
    assert s.boards[1][A("D4")] == WHITE and s.boards[2][A("D4")] == EMPTY
    assert s.boards[2][A("E5")] == BLACK and s.boards[3][A("E5")] == EMPTY
    assert s.to_move == WHITE and s.last_move == A("C3") and s.legal[A("C3")] == 0 and s.legal[PASS] == 1
