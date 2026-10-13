"""Environment API tests and cross-validation against DeepMind's OpenSpiel Go implementation."""
import os
import random

import pytest

from go9py.board import BLACK, EMPTY, NN, PASS, SIMPLE_KO, SUPERKO, WHITE, Board, gtp_to_action, sgf_to_action
from go9py.env import GoEnv
from go9py.features import INPUT_PLANES

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "openspiel_games.txt")


def test_reset_and_observation():
    env = GoEnv()
    obs = env.reset()
    assert obs.shape == (INPUT_PLANES, 9, 9)
    assert env.to_move == BLACK and not env.done and len(env.legal_actions()) == NN + 1


def test_two_passes_reward():
    env = GoEnv(7.0)
    env.reset()
    _, r, done, _ = env.step(PASS)
    assert r == 0 and not done
    _, r, done, _ = env.step(PASS)
    assert done and r == 1.0 and env.winner() == WHITE and env.result_string() == "W+7.0"
    _, _, _, info = env.step(gtp_to_action("E5"))
    assert info["illegal"]


def test_illegal_undo_resign():
    env = GoEnv()
    env.step(gtp_to_action("E5"))
    _, _, _, info = env.step(gtp_to_action("E5"))
    assert info["illegal"] and env.to_move == WHITE and len(env.moves) == 1
    h = env.board.hash
    env.step(gtp_to_action("D4"))
    assert env.undo() and env.board.hash == h and env.to_move == WHITE
    env.resign(BLACK)
    assert env.done and env.winner() == WHITE and env.result_string() == "W+R"


def test_sgf_roundtrip():
    rng = random.Random(3)
    for _ in range(5):
        env = GoEnv(6.5)
        while not env.done:
            legal = env.legal_actions()
            a = rng.choice(legal)
            if a == PASS and len(legal) > 1 and env.board.move_number < 120:
                a = legal[0]
            env.step(a)
        copy = GoEnv()
        copy.load_sgf(env.to_sgf())
        assert copy.moves == env.moves and copy.board.hash == env.board.hash
        assert copy.komi == 6.5 and copy.result_string() == env.result_string()
    with pytest.raises(ValueError):
        GoEnv().load_sgf("(;GM[1]SZ[19];B[pd])")
    e = GoEnv()
    e.load_sgf("(;GM[1]SZ[9]KM[5.5];B[ee];W[dd](;B[cc])(;B[gg]))")
    assert len(e.moves) == 3 and e.moves[2] == sgf_to_action("cc")


def test_random_games_terminate_with_correct_reward():
    rng = random.Random(11)
    for _ in range(20):
        env = GoEnv(7.0, SUPERKO, 3 * NN)
        last_r, last_mover = 0.0, EMPTY
        while not env.done:
            last_mover = env.to_move
            _, last_r, _, info = env.step(rng.choice(env.legal_actions()))
            assert not info["illegal"]
        w = env.winner()
        assert last_r == (0.0 if w == EMPTY else (1.0 if w == last_mover else -1.0))


def test_gymnasium_wrapper_if_available():
    pytest.importorskip("gymnasium")
    from go9py.env import GymGoEnv
    env = GymGoEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (INPUT_PLANES, 9, 9) and info["action_mask"].sum() == 82
    obs, r, term, trunc, info = env.step(40)
    assert not term and info["action_mask"][40] == 0


@pytest.mark.skipif(not os.path.exists(DATA), reason="OpenSpiel fixture not found")
def test_crossvalidate_against_openspiel():
    """Every legal-move set, game ending and winner must agree with OpenSpiel (simple-ko rules).
    OpenSpiel declares a draw when a whole-board position repeats; go9py's superko forbids that move."""
    games = positions = repetition_draws = 0
    board = superko = None
    seen, repetition = set(), False
    with open(DATA) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            tok = line.split()
            if tok[0] == "game":
                komi, max_len = float(tok[1]), int(tok[2])
                board, superko = Board(komi, SIMPLE_KO, max_len), Board(komi, SUPERKO, max_len)
                seen, repetition = {board.hash}, False
                games += 1
            elif tok[0] == "m":
                theirs = {gtp_to_action(v) for v in tok[3:3 + int(tok[2])]}
                assert set(board.legal_actions()) == theirs, f"legal move mismatch at {line[:40]}\n{board}"
                positions += 1
                a = gtp_to_action(tok[1])
                assert board.play(a)
                repetition = a != PASS and board.hash in seen
                seen.add(board.hash)
                if repetition:
                    assert not superko.is_legal(a)
                else:
                    assert superko.play(a)
            elif tok[0] == "end":
                ret = int(tok[1])
                if ret == 0 and repetition and not board.is_over():
                    repetition_draws += 1
                    continue
                assert board.is_over()
                ours = 1 if board.winner() == BLACK else -1 if board.winner() == WHITE else 0
                assert ours == ret
    assert games >= 50 and positions > 10000
    print(f"cross-validated {games} games / {positions} positions ({repetition_draws} repetition draws)")
