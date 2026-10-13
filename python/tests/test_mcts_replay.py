import numpy as np

from go9py.board import NUM_ACTIONS, PASS, WHITE, Board, gtp_to_action
from go9py.mcts import MCTSConfig, Search, tt_safe_move
from go9py.model import Evaluation
from go9py.replay import ReplayBuffer, SumTree


def run_uniform(s: Search, visits: int, batch: int = 4):
    """Know-nothing evaluator (uniform policy, value 0): only exact terminal scores carry information."""
    guard = 0
    while s.root_visits < visits and guard < 100000:
        guard += 1
        leaves = s.gather(1 if s.root_visits == 0 else batch)
        evs = []
        for _, node in leaves:
            legal = np.array(node.board.legal_mask(), dtype=np.float32)
            evs.append(Evaluation(legal / legal.sum(), 0.0, 0.0, np.zeros(81, np.float32)))
        s.apply(leaves, evs)


def test_visit_accounting():
    s = Search(Board())
    run_uniform(s, 300)
    assert s.root_visits >= 300
    assert s.root.cn.sum() == s.root_visits - 1
    assert abs(s.visit_policy().sum() - 1) < 1e-5
    assert s.root.vl == 0 and s.root.cvl.sum() == 0


def test_passes_to_win_when_ahead():
    b = Board(7.0)
    b.play(gtp_to_action("E5"))
    b.play(PASS)
    s = Search(b, seed=3)
    run_uniform(s, 400)
    assert s.best_move() == PASS and s.root_value() > 0.5
    st = s.move_stats()
    assert st[0].move == PASS and st[0].winrate > 0.99 and abs(st[0].score_lead - 74) < 1e-6


def test_does_not_pass_when_behind():
    b = Board(100.0)
    b.play(gtp_to_action("E5"))
    b.play(PASS)
    s = Search(b, seed=5)
    run_uniform(s, 400)
    assert s.best_move() != PASS
    assert all(m.winrate < 0.01 for m in s.move_stats() if m.move == PASS)


def test_tree_reuse_and_noise():
    s = Search(Board(), seed=7)
    run_uniform(s, 200)
    mv = s.best_move()
    child_visits = int(s.root.cn[list(s.root.moves).index(mv)])
    s.advance(mv)
    assert s.root_visits == child_visits and s.root_board.move_number == 1 and s.root_board.to_move == WHITE
    n = Search(Board(), MCTSConfig(root_noise=True), seed=9)
    run_uniform(n, 10)
    assert np.abs(n.root.prior - n.root.prior_raw).sum() > 1e-3 and abs(n.root.prior.sum() - 1) < 1e-6
    n.set_root_noise(False)
    assert np.array_equal(n.root.prior, n.root.prior_raw)


def test_tt_safe_pass():
    cfg = MCTSConfig()
    b = Board(100.0)
    b.play(gtp_to_action("E5"))
    b.play(PASS)
    s = Search(b, cfg, 5)
    run_uniform(s, 200)
    m = tt_safe_move(s, PASS)
    assert m != PASS and b.is_legal(m)
    w = Board(7.0)
    w.play(gtp_to_action("E5"))
    w.play(PASS)
    s2 = Search(w, cfg, 5)
    run_uniform(s2, 50)
    assert tt_safe_move(s2, PASS) == PASS
    c = Board(100.0)
    c.play(gtp_to_action("E5"))
    c.play(gtp_to_action("D5"))
    s3 = Search(c, cfg, 5)
    run_uniform(s3, 50)
    assert tt_safe_move(s3, PASS) == PASS  # opponent did not pass: our pass does not end the game


def test_finished_game_root():
    b = Board()
    b.play(PASS)
    b.play(PASS)
    s = Search(b)
    run_uniform(s, 5)
    assert s.root.terminal and abs(s.root_value() + 1) < 1e-9 and abs(s.root_score() + 7) < 1e-9


def _sample(i):
    return {"action": i, "z": 1.0 if i % 2 else -1.0}


def test_sumtree():
    t = SumTree(5)
    for i, p in enumerate([1, 2, 0, 3, 4]):
        t.set(i, p)
    assert t.total == 10
    assert [t.find(m) for m in (0.5, 1.5, 2.99, 3.01, 6.5)] == [0, 1, 1, 3, 4]


def test_per_sampling_proportional_and_weights():
    rb = ReplayBuffer(4, alpha=1.0, eps=0.0)
    for i in range(4):
        rb.add(_sample(i))
    rb.update_priorities([0, 1, 2, 3], [1, 2, 3, 4])
    idx, w = rb.sample(200000, 1.0, np.random.default_rng(1))
    freq = np.bincount(idx, minlength=4) / len(idx)
    assert np.allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=0.01)
    assert np.allclose(w, 1.0 / (idx + 1), atol=1e-4)


def test_ring_buffer_and_save_load(tmp_path):
    rb = ReplayBuffer(10)
    for i in range(14):
        rb.add(_sample(i))
    rb.update_priorities([1, 2], [3.0, 0.5])
    path = str(tmp_path / "r.npz")
    rb.save(path)
    r2 = ReplayBuffer(10)
    assert r2.load(path)
    assert len(r2) == 10 and r2.total_added == 14
    assert sorted(r2.data["action"][:10].tolist()) == list(range(4, 14))
    assert r2.data["action"][0] == 4  # chronological
    small = ReplayBuffer(4)
    assert small.load(path) and len(small) == 4 and (small.data["action"] >= 10).all()
    assert not ReplayBuffer(4).load(str(tmp_path / "missing.npz"))
    assert rb.data["policy"].shape == (10, NUM_ACTIONS)
