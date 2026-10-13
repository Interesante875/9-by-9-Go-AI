import io
import json
import math
import os
import threading
import urllib.request

import numpy as np
import pytest
import torch

from go9py.arena import elo_from_score, play_match
from go9py.board import BLACK, EMPTY, PASS, WHITE, Board, gtp_to_action
from go9py.config import TrainConfig
from go9py.engine import PlayEngine
from go9py.features import INPUT_PLANES, snapshot
from go9py.model import Evaluator, GoNet, load_model, save_model
from go9py.selfplay import MoveRecord, SelfPlay, finalize_game
from go9py.trainer import Trainer

torch.set_num_threads(2)


def tiny_net():
    return GoNet(blocks=3, channels=16, head_channels=8)


def tiny_cfg(run_dir):
    return TrainConfig(run_dir=str(run_dir), device="cpu", blocks=3, channels=16, games_per_iter=4, parallel_games=4,
                       full_sims=16, fast_sims=8, full_prob=0.5, max_moves=60, batch_size=32, min_replay=16,
                       replay_capacity=2000, eval_every=0, snapshot_every=0, lr_decay_iters=0)


def test_forward_shapes_and_save_load(tmp_path):
    net = tiny_net().eval()
    x = torch.randn(3, INPUT_PLANES, 9, 9)
    logits, v, s, own = net(x)
    assert logits.shape == (3, 82) and v.shape == (3,) and s.shape == (3,) and own.shape == (3, 81)
    assert v.abs().max() <= 1
    p = str(tmp_path / "m.pt")
    save_model(net, p)
    net2 = load_model(p).eval()
    with torch.no_grad():
        assert torch.allclose(net(x)[0], net2(x)[0])


def test_evaluator_masks_illegal_moves():
    ev = Evaluator(tiny_net(), "cpu")
    b = Board()
    b.play(gtp_to_action("E5"))
    b.play(gtp_to_action("D5"))
    for sym in range(8):
        e = ev.evaluate([snapshot(b)], sym)[0]
        assert abs(e.policy.sum() - 1) < 1e-4
        assert e.policy[gtp_to_action("E5")] == 0 and e.policy[gtp_to_action("D5")] == 0
        assert -1 <= e.value <= 1


def test_selfplay_samples_valid():
    cfg = tiny_cfg("unused")
    games, st = SelfPlay(cfg, Evaluator(tiny_net(), "cpu")).play(4, 123)
    assert len(games) == 4 and st.games == 4
    for g in games:
        assert g.moves
        for s in g.samples:
            assert abs(s["policy"].sum() - 1) < 1e-4
            assert all(s["legal"][a] for a in np.nonzero(s["policy"])[0])
            if g.winner != EMPTY:
                assert s["z"] == (1.0 if s["to_move"] == g.winner else -1.0)
            assert s["legal"][s["action"]] and math.isfinite(s["advantage"])


def test_ppo_rollouts():
    cfg = tiny_cfg("unused")
    cfg.algo = "ppo"
    games, _ = SelfPlay(cfg, Evaluator(tiny_net(), "cpu")).play(3, 5)
    assert len(games) == 3
    for g in games:
        assert len(g.samples) == len(g.moves)
        assert all(s["policy_weight"] == 0 for s in g.samples)


def test_gae_lambda_one_is_monte_carlo():
    b = Board()
    recs = []
    for t in range(5):
        a = gtp_to_action(f"A{t + 1}")
        recs.append(MoveRecord(snapshot(b), np.eye(82, dtype=np.float32)[PASS], 0, 0.1 * t - 0.2, 0, a, True))
        b.play(a)
    samples = finalize_game(recs, b, BLACK, True, 1.0)
    for t, s in enumerate(samples):
        z = 1.0 if t % 2 == 0 else -1.0
        assert s["z"] == z and abs(s["advantage"] - (z - recs[t].nn_value)) < 1e-5 and s["aux_weight"] == 0


def test_training_reduces_loss(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg.lr = 3e-3
    tr = Trainer(cfg)
    games, _ = SelfPlay(cfg, Evaluator(tr.net, "cpu")).play(4, 1)
    for g in games:
        for s in g.samples:
            tr.replay.add(s)
    assert len(tr.replay) > 20
    first = tr.train_steps(5)
    for _ in range(8):
        last = tr.train_steps(10)
    assert math.isfinite(last["total"]) and last["total"] < first["total"]
    assert last["value"] < first["value"] * 0.5 and last["kl"] < first["kl"]


def test_checkpoint_resume(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg.max_iters, cfg.replay_save_every = 2, 1
    assert Trainer(cfg).run() == 0
    for f in ("checkpoint.pt", "model_latest.pt", "replay.npz", "metrics.csv", "config.ini", "train.log"):
        assert os.path.exists(tmp_path / f), f
    tr2 = Trainer(cfg)
    assert tr2.load_checkpoint() and tr2.iter == 2 and len(tr2.replay) > 0
    saved = load_model(str(tmp_path / "model_latest.pt"))
    for a, b in zip(saved.state_dict().values(), tr2.net.state_dict().values()):
        assert torch.equal(a, b)
    cfg.max_iters = 3
    tr3 = Trainer(cfg)
    assert tr3.run() == 0 and tr3.iter == 3
    with open(tmp_path / "metrics.csv") as f:
        assert len(f.read().strip().splitlines()) == 1 + 3  # header + one row per iteration


def test_evaluation_and_arena(tmp_path):
    cfg = tiny_cfg(tmp_path)
    cfg.max_iters, cfg.eval_every, cfg.eval_games, cfg.eval_sims = 2, 1, 4, 8
    assert Trainer(cfg).run() == 0
    rows = open(tmp_path / "eval.csv").read().strip().splitlines()
    assert len(rows) == 3  # header, vs random, vs previous
    r = play_match(Evaluator(tiny_net(), "cpu"), None, games=4, sims=8, max_moves=60)
    assert r.games == 4 and len(r.sgfs) == 4
    assert elo_from_score(0.5) == 0 and elo_from_score(0.75) > 150


def test_play_engine_and_gtp():
    from go9py.gtp import run_gtp
    eng = PlayEngine(Evaluator(tiny_net(), "cpu"))
    script = "protocol_version\n1 boardsize 9\nplay b E5\ngenmove w\nshowboard\nfinal_score\nundo\nlist_commands\nquit\n"
    out = io.StringIO()
    run_gtp(eng, visits=16, inp=io.StringIO(script), out=out)
    text = out.getvalue()
    assert text.startswith("= 2") and "=1 " in text and "kata-analyze" in text
    assert text.count("?") == 0, text
    assert len(eng.env.moves) == 1  # E5 + genmove, then undo


def test_http_api_drives_a_game(tmp_path):
    from go9py.server import make_handler
    from http.server import ThreadingHTTPServer
    web = os.path.join(os.path.dirname(__file__), "..", "..", "web")
    eng = PlayEngine(Evaluator(tiny_net(), "cpu"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(eng, web, str(tmp_path), 16))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, method="POST", body=None):
        req = urllib.request.Request(base + path, data=body if body is not None else (b"" if method == "POST" else None),
                                     method=method)
        with urllib.request.urlopen(req) as r:
            return r.read().decode()

    try:
        assert "<canvas" in call("/", "GET")
        st = json.loads(call("/api/state", "GET"))["state"]
        assert st["to_move"] == BLACK and len(st["board"]) == 81 and len(st["legal"]) == 81
        st = json.loads(call("/api/play?move=E5"))["state"]
        assert st["board"][40] == BLACK and st["to_move"] == WHITE
        res = json.loads(call("/api/genmove?visits=16"))
        assert res["state"]["move_number"] == 2 and res["analysis"]["moves"]
        an = json.loads(call("/api/analyze?visits=32"))["analysis"]
        assert an["visits"] >= 32 and len(an["ownership"]) == 81 and len(an["policy"]) == 82
        assert "SZ[9]" in call("/api/sgf", "GET")
        st = json.loads(call("/api/undo?n=2"))["state"]
        assert st["move_number"] == 0
        st = json.loads(call("/api/sgf", body=b"(;SZ[9]KM[6.5];B[ee];W[dd])"))["state"]
        assert st["moves"] == ["E5", "D6"] and st["komi"] == 6.5
        info = json.loads(call("/api/info", "GET"))
        assert info["blocks"] == 3 and info["channels"] == 16
        with pytest.raises(urllib.error.HTTPError):
            call("/api/play?move=E5")  # occupied
    finally:
        srv.shutdown()
