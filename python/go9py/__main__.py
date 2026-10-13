"""Command-line interface:  python -m go9py <command> [options]

  train     self-play RL training (resumes automatically)
  gui       browser GUI: play, KataGo-style analysis, training dashboard
  play      play the AI in the terminal
  gtp       GTP engine for Sabaki / Lizzie / GoGui
  match     head-to-head evaluation between two models (or random)
  selftest  random games with rules invariants checked
  bench     network and MCTS speed
  config    print the default training configuration
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import torch


def _add_common(p, model=True):
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N | mps")
    p.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads (0 = default)")
    if model:
        p.add_argument("--run", default="", help="run directory (uses <run>/model_latest.pt)")
        p.add_argument("--model", default="", help="model file (.pt)")
        p.add_argument("--komi", type=float, default=7.0)


def _resolve_model(args) -> str:
    if args.model:
        if not os.path.exists(args.model):
            sys.exit(f"Model file not found: {args.model}")
        return args.model
    if args.run:
        p = os.path.join(args.run, "model_latest.pt")
        if os.path.exists(p):
            return p
        print(f"[warn] {p} does not exist yet - using an untrained network")
    return ""


def _engine(args):
    from .engine import PlayEngine
    from .model import Evaluator, pick_device
    if args.threads:
        torch.set_num_threads(args.threads)
    path = _resolve_model(args)
    return PlayEngine(Evaluator.create(path or None, pick_device(args.device)), path, args.komi)


def cmd_train(argv) -> int:
    from .config import TrainConfig
    from .trainer import Trainer
    p = argparse.ArgumentParser(prog="go9py train", description="Self-play training. Any config key can be "
                                "overridden with --key value (see `python -m go9py config`).")
    p.add_argument("--config", default="")
    p.add_argument("--run_dir", "--run", default="")
    args, rest = p.parse_known_args(argv)
    cfg = TrainConfig()
    run_dir = args.run_dir or cfg.run_dir
    saved = os.path.join(run_dir, "config.ini")
    if os.path.exists(saved):
        cfg.load_file(saved)
        print(f"Resuming from {run_dir} (saved config + your overrides)")
    if args.config:
        cfg.load_file(args.config)
    i = 0
    while i < len(rest):
        a = rest[i]
        if not a.startswith("--"):
            sys.exit(f"unexpected argument {a}")
        if "=" in a:
            k, v = a[2:].split("=", 1)
        else:
            if i + 1 >= len(rest):
                sys.exit(f"missing value for {a}")
            k, v = a[2:], rest[i + 1]
            i += 1
        try:
            cfg.set(k, v)
        except (KeyError, ValueError) as e:
            sys.exit(f"Argument error: {e}")
        i += 1
    cfg.run_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)
    print("Configuration:\n" + "".join(f"  {line}\n" for line in cfg.dump().splitlines()[1:]), end="")
    stop = threading.Event()

    def on_sigint(sig, frame):
        if stop.is_set():
            print("\nSecond interrupt - exiting immediately (last checkpoint is kept).")
            os._exit(130)
        print("\nInterrupt received - finishing the current step and saving a checkpoint "
              "(press Ctrl+C again to abort)...")
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)
    return Trainer(cfg).run(stop)


def cmd_gui(argv) -> int:
    from .server import run_server
    p = argparse.ArgumentParser(prog="go9py gui")
    _add_common(p)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--visits", type=int, default=400)
    p.add_argument("--web", default=None, help="folder with index.html (default: the repository's web/)")
    p.add_argument("--open", action="store_true", help="open the browser")
    args = p.parse_args(argv)
    eng = _engine(args)
    print(f"Model: {eng.model_path or '(untrained)'} | device: {eng.ev.device}")
    run_server(eng, args.host, args.port, args.web, args.run, args.visits, args.open)
    return 0


def cmd_gtp(argv) -> int:
    from .gtp import run_gtp
    p = argparse.ArgumentParser(prog="go9py gtp")
    _add_common(p)
    p.add_argument("--visits", type=int, default=400)
    args = p.parse_args(argv)
    return run_gtp(_engine(args), args.visits)


def cmd_play(argv) -> int:
    from .board import BLACK, WHITE, gtp_to_action, action_to_gtp
    p = argparse.ArgumentParser(prog="go9py play")
    _add_common(p)
    p.add_argument("--visits", type=int, default=400)
    p.add_argument("--color", default="black")
    args = p.parse_args(argv)
    eng = _engine(args)
    human = WHITE if args.color.lower().startswith("w") else BLACK
    print(f"You are {'Black (X)' if human == BLACK else 'White (O)'}. Enter moves like D5, 'pass', 'undo', "
          "'resign' or 'quit'.")
    while not eng.env.done:
        print("\n" + eng.env.render())
        if eng.env.to_move == human:
            try:
                s = input("your move> ").strip()
            except EOFError:
                break
            if s == "quit":
                break
            if s == "undo":
                eng.undo()
                eng.undo()
                continue
            if s == "resign":
                eng.resign(human)
                break
            a = gtp_to_action(s)
            if a < 0 or not eng.play(a):
                print(f"illegal move: {s}")
        else:
            move, an = eng.choose_move(args.visits, True)
            if move == -2:
                print("AI resigns.")
                eng.resign(eng.env.to_move)
                break
            wr = an.winrate_black if an.to_move == BLACK else 1 - an.winrate_black
            lead = an.score_black if an.to_move == BLACK else -an.score_black
            print(f"AI plays {action_to_gtp(move)}  (AI win rate {100 * wr:.1f}%, score lead {lead:+.1f}, "
                  f"{an.visits} visits)")
            eng.play(move)
    print("\n" + eng.env.render() + f"Result: {eng.env.result_string()}\n" + eng.env.to_sgf())
    return 0


def cmd_match(argv) -> int:
    from .arena import elo_from_score, play_match
    from .model import Evaluator, pick_device
    p = argparse.ArgumentParser(prog="go9py match")
    p.add_argument("--a", required=True, help="model file or 'random'")
    p.add_argument("--b", default="random")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--komi", type=float, default=7.0)
    p.add_argument("--sgf", default="")
    _add_common(p, model=False)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = pick_device(args.device)
    load = lambda m: None if m == "random" else Evaluator.create(m, dev)  # noqa: E731
    t0 = time.time()
    r = play_match(load(args.a), load(args.b), games=args.games, sims=args.sims, parallel=min(args.games, 32),
                   komi=args.komi, name_a=args.a, name_b=args.b)
    print(f"{args.a} vs {args.b}: {r.a_wins} wins, {r.b_wins} losses, {r.draws} draws "
          f"(score {100 * r.a_score:.1f}%, Elo diff {elo_from_score(r.a_score):+.0f})")
    print(f"  as Black: {r.a_black_wins} wins, as White: {r.a_white_wins} wins, avg length {r.avg_length:.1f} moves, "
          f"{time.time() - t0:.1f}s")
    if args.sgf:
        with open(args.sgf, "w") as f:
            f.write("".join(r.sgfs))
    return 0


def cmd_selftest(argv) -> int:
    import random
    from .board import BLACK, EMPTY, NN, PASS
    from .env import GoEnv
    rng = random.Random(2024)
    t0, moves = time.time(), 0
    for g in range(60):
        env = GoEnv(7.0, max_moves=300)
        while not env.done:
            legal = env.legal_actions()
            a = rng.choice(legal)
            if a == PASS and len(legal) > 1 and env.board.move_number < 150:
                a = rng.choice(legal[:-1])
            _, _, _, info = env.step(a)
            if info["illegal"]:
                print("selftest FAILED: legal action rejected")
                return 1
            moves += 1
            b = env.board
            if b.hash != b.compute_hash():
                print("selftest FAILED: hash mismatch")
                return 1
            for p in range(NN):
                if b.at(p) != EMPTY and b.group_liberties(p) == 0:
                    print("selftest FAILED: group without liberties\n" + str(b))
                    return 1
    print(f"selftest OK: 60 random games, {moves} moves ({moves / (time.time() - t0):.0f} moves/s incl. checks)")
    return 0


def cmd_bench(argv) -> int:
    from .board import Board
    from .features import snapshot
    from .mcts import Search
    from .model import Evaluator, pick_device
    p = argparse.ArgumentParser(prog="go9py bench")
    _add_common(p)
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    ev = Evaluator.create(_resolve_model(args) or None, pick_device(args.device))
    snap = snapshot(Board())
    for bs in (1, 16, 64, 256):
        ev.evaluate([snap] * bs)
        t0, reps = time.time(), 0
        while time.time() - t0 < 2:
            ev.evaluate([snap] * bs)
            reps += 1
        print(f"batch {bs:3d}: {bs * reps / (time.time() - t0):8.0f} positions/s")
    s = Search(Board())
    t0 = time.time()
    s.run(ev, 800, batch=16)
    dt = time.time() - t0
    print(f"MCTS: 800 visits in {dt:.2f}s ({800 / dt:.0f} visits/s) on {ev.device}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "config":
        from .config import TrainConfig
        print(TrainConfig().dump(), end="")
        return 0
    table = {"train": cmd_train, "gui": cmd_gui, "gtp": cmd_gtp, "play": cmd_play, "match": cmd_match,
             "selftest": cmd_selftest, "bench": cmd_bench}
    if cmd not in table:
        print(__doc__)
        return 1
    return table[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
