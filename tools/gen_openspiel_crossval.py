"""Generate random 9x9 Go games with DeepMind's OpenSpiel (open-source reference implementation)
and record, for every position, the set of legal moves plus the final result.

The C++ test `go9_crossval` replays these games through go9's rules engine and checks that
legal moves, game termination and the winner agree exactly.

    pip install open_spiel
    python tools/gen_openspiel_crossval.py --games 60 --out tests/data/openspiel_games.txt
"""
import argparse
import random

import pyspiel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="tests/data/openspiel_games.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    game = pyspiel.load_game(
        "go", {"board_size": 9, "komi": args.komi, "max_game_length": args.max_len}
    )
    pass_action = 81
    lines = [f"# OpenSpiel {pyspiel.__file__.split('/')[-1]} go 9x9 komi={args.komi} games={args.games}"]
    for _ in range(args.games):
        state = game.new_initial_state()
        lines.append(f"game {args.komi} {args.max_len}")
        while not state.is_terminal():
            legal = state.legal_actions()
            # Mostly avoid passing so games reach captures, ko and filled-in endgames,
            # but pass sometimes so the two-pass ending is exercised too.
            non_pass = [a for a in legal if a != pass_action]
            if non_pass and rng.random() > 0.03:
                action = rng.choice(non_pass)
            else:
                action = pass_action
            names = [state.action_to_string(a).split()[1] for a in legal]
            mv = state.action_to_string(action).split()[1]
            lines.append(f"m {mv} {len(names)} {' '.join(names)}")
            state.apply_action(action)
        lines.append(f"end {state.returns()[0]:+.0f}")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.games} games to {args.out}")


if __name__ == "__main__":
    main()
