"""Plot go9 training curves (losses, Elo, self-play statistics) from a run directory.

    pip install matplotlib
    python tools/plot_training.py runs/main                # writes runs/main/training_curves.png
    python tools/plot_training.py runs/main --show         # also opens a window
    python tools/plot_training.py runs/main --smooth 0.3   # EMA smoothing factor (0 = raw)

The same curves are shown live in the GUI's "Training" tab (go9 gui --run runs/main).
"""
import argparse
import csv
import math
import os
import sys


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else math.nan) for k, v in r.items()})
        return rows


def ema(values, alpha):
    if alpha <= 0:
        return values
    out, m = [], None
    for v in values:
        if math.isnan(v):
            out.append(math.nan)
            continue
        m = v if m is None else alpha * v + (1 - alpha) * m
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = read_csv(os.path.join(args.run_dir, "metrics.csv"))
    ev = read_csv(os.path.join(args.run_dir, "eval.csv"))
    if not m:
        sys.exit(f"no metrics.csv in {args.run_dir} yet")
    it = [r["iter"] for r in m]

    def series(key, scale=1.0):
        return ema([r.get(key, math.nan) * scale for r in m], args.smooth)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        f"go9 training — {os.path.basename(os.path.abspath(args.run_dir))}  "
        f"(iter {int(it[-1])}, {m[-1]['games_total']:.0f} games, {m[-1]['hours']:.2f} h)",
        fontsize=14,
    )

    ax = axes[0][0]
    ax.plot(it, series("loss_total"), label="total", color="#36c2f0", lw=2)
    ax.set_title("Total loss")
    ax.legend()

    ax = axes[0][1]
    ax.plot(it, series("loss_policy"), label="policy CE (vs MCTS)", color="#f5a65b")
    ax.plot(it, series("policy_kl"), label="KL(pi_mcts || p)", color="#f2d45c")
    ax.plot(it, series("entropy"), label="policy entropy", color="#9aa0a6", ls="--")
    ax.set_title("Policy")
    ax.legend()

    ax = axes[0][2]
    ax.plot(it, series("loss_value"), label="value MSE", color="#7be08f")
    ax.plot(it, series("loss_ownership"), label="ownership MSE", color="#b18cff")
    ax.plot(it, series("loss_score"), label="score Huber", color="#c8ccd4")
    ax.set_title("Value / ownership / score")
    ax.legend()

    ax = axes[1][0]
    ax.plot(it, series("loss_ppo"), label="PPO clipped surrogate", color="#ef6a5b")
    ax2 = ax.twinx()
    ax2.plot(it, series("ppo_clip_frac"), label="clip fraction", color="#9aa0a6", ls="--")
    ax.set_title("PPO term")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1][1]
    rows = [r for r in ev if r.get("opponent", 0) == 0]
    if rows:
        ax.plot([r["iter"] for r in rows], [r["elo"] for r in rows], "o-", color="#7be08f", label="Elo")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no evaluations yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Elo vs previous checkpoints")

    ax = axes[1][2]
    ax.plot(it, series("avg_game_len"), label="avg game length", color="#f5a65b")
    ax2 = ax.twinx()
    ax2.plot(it, series("black_winrate", 100), label="Black win %", color="#c8ccd4", ls="--")
    ax.set_title("Self-play games")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    for row in axes:
        for a in row:
            a.set_xlabel("iteration")
            a.grid(alpha=0.15)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "training_curves.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
