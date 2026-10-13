"""Self-play reinforcement-learning loop with checkpoint/resume.

Run directory contents (everything needed to resume):
  config.ini, checkpoint.pt (model + optimizer + counters), model_latest.pt, replay.npz,
  metrics.csv, eval.csv, train.log, snapshots/, sgf/
"""
from __future__ import annotations

import csv
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .arena import elo_from_score, play_match
from .board import NUM_ACTIONS
from .features import INPUT_PLANES, SYM, Snapshot, encode_snapshot
from .model import Evaluator, GoNet, count_parameters, load_model, pick_device, save_model
from .replay import ReplayBuffer
from .selfplay import SelfPlay

METRIC_COLUMNS = [
    "iter", "hours", "games_total", "samples_total", "buffer", "steps_total", "loss_total", "loss_policy",
    "loss_value", "loss_ownership", "loss_score", "loss_ppo", "entropy", "policy_kl", "value_mse", "ppo_clip_frac",
    "lr", "avg_game_len", "black_winrate", "resign_rate", "false_resign_rate", "selfplay_sec", "train_sec",
    "evals_per_sec", "elo",
]
EVAL_COLUMNS = ["iter", "opponent", "games", "score", "wins", "losses", "elo_delta", "elo"]

log = logging.getLogger("go9py")


def _append_csv(path, columns, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(columns)
        w.writerow(["" if row.get(c) is None else (f"{row[c]:.6g}" if isinstance(row[c], float) else row[c])
                    for c in columns])


def setup_logging(path=None):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    log.addHandler(h)
    if path:
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        log.addHandler(fh)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed * 1000003 + 17)
        self.replay = ReplayBuffer(max(1, cfg.replay_capacity), cfg.per_alpha)
        self.net = GoNet(cfg.blocks, cfg.channels).to(self.device)
        self._make_optimizer()
        self.iter = 0
        self.total_steps = 0
        self.total_games = 0
        self.total_samples = 0
        self.elo = 0.0
        self.hours = 0.0

    def path(self, name):
        return os.path.join(self.cfg.run_dir, name)

    def _make_optimizer(self):
        if self.cfg.optimizer == "sgd":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum,
                                       weight_decay=self.cfg.weight_decay, nesterov=True)
        else:
            self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def current_lr(self):
        c = self.cfg
        if c.lr_decay_iters <= 0:
            return c.lr
        t = min(1.0, self.iter / c.lr_decay_iters)
        return c.lr_min + 0.5 * (c.lr - c.lr_min) * (1 + math.cos(math.pi * t))

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, with_replay: bool):
        t0 = time.time()
        state = {"config": self.net.config, "input_planes": INPUT_PLANES, "model": self.net.state_dict(),
                 "optimizer": self.opt.state_dict(),
                 "counters": {"iter": self.iter, "total_steps": self.total_steps, "total_games": self.total_games,
                              "total_samples": self.total_samples, "elo": self.elo, "hours": self.hours}}
        tmp = self.path("checkpoint.pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.path("checkpoint.pt"))
        save_model(self.net, self.path("model_latest.pt"))
        if with_replay:
            self.replay.save(self.path("replay.npz"))
        log.info(f"checkpoint saved (iter {self.iter}{', with replay buffer' if with_replay else ''}) "
                 f"in {time.time() - t0:.1f}s")

    def load_checkpoint(self) -> bool:
        p = self.path("checkpoint.pt")
        if not os.path.exists(p):
            return False
        ck = torch.load(p, map_location="cpu", weights_only=True)  # tensors + plain values only
        if ck.get("input_planes") != INPUT_PLANES:
            raise ValueError("checkpoint was built for a different input encoding")
        self.net = GoNet(**ck["config"]).to(self.device)
        self.net.load_state_dict(ck["model"])
        self.cfg.blocks, self.cfg.channels = ck["config"]["blocks"], ck["config"]["channels"]
        self._make_optimizer()
        try:
            self.opt.load_state_dict(ck["optimizer"])
        except Exception as e:  # noqa: BLE001
            log.warning(f"could not restore optimizer state ({e}); starting it fresh")
        c = ck["counters"]
        self.iter, self.total_steps = c["iter"], c["total_steps"]
        self.total_games, self.total_samples = c["total_games"], c["total_samples"]
        self.elo, self.hours = c["elo"], c["hours"]
        if self.replay.load(self.path("replay.npz")):
            log.info(f"restored replay buffer: {len(self.replay)} samples")
        return True

    # ------------------------------------------------------------------ SGD
    def train_steps(self, steps: int) -> dict:
        cfg = self.cfg
        st = {k: 0.0 for k in ("total", "policy", "value", "ownership", "score", "ppo", "entropy", "kl", "clip")}
        if steps <= 0 or len(self.replay) == 0:
            st["steps"] = 0
            return st
        self.net.train()
        B = cfg.batch_size
        beta = cfg.per_beta + (1 - cfg.per_beta) * min(1.0, self.iter / max(1, cfg.per_beta_iters))
        use_ppo = cfg.algo != "alphazero" and cfg.w_ppo > 0
        w_policy = 0.0 if cfg.algo == "ppo" else cfg.w_policy
        x = np.empty((B, INPUT_PLANES, 9, 9), dtype=np.float32)
        dev = self.device
        for _ in range(steps):
            idx, wis = self.replay.sample(B, beta, self.rng)
            d = self.replay.batch(idx)
            syms = self.rng.integers(0, 8, B)
            pi = np.zeros((B, NUM_ACTIONS), np.float32)
            mask = np.zeros((B, NUM_ACTIONS), np.float32)
            own = np.zeros((B, 81), np.float32)
            act = np.zeros(B, np.int64)
            for i in range(B):
                s = int(syms[i])
                m = SYM[s]
                snap = Snapshot(d["boards"][i], d["legal"][i], int(d["to_move"][i]), int(d["last_move"][i]),
                                float(d["komi"][i]))
                encode_snapshot(snap, s, x[i])
                pi[i, m] = d["policy"][i]
                mask[i, m] = d["legal"][i]
                own[i, m[:81]] = d["ownership"][i]
                act[i] = m[int(d["action"][i])]
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
            tx, tpi, tmask, town = t(x), t(pi), t(mask), t(own)
            tz, tscore = t(d["z"]), t(d["score"] / 10.0)
            tlogp_old, tadv, twis = t(d["old_logp"]), t(d["advantage"]), t(wis)
            tpw, taw, tact = t(d["policy_weight"]), t(d["aux_weight"]), t(act)

            logits, value, score, ownership = self.net(tx)
            logp = torch.log_softmax(logits.masked_fill(tmask < 0.5, -1e4), dim=1)
            probs = logp.exp()
            l_policy = -(tpi * logp).sum(1) * tpw
            l_value = (value - tz) ** 2
            l_own = ((ownership - town) ** 2).mean(1) * taw
            l_score = F.smooth_l1_loss(score, tscore, reduction="none") * taw
            entropy = -(probs * logp).sum(1)
            per_sample = (w_policy * l_policy + cfg.w_value * l_value + cfg.w_ownership * l_own
                          + cfg.w_score * l_score - cfg.w_entropy * entropy)
            l_ppo = torch.zeros_like(l_value)
            clip_frac = torch.zeros(())
            if use_ppo:
                a_norm = (tadv - tadv.mean()) / (tadv.std() + 1e-6)
                logp_a = logp.gather(1, tact[:, None]).squeeze(1)
                ratio = (logp_a - tlogp_old).clamp(-20, 20).exp()
                l_ppo = -torch.min(ratio * a_norm, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * a_norm)
                clip_frac = ((ratio - 1).abs() > cfg.ppo_clip).float().mean()
                per_sample = per_sample + cfg.w_ppo * l_ppo
            loss = (per_sample * twis).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            self.opt.step()
            with torch.no_grad():
                kl = (tpi * (tpi.clamp_min(1e-8).log() - logp)).sum(1) * tpw
                prio = ((value - tz).abs() * 0.5 + kl).cpu().numpy()
            self.replay.update_priorities(idx, prio)
            st["total"] += loss.item()
            st["policy"] += l_policy.mean().item()
            st["value"] += l_value.mean().item()
            st["ownership"] += l_own.mean().item()
            st["score"] += l_score.mean().item()
            st["ppo"] += l_ppo.mean().item()
            st["entropy"] += entropy.mean().item()
            st["kl"] += kl.mean().item()
            st["clip"] += float(clip_frac)
        self.net.eval()
        for k in st:
            st[k] /= steps
        st["steps"] = steps
        self.total_steps += steps
        return st

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, stop=None):
        cfg = self.cfg
        ref_path = self.path("model_eval_ref.pt")
        current = Evaluator(self.net, self.device)
        t0 = time.time()
        kw = dict(games=cfg.eval_games, sims=cfg.eval_sims, parallel=min(cfg.eval_games, max(2, cfg.parallel_games)),
                  komi=cfg.komi, max_moves=cfg.max_moves, seed=self.iter * 31 + 7, stop=stop)
        if not os.path.exists(ref_path):
            r = play_match(current, None, name_a="current", name_b="random", **kw)
            log.info(f"eval vs random: {r.a_wins}-{r.b_wins} (score {100 * r.a_score:.1f}%), "
                     f"avg length {r.avg_length:.0f}")
            _append_csv(self.path("eval.csv"), EVAL_COLUMNS, {"iter": self.iter, "opponent": -1, "games": r.games,
                        "score": r.a_score, "wins": r.a_wins, "losses": r.b_wins, "elo_delta": 0.0, "elo": self.elo})
        else:
            ref = Evaluator(load_model(ref_path, self.device), self.device)
            r = play_match(current, ref, name_a="current", name_b="previous", **kw)
            delta = elo_from_score(r.a_score)
            self.elo += delta
            log.info(f"eval vs previous checkpoint: {r.a_wins}-{r.b_wins}-{r.draws} (score {100 * r.a_score:.1f}%, "
                     f"as black {r.a_black_wins}, as white {r.a_white_wins}) Elo {delta:+.0f} -> {self.elo:.0f}")
            _append_csv(self.path("eval.csv"), EVAL_COLUMNS, {"iter": self.iter, "opponent": 0, "games": r.games,
                        "score": r.a_score, "wins": r.a_wins, "losses": r.b_wins, "elo_delta": delta, "elo": self.elo})
            if r.sgfs:
                with open(self.path("sgf/eval_latest.sgf"), "w") as f:
                    f.write(r.sgfs[0])
        save_model(self.net, ref_path)
        self.net.eval()
        log.info(f"evaluation took {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ main loop
    def run(self, stop=None) -> int:
        cfg = self.cfg
        os.makedirs(self.path("snapshots"), exist_ok=True)
        os.makedirs(self.path("sgf"), exist_ok=True)
        setup_logging(self.path("train.log"))
        if self.load_checkpoint():
            log.info(f"RESUMED run {cfg.run_dir} at iteration {self.iter} ({self.total_games} games, "
                     f"{self.total_steps} train steps, Elo {self.elo:.0f})")
        else:
            log.info(f"starting a NEW run in {cfg.run_dir}")
        cfg.save_file(self.path("config.ini"))
        log.info(f"device: {self.device} | net: {cfg.blocks} blocks x {cfg.channels} channels "
                 f"({count_parameters(self.net)} params) | algo: {cfg.algo}")
        self.net.eval()
        evaluator = Evaluator(self.net, self.device)
        sp = SelfPlay(cfg, evaluator)
        while not (stop and stop.is_set()):
            if cfg.max_iters > 0 and self.iter >= cfg.max_iters:
                break
            it0 = time.time()
            evaluator.set_net(self.net)
            sp.resign_enabled = self.iter >= cfg.resign_start_iter
            games, s = sp.play(cfg.games_per_iter, cfg.seed * 7777 + self.iter * 104729, stop)
            if stop and stop.is_set():
                break
            new = 0
            for g in games:
                for smp in g.samples:
                    self.replay.add(smp)
                    new += 1
            self.total_games += s.games
            self.total_samples += new
            if cfg.sgf_every > 0 and self.iter % cfg.sgf_every == 0 and games:
                with open(self.path("sgf/latest_selfplay.sgf"), "w") as f:
                    f.write(games[-1].sgf)
            for group in self.opt.param_groups:
                group["lr"] = self.current_lr()
            t1 = time.time()
            ls = {"steps": 0}
            if len(self.replay) >= min(cfg.min_replay, cfg.replay_capacity):
                steps = max(1, min(cfg.max_train_steps, math.ceil(new * cfg.sample_reuse / cfg.batch_size)))
                ls = self.train_steps(steps)
            train_sec = time.time() - t1
            self.iter += 1
            self.hours += (time.time() - it0) / 3600
            avg_len = s.moves / s.games if s.games else 0
            bw = s.black_wins / s.games if s.games else 0
            eps = s.nn_evals / s.seconds if s.seconds > 0 else 0
            if ls["steps"]:
                log.info(f"iter {self.iter} | games {s.games} (len {avg_len:.0f}, B win {100 * bw:.0f}%, resign "
                         f"{s.resigned}) | samples +{new} buf {len(self.replay)} | steps {ls['steps']} | loss "
                         f"{ls['total']:.3f} (pol {ls['policy']:.3f} val {ls['value']:.3f} own {ls['ownership']:.3f} "
                         f"score {ls['score']:.3f} ppo {ls['ppo']:.3f}) ent {ls['entropy']:.2f} | lr "
                         f"{self.current_lr():.2e} | sp {s.seconds:.0f}s ({eps:.0f} evals/s) train {train_sec:.0f}s")
            else:
                log.info(f"iter {self.iter} | games {s.games} (len {avg_len:.0f}) | samples +{new} buf "
                         f"{len(self.replay)} / {cfg.min_replay} needed before training | sp {s.seconds:.0f}s "
                         f"({eps:.0f} evals/s)")
            row = {"iter": self.iter, "hours": self.hours, "games_total": self.total_games,
                   "samples_total": self.total_samples, "buffer": len(self.replay), "steps_total": self.total_steps,
                   "lr": self.current_lr(), "avg_game_len": float(avg_len), "black_winrate": float(bw),
                   "resign_rate": s.resigned / s.games if s.games else 0.0,
                   "false_resign_rate": s.false_resigns / s.noresign_games if s.noresign_games else 0.0,
                   "selfplay_sec": s.seconds, "train_sec": train_sec, "evals_per_sec": float(eps), "elo": self.elo}
            if ls["steps"]:
                row.update({"loss_total": ls["total"], "loss_policy": ls["policy"], "loss_value": ls["value"],
                            "loss_ownership": ls["ownership"], "loss_score": ls["score"], "loss_ppo": ls["ppo"],
                            "entropy": ls["entropy"], "policy_kl": ls["kl"], "value_mse": ls["value"],
                            "ppo_clip_frac": ls["clip"]})
            if cfg.eval_every > 0 and self.iter % cfg.eval_every == 0 and ls["steps"]:
                self.evaluate(stop)
                row["elo"] = self.elo
            _append_csv(self.path("metrics.csv"), METRIC_COLUMNS, row)
            if cfg.save_every > 0 and self.iter % cfg.save_every == 0:
                self.save_checkpoint(cfg.replay_save_every > 0 and self.iter % cfg.replay_save_every == 0)
            if cfg.snapshot_every > 0 and self.iter % cfg.snapshot_every == 0:
                name = f"snapshots/model_iter_{self.iter:06d}.pt"
                save_model(self.net, self.path(name))
                log.info(f"snapshot saved: {name}")
        log.info("stopping: saving final checkpoint (including replay buffer)...")
        self.save_checkpoint(True)
        log.info("done. Resume any time with the same --run_dir.")
        return 0
