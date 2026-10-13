"""Training configuration (same keys and meaning as the C++ version's configs/*.ini)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    # Run / hardware
    run_dir: str = "runs/py"
    device: str = "auto"            # auto | cpu | cuda | cuda:N | mps
    seed: int = 1
    max_iters: int = 0              # 0 = forever (Ctrl+C saves and stops)
    torch_threads: int = 0

    # alphazero (MCTS distillation) | ppo (pure PPO on policy rollouts) | hybrid (both)
    algo: str = "hybrid"

    # Network
    blocks: int = 6
    channels: int = 96

    # Game
    komi: float = 7.0
    max_moves: int = 200

    # Self-play
    games_per_iter: int = 32
    parallel_games: int = 32
    full_sims: int = 160
    fast_sims: int = 32
    full_prob: float = 0.25
    temp_moves: int = 12
    temperature: float = 1.0
    c_puct: float = 1.1
    fpu_reduction: float = 0.2
    noise_eps: float = 0.25
    noise_alpha_total: float = 10.83
    resign_threshold: float = -0.92
    noresign_frac: float = 0.2
    resign_min_moves: int = 20
    resign_start_iter: int = 30
    resign_consecutive: int = 3
    leaves_per_search: int = 2

    # Optimisation
    optimizer: str = "adamw"        # adamw | sgd
    lr: float = 1e-3
    lr_min: float = 1e-4
    lr_decay_iters: int = 3000
    weight_decay: float = 1e-4
    momentum: float = 0.9
    batch_size: int = 256
    sample_reuse: float = 4.0
    min_replay: int = 4096
    max_train_steps: int = 400
    grad_clip: float = 5.0

    # Prioritised replay
    replay_capacity: int = 200000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_iters: int = 1000

    # Loss weights
    w_policy: float = 1.0
    w_value: float = 1.0
    w_ownership: float = 0.5
    w_score: float = 0.1
    w_ppo: float = 0.25
    w_entropy: float = 0.0
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95

    # Checkpoints / evaluation / logging
    save_every: int = 1
    replay_save_every: int = 5
    snapshot_every: int = 25
    eval_every: int = 10
    eval_games: int = 40
    eval_sims: int = 64
    sgf_every: int = 1

    def set(self, key: str, value: str) -> None:
        key = key.replace("-", "_")
        types = {f.name: f.type for f in fields(self)}
        if key not in types:
            raise KeyError(f"unknown config key '{key}'")
        t = types[key]
        try:
            v = int(value) if t in (int, "int") else float(value) if t in (float, "float") else str(value)
        except ValueError:
            raise ValueError(f"bad value for '{key}': {value}") from None
        if key == "algo" and v not in ("alphazero", "ppo", "hybrid"):
            raise ValueError("algo must be alphazero, ppo or hybrid")
        setattr(self, key, v)

    def load_file(self, path: str) -> None:
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError(f"{path}:{ln}: expected key = value")
                k, v = line.split("=", 1)
                try:
                    self.set(k.strip(), v.strip())
                except (KeyError, ValueError) as e:
                    raise ValueError(f"{path}:{ln}: {e}") from None

    def dump(self) -> str:
        lines = ["# go9py training configuration. Use with:  python -m go9py train --config <file>"]
        lines += [f"{k} = {v}" for k, v in asdict(self).items()]
        return "\n".join(lines) + "\n"

    def save_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.dump())
