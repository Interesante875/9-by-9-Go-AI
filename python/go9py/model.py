"""Residual policy/value network (same architecture as the C++ version) and batched evaluator.

Heads: policy (82 logits), value (tanh, player to move), score lead (/10), ownership (81, tanh).
"""
from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import NN, NUM_ACTIONS
from .features import INPUT_PLANES, SYM, Snapshot, encode_snapshot


def global_pool(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.mean(dim=(2, 3)), x.flatten(2).amax(dim=2)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch: int, gpool: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.gpool_fc = nn.Linear(2 * ch, ch) if gpool else None

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        if self.gpool_fc is not None:  # KataGo-style global pooling bias
            h = h + self.gpool_fc(global_pool(h))[:, :, None, None]
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class GoNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 96, head_channels: int = 32):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "head_channels": head_channels}
        ch, hc = channels, head_channels
        self.stem = nn.Conv2d(INPUT_PLANES, ch, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(ch)
        self.trunk = nn.Sequential(*[ResBlock(ch, gpool=(i % 3 == 2)) for i in range(blocks)])
        self.p_conv = nn.Conv2d(ch, hc, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(hc)
        self.p_gpool = nn.Linear(2 * hc, hc)
        self.p_out = nn.Conv2d(hc, 1, 1)
        self.p_pass = nn.Linear(2 * hc, 1)
        self.v_conv = nn.Conv2d(ch, hc, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(hc)
        self.v_fc = nn.Linear(2 * hc, 128)
        self.v_out = nn.Linear(128, 1)
        self.s_out = nn.Linear(128, 1)
        self.o_out = nn.Conv2d(hc, 1, 1)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        h = self.trunk(h)
        p = F.relu(self.p_bn(self.p_conv(h)))
        pg = global_pool(p)
        p = p + self.p_gpool(pg)[:, :, None, None]
        logits = torch.cat([self.p_out(F.relu(p)).flatten(1), self.p_pass(pg)], dim=1)
        v = F.relu(self.v_bn(self.v_conv(h)))
        ownership = torch.tanh(self.o_out(v).flatten(1))
        vf = F.relu(self.v_fc(global_pool(v)))
        value = torch.tanh(self.v_out(vf)).squeeze(1)
        score = self.s_out(vf).squeeze(1)
        return logits, value, score, ownership


def save_model(net: GoNet, path: str) -> None:
    tmp = path + ".tmp"
    torch.save({"config": net.config, "input_planes": INPUT_PLANES, "state_dict": net.state_dict()}, tmp)
    os.replace(tmp, path)


def load_model(path: str, device="cpu") -> GoNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors + plain values only
    if ckpt.get("input_planes") != INPUT_PLANES:
        raise ValueError(f"{path} was built for a different input encoding")
    net = GoNet(**ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device)


def pick_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warn] CUDA requested but not available, using CPU")
            return torch.device("cpu")
        return torch.device(name)
    if name == "mps" or (name == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                         and not torch.cuda.is_available()):
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Evaluation:
    policy: np.ndarray     # (82,) probabilities, illegal = 0
    value: float           # [-1, 1] for the player to move
    score: float           # score lead for the player to move (points)
    ownership: np.ndarray  # (81,) +1 = player to move owns


class Evaluator:
    """Thread-safe batched inference with random symmetry augmentation."""

    def __init__(self, net: GoNet, device):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.lock = threading.Lock()
        self.rng = random.Random(12345)
        self.evaluations = 0

    @classmethod
    def create(cls, model_path: str | None, device, blocks: int = 6, channels: int = 96) -> "Evaluator":
        net = load_model(model_path, device) if model_path else GoNet(blocks, channels)
        return cls(net, device)

    def set_net(self, net: GoNet) -> None:
        with self.lock:
            self.net = net.to(self.device).eval()

    def evaluate(self, snaps: list[Snapshot], sym: int | None = None) -> list[Evaluation]:
        B = len(snaps)
        if B == 0:
            return []
        syms = [sym if sym is not None else self.rng.randrange(8) for _ in range(B)]
        x = np.empty((B, INPUT_PLANES, 9, 9), dtype=np.float32)
        for i, s in enumerate(snaps):
            encode_snapshot(s, syms[i], x[i])
        with self.lock, torch.no_grad():
            logits, value, score, own = self.net(torch.from_numpy(x).to(self.device))
            self.evaluations += B
            logits = logits.float().cpu().numpy()
            value = value.float().cpu().numpy()
            score = score.float().cpu().numpy()
            own = own.float().cpu().numpy()
        out = []
        for i, s in enumerate(snaps):
            m = SYM[syms[i]]
            lg = logits[i, m]  # back to the real board orientation
            legal = s.legal.astype(bool)
            lg = np.where(legal, lg, -np.inf)
            lg = lg - lg.max()
            p = np.exp(lg)
            p /= p.sum()
            out.append(Evaluation(p.astype(np.float32), float(value[i]), float(score[i] * 10.0),
                                  own[i, m[:NN]].astype(np.float32)))
        return out


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


__all__ = ["GoNet", "Evaluator", "Evaluation", "save_model", "load_model", "pick_device", "count_parameters",
           "NUM_ACTIONS"]
