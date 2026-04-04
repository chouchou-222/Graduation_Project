"""Lookahead optimizer with PyTorch 2.x-compatible state_dict hooks.

This is a lightweight, local copy adapted from Catalyst's Lookahead.
It calls the base Optimizer initializer so that PyTorch can attach
state_dict/load_state_dict hooks, which fixes resume compatibility
with newer torch versions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import torch
from torch.optim import Optimizer


class Lookahead(Optimizer):
    """Implements Lookahead algorithm.

    It has been proposed in `Lookahead Optimizer: k steps forward,
    1 step back`_.

    Adapted from:
    https://github.com/alphadl/lookahead.pytorch (MIT License)
    """

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        # Important: call base Optimizer init so hook dicts exist in torch>=2.0.
        # We set self.optimizer first so add_param_group can run safely.
        self.optimizer = optimizer
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group.setdefault("counter", 0)

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: Optional[Callable] = None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        # Keep Lookahead's param_groups in sync; base optimizer already owns these groups.
        Optimizer.add_param_group(self, param_group)
        param_group["counter"] = 0
