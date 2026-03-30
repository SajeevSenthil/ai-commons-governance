"""Commons simulation close to the reference design."""

from __future__ import annotations

import numpy as np

from agent import HIGH, LOW


class CommonsSim:
    """Simple commons simulator."""

    def __init__(self, n_agents: int = 10, seed: int = 42, a: float = 0.1, b: float = 0.2):
        self.n_agents = n_agents
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.a = a
        self.b = b

        self.low = LOW
        self.high = HIGH
        self.cut = 0.3

        self.field_health = 1.0
        self.greeds = self.rng.uniform(0.0, 1.0, size=n_agents)

        self.prev_avg_harvest = 0.0
        self.prev_avg_reward = 0.0

    def reset(self) -> dict:
        self.field_health = 1.0
        self.greeds = self.rng.uniform(0.0, 1.0, size=self.n_agents)
        self.prev_avg_harvest = 0.0
        self.prev_avg_reward = 0.0

        return {
            "field_health": self.field_health,
            "avg_harvest": self.prev_avg_harvest,
            "avg_reward": self.prev_avg_reward,
        }

    def softmax(self, x1: float, x2: float) -> float:
        e1 = np.exp(x1)
        e2 = np.exp(x2)
        return float(e1 / (e1 + e2))

    def step(self, tax: float) -> tuple[dict, float, dict]:
        tax = max(0.0, min(1.0, float(tax)))

        h0 = self.field_health
        harvests = []

        for i in range(self.n_agents):
            g = float(self.greeds[i])

            r_low = self.low * (1.0 - tax)
            r_high = self.high * (1.0 - tax)

            score_low = r_low
            score_high = r_high * (1.0 + g)

            p_high = self.softmax(score_high, score_low)

            if self.rng.random() < p_high:
                h = self.high
            else:
                h = self.low

            harvests.append(h)

        harvests = np.array(harvests, dtype=float)

        tax_collected = float(np.sum(harvests * tax))
        redistribution = tax_collected / self.n_agents
        rewards = harvests * (1.0 - tax) + redistribution

        collapsed = h0 < self.cut
        if collapsed:
            rewards = rewards * 0.5

        avg_harvest = float(np.mean(harvests))
        delta_h = self.a * (1.0 - avg_harvest) - self.b * (avg_harvest ** 2)
        self.field_health += delta_h
        self.field_health = float(np.clip(self.field_health, 0.0, 1.0))

        avg_reward = float(np.mean(rewards))

        obs = {
            "field_health": self.field_health,
            "avg_harvest": avg_harvest,
            "avg_reward": avg_reward,
        }

        self.prev_avg_harvest = avg_harvest
        self.prev_avg_reward = avg_reward

        info = {
            "harvests": harvests,
            "rewards": rewards,
            "tax_collected": tax_collected,
            "redistribution": redistribution,
            "collapsed": collapsed,
            "field_before": h0,
            "field_after": self.field_health,
            "high_share": float(np.mean(harvests == self.high)),
        }

        return obs, self.field_health, info
