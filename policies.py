"""Policy code."""

import os
import re

from openai import OpenAI


PRICE = {
    "gpt-4o-mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    }
}


def clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class RuleBasedPolicy:
    def __init__(self) -> None:
        self.base = 0.7
        self.k1 = 2.5
        self.k2 = 1.2
        self.k3 = 0.2
        self.H_target = 0.7
        self.h_target = 0.5
        self.r_target = 0.5
        self.prev_h = None
        self.prev_tax = 0.7
        self.history = []

    def act(self, obs: dict) -> float:
        h = obs["field_health"]
        harvest = obs["avg_harvest"]
        reward = obs["avg_reward"]

        tax = (
            self.base
            + self.k1 * (self.H_target - h)
            + self.k2 * (harvest - self.h_target)
            - self.k3 * (reward - self.r_target)
        )

        if self.prev_h is not None:
            delta_h = h - self.prev_h
            tax += -1.0 * delta_h

        self.prev_h = h

        if h > 0.85:
            tax = max(tax, 0.7)
        elif h > 0.7:
            tax = max(tax, 0.65)
        elif h > 0.55:
            tax = max(tax, 0.75)

        tax = 0.7 * tax + 0.3 * self.prev_tax
        tax = max(0.05, min(1.0, tax))
        self.prev_tax = tax
        return tax

    def update(self, tax: float, obs: dict) -> None:
        self.history.append(
            {
                "tax": round(tax, 4),
                "health": round(obs["field_health"], 4),
                "reward": round(obs["avg_reward"], 4),
            }
        )
        self.history = self.history[-5:]


class LLMPolicy:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=key)
        self.model = model
        self.history = []
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.total_cost_usd = 0.0

    def get_signals(self, obs: dict) -> dict:
        prev_health = self.history[-1]["health"] if self.history else obs["field_health"]
        delta_health = obs["field_health"] - prev_health

        if delta_health > 0:
            trend_direction = "increasing"
        elif delta_health < 0:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        if obs["avg_harvest"] > 0.7:
            harvest_signal = "high"
        elif obs["avg_harvest"] < 0.3:
            harvest_signal = "low"
        else:
            harvest_signal = "moderate"

        if obs["avg_reward"] < 0.3:
            reward_signal = "low"
        elif obs["avg_reward"] > 0.6:
            reward_signal = "high"
        else:
            reward_signal = "moderate"

        if obs["field_health"] < 0.4:
            risk_signal = "HIGH"
        else:
            risk_signal = "LOW"

        return {
            "delta_health": delta_health,
            "trend_direction": trend_direction,
            "harvest_signal": harvest_signal,
            "reward_signal": reward_signal,
            "risk_signal": risk_signal,
        }

    def build_prompt(self, obs: dict) -> str:
        sig = self.get_signals(obs)

        lines = [
            "You are a policymaker managing a shared resource (apple field).",
            "",
            "Your goal is to maintain high field health over time while ensuring agents continue participating.",
            "",
            "Current State:",
            f"Field health: {obs['field_health']:.2f}",
            f"Average harvest: {obs['avg_harvest']:.2f}",
            f"Average reward: {obs['avg_reward']:.2f}",
            "",
            "Trend Signals:",
            f"- Field health change: {sig['delta_health']:.2f} ({sig['trend_direction']})",
            f"- Harvest pressure: {sig['harvest_signal']}",
            f"- Reward level: {sig['reward_signal']}",
            f"- System risk: {sig['risk_signal']}",
            "",
            "Recent History:",
        ]

        if len(self.history) == 0:
            lines.append("No previous rounds.")
        else:
            for i, row in enumerate(self.history, start=1):
                lines.append(
                    f"Round {i}: tax={row['tax']:.2f}, health={row['health']:.2f}, reward={row['reward']:.2f}"
                )

        lines += [
            "",
            "Important rules:",
            "- If field health drops below 0.3, rewards are halved.",
            "- High harvesting damages the field.",
            "- High tax reduces harvesting but may reduce rewards.",
            "- Early intervention is critical to avoid collapse.",
            "",
            "Step 1: Briefly analyze the situation (1 line).",
            "Step 2: Choose the next tax rate.",
            "",
            "Return ONLY the final number between 0 and 1.",
        ]

        return "\n".join(lines)

    def act(self, obs: dict) -> float:
        prompt = self.build_prompt(obs)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        self.add_usage(response)
        text = (response.output_text or "").strip()
        m = re.search(r"\d*\.?\d+", text)
        if m:
            tax = float(m.group())
        else:
            tax = 0.5
        return clamp(tax)

    def add_usage(self, response) -> None:
        meta = getattr(response, "usage", None)
        if not meta:
            return

        in_tokens = getattr(meta, "input_tokens", 0) or 0
        out_tokens = getattr(meta, "output_tokens", 0) or 0

        self.prompt_tokens += int(in_tokens)
        self.output_tokens += int(out_tokens)
        self.total_cost_usd = self.estimate_cost()

    def estimate_cost(self) -> float:
        if self.model not in PRICE:
            return 0.0

        p = PRICE[self.model]
        in_cost = (self.prompt_tokens / 1_000_000.0) * p["input_per_million"]
        out_cost = (self.output_tokens / 1_000_000.0) * p["output_per_million"]
        return in_cost + out_cost

    def update(self, tax: float, obs: dict) -> None:
        self.history.append(
            {
                "tax": float(tax),
                "health": float(obs["field_health"]),
                "reward": float(obs["avg_reward"]),
            }
        )
        self.history = self.history[-5:]
