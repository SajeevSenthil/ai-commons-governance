"""Policy code."""

import os
import re

from google import genai


def clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class RuleBasedPolicy:
    def __init__(self) -> None:
        self.history = []

    def act(self, obs: dict) -> float:
        h = obs["field_health"]
        avg_h = obs["avg_harvest"]
        avg_r = obs["avg_reward"]

        if h < 0.3:
            tax = 0.9
        elif h < 0.5:
            tax = 0.7
        elif h < 0.7:
            tax = 0.5
        else:
            tax = 0.2

        if avg_h > 0.65:
            tax += 0.1
        if avg_r < 0.35:
            tax -= 0.1

        return clamp(tax)

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
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        self.client = genai.Client(api_key=key)
        self.model = model
        self.history = []

    def build_prompt(self, obs: dict) -> str:
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
            "- If field health drops too low, the system collapses and rewards are reduced.",
            "- High tax discourages harvesting but protects the field.",
            "- Low tax encourages harvesting but risks overuse.",
            "",
            "What tax rate (between 0 and 1) should be set for the next round?",
            "",
            "Return ONLY a number between 0 and 1.",
            "Do not explain.",
        ]

        return "\n".join(lines)

    def act(self, obs: dict) -> float:
        prompt = self.build_prompt(obs)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = (response.text or "").strip()
        m = re.search(r"\d*\.?\d+", text)
        if m:
            tax = float(m.group())
        else:
            tax = 0.5
        return clamp(tax)

    def update(self, tax: float, obs: dict) -> None:
        self.history.append(
            {
                "tax": float(tax),
                "health": float(obs["field_health"]),
                "reward": float(obs["avg_reward"]),
            }
        )
        self.history = self.history[-5:]
