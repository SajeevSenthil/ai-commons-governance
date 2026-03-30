"""Run one simulation and store simple table rows."""

from environment import CommonsSim
from policies import LLMPolicy, RuleBasedPolicy


def run_one(policy_name: str, seed: int, rounds: int = 150, model: str = "gemini-2.5-flash") -> dict:
    sim = CommonsSim(seed=seed)
    sim.reset()
    if policy_name == "rule":
        policy = RuleBasedPolicy()
    else:
        policy = LLMPolicy(model=model)

    health = []
    reward = []
    taxes = []

    for _ in range(rounds):
        obs = {
            "field_health": sim.field_health,
            "avg_harvest": sim.prev_avg_harvest,
            "avg_reward": sim.prev_avg_reward,
        }
        tax = policy.act(obs)
        obs, _, _ = sim.step(tax)
        policy.update(tax, obs)
        health.append(obs["field_health"])
        reward.append(obs["avg_reward"])
        taxes.append(tax)

    return {
        "health": health,
        "reward": reward,
        "tax": taxes,
    }


def run_all(rounds: int = 150, seeds: list[int] | None = None, model: str = "gemini-2.5-flash") -> dict:
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    results = {
        "rule": [],
        "llm": [],
    }

    for seed in seeds:
        results["rule"].append(run_one("rule", seed, rounds, model))
        results["llm"].append(run_one("llm", seed, rounds, model))

    return results
