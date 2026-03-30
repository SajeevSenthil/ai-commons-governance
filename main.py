"""Run the simulation from the command line."""

import argparse

from environment import CommonsSim
from experiments import run_all
from policies import LLMPolicy, RuleBasedPolicy
from utils import avg_last_50, plot_results, summarize, total_llm_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=150, help="Number of rounds to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for agent greed and noise.")
    parser.add_argument("--policy", type=str, default="both", choices=["rule", "llm", "both"])
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name used by the LLM policymaker.",
    )
    args = parser.parse_args()

    if args.policy == "both":
        results = run_all(rounds=args.rounds, seeds=[0, 1, 2, 3, 4], model=args.model)
        stats = summarize(results)
        print("Average field health over last 50 rounds")
        print("rule:", stats["rule"])
        print("llm :", stats["llm"])
        print("rule mean:", round(sum(stats["rule"]) / len(stats["rule"]), 4))
        print("llm mean :", round(sum(stats["llm"]) / len(stats["llm"]), 4))
        print("total llm api cost (usd):", round(total_llm_cost(results), 6))
        plot_results(results)
        return

    sim = CommonsSim(seed=args.seed)
    obs = sim.reset()

    if args.policy == "rule":
        policy = RuleBasedPolicy()
    else:
        policy = LLMPolicy(model=args.model)

    health = []

    for t in range(args.rounds):
        tax = policy.act(obs)
        obs, _, _ = sim.step(tax)
        policy.update(tax, obs)
        health.append(obs["field_health"])
        print(
            t + 1,
            "tax=", round(tax, 4),
            "health=", round(obs["field_health"], 4),
            "harvest=", round(obs["avg_harvest"], 4),
            "reward=", round(obs["avg_reward"], 4),
        )

    print("avg field health last 50 =", round(avg_last_50(health), 4))


if __name__ == "__main__":
    main()
