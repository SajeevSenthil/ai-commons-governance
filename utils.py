"""Simple text output helpers."""

import matplotlib.pyplot as plt


def avg_last_50(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = xs[-50:]
    return sum(ys) / len(ys)


def summarize(results: dict) -> dict:
    out = {"rule": [], "llm": []}

    for key in results:
        for run in results[key]:
            out[key].append(avg_last_50(run["health"]))

    return out


def avg_curve(runs: list[dict]) -> list[float]:
    if not runs:
        return []

    rounds = len(runs[0]["health"])
    ys = [0.0] * rounds

    for run in runs:
        for i in range(rounds):
            ys[i] += run["health"][i]

    n = len(runs)
    for i in range(rounds):
        ys[i] /= n

    return ys


def total_llm_cost(results: dict) -> float:
    s = 0.0
    for run in results.get("llm", []):
        s += run.get("cost_usd", 0.0)
    return s


def plot_results(results: dict) -> None:
    plt.figure(figsize=(10, 5))

    for key in results:
        ys = avg_curve(results[key])
        if not ys:
            continue
        plt.plot(ys, label=key)

    plt.xlabel("Round")
    plt.ylabel("Field health")
    plt.title("Commons Governance")
    plt.legend()
    plt.tight_layout()
    plt.show()
