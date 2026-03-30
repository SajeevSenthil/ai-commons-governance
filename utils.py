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


def plot_results(results: dict) -> None:
    plt.figure(figsize=(10, 5))

    for key in results:
        n = len(results[key])
        if n == 0:
            continue

        rounds = len(results[key][0]["health"])
        avg_health = [0.0] * rounds

        for run in results[key]:
            for i in range(rounds):
                avg_health[i] += run["health"][i]

        for i in range(rounds):
            avg_health[i] /= n

        plt.plot(avg_health, label=key)

    plt.xlabel("Round")
    plt.ylabel("Field health")
    plt.title("Commons Governance")
    plt.legend()
    plt.tight_layout()
    plt.show()
