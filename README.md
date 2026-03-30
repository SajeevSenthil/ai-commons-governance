# ai-commons-governance

A simplified agent-based commons simulation comparing a rule-based policymaker and an LLM-based policymaker.

## What is implemented

- 10 self-interested agents with fixed greed values
- Harvest choices `0.2` or `0.8`
- Reward and redistribution exactly based on the tax equations
- Field update using `a * (1 - avg_harvest) - b * (avg_harvest^2)`
- Collapse penalty when field health drops below `0.3`
- A rule-based tax policy
- An LLM-based policy using direct OpenAI API calls
- Comparison across 150 rounds and 5 seeds
- Plotting of field health over time
- Final-50-round health reporting
- LLM API cost logging

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

On PowerShell, use:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Run

```bash
python main.py
```

By default this runs both policies for `150` rounds across `5` seeds, prints the comparison report, and shows the plot.

## Result 1

### Observed outcome

- Field health declines over time for both policies
- The LLM-based policy collapses faster, around 30-40 rounds
- The rule-based policy performs slightly better and delays collapse
- Neither policy maintains long-term sustainability under the current environment settings

![Figure 1](Figure_1.png)

### Why this happens

1. Environmental dynamics are harsh.

   The environment updates field health using:

   `delta_H = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`

   `H(t+1) = clip(H(t) + delta_H, 0, 1)`

   With `a = 0.1` and `b = 0.2`, the damage term often dominates recovery. Even moderate harvesting gradually degrades the field, so the system is unstable unless taxation becomes strong early.

2. Feedback is delayed.

   Field damage accumulates over time. By the time field health becomes visibly bad, the system has already lost a lot of recoverable capacity.

3. The LLM policy is reactive.

   The LLM only observes:

   `field_health`

   `avg_harvest`

   `avg_reward`

   plus the last 5 rounds of history. It does not learn across runs and does not explicitly optimize long-term control. That makes it more likely to keep taxes too low in the early rounds and respond too late.

4. The rule-based policy has built-in safeguards.

   Threshold-based logic acts like a conservative controller. It raises tax more aggressively when health falls, so it slows collapse better than the LLM.

5. Collapse creates a negative loop.

   Rewards are computed as:

   `reward = harvest * (1 - tax) + redistribution`

   `redistribution = total_tax / n_agents`

   and if:

   `field_health < 0.3`

   then:

   `reward = reward / 2`

   That pushes the system into a low-reward regime where recovery becomes much harder.

### Equations used in this project

Agent decision:

`R_low = low * (1 - tax)`

`R_high = high * (1 - tax)`

`score_low = R_low`

`score_high = R_high * (1 + greed)`

`P(high) = exp(score_high) / (exp(score_low) + exp(score_high))`

Reward:

`reward = harvest * (1 - tax) + redistribution`

`redistribution = total_tax / n_agents`

Collapse:

`if field_health < 0.3: reward = reward / 2`

Field health:

`avg_harvest = mean(harvests)`

`delta_H = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`

`H(t+1) = clip(H(t) + delta_H, 0, 1)`

### Key insight

`Policy -> Incentives -> Agent behavior -> System outcome`

The policymaker never controls field health directly. It only changes incentives, and the long-run outcome emerges from agent decisions.

### Interpretation

- Heuristic policies can outperform LLMs in dynamic control tasks
- Early intervention matters a lot in commons problems
- Short-term reasoning is not enough for long-term sustainability

### Possible improvements

- Increase regeneration `a`
- Reduce damage `b`
- Add stronger collapse warnings to the LLM prompt
- Include trend information such as change in field health
- Use a hybrid policy with hard rule-based safety constraints and LLM flexibility
