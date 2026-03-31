# ai-commons-governance

A simplified agent-based commons simulation comparing a rule-based policymaker and an LLM-based policymaker.

## What is implemented

- 10 self-interested agents with fixed greed values
- A shared-field commons environment with tax and redistribution
- A control-based rule policy
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

- Field health declines over time for both policies
- The LLM-based policy collapses faster, around 30-40 rounds
- The rule-based policy performs slightly better and delays collapse
- Neither policy maintains long-term sustainability under the current environment settings

![Figure 1](Figure_1.png)

This first result shows the original system in a highly unstable regime. The environment penalizes harvesting strongly, damage accumulates faster than recovery, and the LLM behaves reactively based on short-term observations. The rule-based policy delays collapse because it applies stronger correction earlier, but both approaches ultimately fail.

Equations used in the original setup:

Environment:

`reward = harvest * (1 - tax) + redistribution`

`redistribution = total_tax / n_agents`

If `H < 0.3`:

`reward = reward / 2`

`avg_harvest = mean(harvests)`

`delta_H = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`

`H(t+1) = clip(H(t) + delta_H, 0, 1)`

Original parameters:

- `a = 0.1`
- `b = 0.2`
- collapse threshold = `0.3`

Original agent response:

- `LOW = 0.2`
- `HIGH = 0.8`

`R_low = low * (1 - tax)`

`R_high = high * (1 - tax)`

`score_low = R_low`

`score_high = R_high * (1 + greed)`

`p_high = exp(score_high) / (exp(score_high) + exp(score_low))`

Original rule logic:

Threshold-style tax adjustments based on health, harvest, and reward bands.

## Result 2: Passing Structured Reasoning Signals

Adding structured reasoning signals to the prompt improves the LLM’s context, but it still does not produce stable long-term control. The LLM remains reactive rather than preventive, so it continues to lag behind the stronger rule-based baseline when the field begins to degrade.

![Figure 2](Figure_2.png)

This stage kept the same unstable L2-damage environment:

`delta_H = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`

with:

- `a = 0.1`
- `b = 0.2`

The main change was in the LLM prompt, where we added:

- field health trend
- harvest pressure
- reward level
- system risk
- last 5 rounds of `(tax, health, reward)`

This improved the LLM’s reasoning context, but it still operated inside the same unstable environment, so collapse remained likely.

## Result 3: Stabilized Control Policy with Softer Agent Response

After refining the environment, softening agent aggressiveness, and replacing the threshold baseline with a continuous controller, the system becomes stabilizable. In the sanity check, field health remains high instead of collapsing, showing that stability depends on co-design of environment dynamics, policy, and agent sensitivity.

![Figure 3](Figure_3.png)

### Final equations and parameters used

Environment:

`reward = harvest * (1 - tax) + redistribution`

`redistribution = total_tax / n_agents`

If `H < 0.3`:

`penalty = 0.5 + 0.5 * H`

`reward = reward * penalty`

`avg_harvest = mean(harvests)`

`delta_H = a * (1 - avg_harvest) * H - b * avg_harvest`

`H(t+1) = clip(H(t) + delta_H, 0, 1)`

Environment parameters:

- `a = 0.2`
- `b = 0.2`
- collapse threshold = `0.3`

This is the key change from the earlier setup:

- old: `delta_H = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`  (L2 damage)
- new: `delta_H = a * (1 - avg_harvest) * H - b * avg_harvest`  (linear damage with health-dependent recovery)

Agent response:

- `LOW = 0.2`
- `HIGH = 0.65`

`R_low = low * (1 - tax)`

`R_high = high * (1 - tax)`

`score_low = R_low`

`score_high = R_high * (1 + 0.3 * greed)`

`temp = 0.6`

`p_high = exp(score_high / temp) / (exp(score_high / temp) + exp(score_low / temp))`

`p_high = min(p_high, 0.8)`

Rule-based control policy:

`tax = base + k1 * (H_target - H) + k2 * (avg_harvest - h_target) - k3 * (avg_reward - r_target)`

If previous health exists:

`tax = tax - 1.0 * (H_t - H_{t-1})`

Preventive floors:

- if `H > 0.85`, `tax >= 0.7`
- if `H > 0.7`, `tax >= 0.65`
- if `H > 0.55`, `tax >= 0.75`

Tax smoothing:

`tax = 0.7 * tax + 0.3 * prev_tax`

Clamp:

`tax in [0.05, 1.0]`

Rule-based parameters:

- `base = 0.7`
- `k1 = 2.5`
- `k2 = 1.2`
- `k3 = 0.2`
- `H_target = 0.7`
- `h_target = 0.5`
- `r_target = 0.5`

LLM policy:

The LLM receives:

- current `field_health`
- current `avg_harvest`
- current `avg_reward`
- health change
- last 5 rounds of `(tax, health, reward)`

LLM post-processing:

`tax = 0.7 * llm_tax + 0.3 * prev_tax`

`tax = max(tax, 0.65)`

If health is falling fast:

`if delta_h < -0.02: tax = tax + 0.1`

Safety floors:

- if `H > 0.85`, `tax >= 0.7`
- if `H < 0.5`, `tax >= 0.8`
- if `H < 0.3`, `tax >= 0.9`

Clamp:

`tax in [0, 1]`

## Result 4: Final Comparison Under Stabilized Dynamics

Under the final stabilized setup, both policymakers maintain high field health over 150 rounds and 5 seeds. The LLM-based policy slightly outperforms the control-based rule policy on the final-50-round metric, while also incurring a small API cost.

Final 50-round field health:

- `rule = [0.9774621218042437, 0.989423095357876, 0.9955594184800001, 0.9879524891710304, 0.9831281726434398]`
- `llm  = [0.9965356991644037, 0.9953180542168001, 0.9990399999999999, 0.9990399999999999, 0.9949543702143999]`

Aggregate results:

- `rule mean = 0.9867`
- `llm mean = 0.9970`
- `total llm api cost = 0.032493 USD`

This final result shows that once the environment is stabilizable and the agent response is softened, the LLM can become competitive and even slightly outperform the control baseline. In other words, the earlier failure was not only about prompt quality but also about the underlying controllability of the system.

![Figure 4](Figure_4.png)
