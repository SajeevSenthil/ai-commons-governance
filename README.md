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

   With `a = 0.1` and `b = 0.2`, the damage term often dominates recovery. Even moderate harvesting gradually degrades the field, so the system is unstable unless taxation becomes strong early.

2. Feedback is delayed.

   Field damage accumulates over time. By the time field health becomes visibly bad, the system has already lost a lot of recoverable capacity.

3. The LLM policy is reactive.

   The LLM only observes aggregate signals and short recent history. It does not learn across runs and does not explicitly optimize long-term control. That makes it more likely to keep taxes too low in the early rounds and respond too late.

4. The rule-based policy has built-in safeguards.

   Threshold-based logic acts like a conservative controller. It raises tax more aggressively when health falls, so it slows collapse better than the LLM.

5. Collapse creates a negative loop.

   Once the field enters the collapse region, rewards are halved. That pushes the system into a low-reward regime where recovery becomes much harder.

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

## Result 2: Passing Structured Reasoning Signals

Figure 2 illustrates the evolution of field health over 150 rounds for both rule-based and LLM-based policymakers, where the objective is to maintain sustainability of a shared resource. We observe that both approaches eventually lead to system collapse, but the LLM-based policy degrades significantly faster, around 30 rounds, compared to the rule-based policy, which delays collapse until approximately 60-70 rounds. Quantitatively, the average field health over the final 50 rounds is `0.0234` for the rule-based policy and only `0.001` for the LLM, indicating near-total failure of the LLM approach.

This behavior is primarily due to the underlying environment dynamics, where field health evolves as `delta_H = a(1 - avg_harvest) - b(avg_harvest^2)` with stronger quadratic damage and weak regeneration, making early overharvesting highly detrimental. Additionally, once field health drops below `0.3`, rewards are halved, creating a collapse regime from which recovery is extremely difficult.

The LLM policymaker fails because it operates reactively based on short-term observations and limited history, lacking long-term planning and the ability to anticipate delayed consequences. It tends to apply low taxes in early rounds when the system appears healthy, allowing irreversible damage to accumulate. In contrast, the rule-based policy encodes explicit control logic and safety constraints, enforcing higher taxes when degradation begins and thereby slowing collapse.

Even with prompt refinements and added trend signals, the LLM does not exhibit true learning or optimization over time, highlighting that reasoning alone is insufficient for control in dynamic multi-agent systems with delayed feedback. This result demonstrates that structured, preventive policies outperform reactive LLM-based approaches in such environments and suggests that learning-based methods such as gradient-based or reinforcement learning approaches would be required to achieve stable long-term outcomes.

![Figure 2](Figure_2.png)

## Result 3: Stabilized Control Policy with Softer Agent Response

Figure 3 shows the behavior after jointly refining the environment, the control policy, and the agent response. In this setting, the system no longer collapses over the 50-round sanity check. Instead, field health stays in a high and stable range, roughly between `0.93` and `1.0`, showing that stabilization is possible when incentives and agent behavior are co-designed rather than tuned independently.

This result is important because it shows that policy alone was not enough. Earlier versions of the controller still failed because agents remained too aggressive in choosing high harvest. After reducing that aggressiveness and smoothing the controller response, the field became controllable.

### Environment equations used

Field health update:

`avg_harvest = mean(harvests)`

`delta_H = a * (1 - avg_harvest) * H - b * avg_harvest`

`H(t+1) = clip(H(t) + delta_H, 0, 1)`

Reward:

`reward = harvest * (1 - tax) + redistribution`

`redistribution = total_tax / n_agents`

Smooth collapse penalty:

If `H < 0.3`, then:

`penalty = 0.5 + 0.5 * H`

`reward = reward * penalty`

### Environment parameters used

- `a = 0.2`
- `b = 0.2`
- collapse threshold = `0.3`

### Agent-response equations used

Harvest levels:

- `LOW = 0.2`
- `HIGH = 0.65`

Reward scores:

`R_low = low * (1 - tax)`

`R_high = high * (1 - tax)`

`score_low = R_low`

`score_high = R_high * (1 + 0.3 * greed)`

Softmax-style response:

`temp = 0.6`

`p_high = exp(score_high / temp) / (exp(score_high / temp) + exp(score_low / temp))`

`p_high = min(p_high, 0.8)`

### Control policy used

The final control policy was a continuous controller with smoothing:

`tax = base + k1 * (H_target - H) + k2 * (avg_harvest - h_target) - k3 * (avg_reward - r_target)`

Trend correction:

`tax = tax - 1.0 * delta_H_obs`

where:

`delta_H_obs = H_t - H_{t-1}`

Preventive bias:

- if `H > 0.85`, enforce `tax >= 0.7`
- if `H > 0.7`, enforce `tax >= 0.65`
- if `H > 0.55`, enforce `tax >= 0.75`

Tax smoothing:

`tax = 0.7 * tax + 0.3 * prev_tax`

Clamp:

`tax in [0.05, 1.0]`

### Control-policy parameters used

- `base = 0.7`
- `k1 = 2.5`
- `k2 = 1.2`
- `k3 = 0.2`
- `H_target = 0.7`
- `h_target = 0.5`
- `r_target = 0.5`

### Interpretation

This final run shows that the commons environment can be stabilized, but only after modifying both the policy and the agent response. The key lesson is that stability in multi-agent systems depends on co-design: controller strength, environmental dynamics, and agent sensitivity to incentives must all be aligned.

![Figure 3](Figure_3.png)
