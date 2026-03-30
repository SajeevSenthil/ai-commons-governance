# ai-commons-governance

A simplified agent-based commons simulation with an LLM policymaker built on Google ADK.

## What is implemented

- 10 self-interested agents with fixed greed values
- Harvest choices `0.2` or `0.8`
- Reward and redistribution exactly based on the tax equations
- Field update using `a * (1 - avg_harvest) - b * (avg_harvest^2)`
- Collapse penalty when field health drops below `0.3`
- A Google ADK policymaker that observes only aggregate state and recent history
- A CLI entry point for running seeded simulations

## What is intentionally deferred

- Rule-based policymaker logic
- Plotting and batch experiment comparison

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key:

```bash
set GOOGLE_API_KEY=your_api_key_here
```

On PowerShell, use:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

## Run

```bash
python main.py --rounds 30 --seed 7 --model gemini-2.0-flash
```

## Notes

- The policymaker receives current field health, previous average harvest rate, previous average reward, and a short history window.
- Round data is stored as simple table-like dictionaries so it is easy to inspect and extend.
- The LLM is instructed to return JSON with a single tax rate.
- If you want, the next step can be adding a rule-based baseline and a comparison script.
