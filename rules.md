# Rules

- Agents choose `0.2` or `0.8`
- Tax is in `[0, 1]`
- Reward is `harvest * (1 - tax) + redistribution`
- If `field_health < 0.3`, rewards are halved
- Field health update:
  `delta = a * (1 - avg_harvest) - b * (avg_harvest ** 2)`
