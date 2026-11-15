# Discard Hold'em ML experiments

`bot.py` remains the tournament entry point. The surrounding modules support
offline experiments without adding a runtime dependency to the bot.

- `features.py`: stable numeric state representation and discrete action buckets.
- `simulator.py`: card-level rollout equity; extend it with engine betting rules.
- `data.py`: append/load JSONL decision records.
- `train.py`: dependency-free linear action-value learner.
- `policy.py`: model inference and conversion back to legal protocol actions.

Generate rows as JSONL in this shape:

```json
{"features":[0.2,0.7],"action":"call","reward":0.14}
```

Train a model:

```sh
python3 train.py data/rollouts.jsonl models/action_value.json
```

The intended first experiment is *rollout policy improvement*: play or sample
states with the heuristic bot, evaluate every legal action through a local
simulator, store its terminal chip EV as `reward`, then train on those labels.
Keep the heuristic policy as a fallback until the model beats it on held-out,
seeded matches.
