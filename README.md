# Discard Hold'em ML experiments

`bot.py` is the regular heuristic bot. `ml_bot.py` is the second,
protocol-compatible entry point: run `python3 ml_bot.py`. It safely falls back
to the heuristic until `ML_MODEL_PATH` names a valid trained model.

- `features.py`: stable numeric state representation and discrete action buckets.
- `simulator.py`: card-level rollout equity; extend it with engine betting rules.
- `data.py`: append/load JSONL decision records.
- `train.py`: dependency-free linear action-value learner.
- `policy.py`: model inference and conversion back to legal protocol actions.

Generate rows as JSONL in this shape:

```json
{"state_id":"hand-17-decision-2","features":[0.2,0.7],"action":"call","reward":0.14}
```

Train a model:

```sh
python3 train.py data/rollouts.jsonl models/action_value.json
```

The intended first experiment is *counterfactual rollout policy improvement*:
sample a state with the heuristic bot, evaluate **every legal action** from
that exact state through a betting-aware simulator, and store one row per
action with the same `state_id`. Training rejects ordinary hand histories,
because they label only the action taken and caused the losing behaviour.

Test `ml_bot.py` without a model first: it should be behaviourally equivalent
to `bot.py`. Only then set `ML_MODEL_PATH=models/action_value.json` and run a
fresh seeded evaluation against the heuristic baseline.
