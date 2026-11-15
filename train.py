"""Train a lightweight action-value model from rollout-labelled JSONL data.

Usage: python train.py data/rollouts.jsonl models/action_value.json
Each row must have: {"features": [...], "action": "call", "reward": 0.13}.
"""

import json
import math
import random
import sys
from pathlib import Path

from data import load_transitions
from features import ACTIONS


def train(rows, epochs=80, learning_rate=0.04, l2=1e-4):
    rows = list(rows)
    if not rows:
        raise ValueError("no training rows")
    # Q(s, a) can only compare actions when at least two actions were measured
    # from the same state.  A terminal outcome from one historical action does
    # not label the actions that were never taken.
    state_actions = {}
    for row in rows:
        state_id = row.get("state_id")
        if state_id is None:
            raise ValueError("each row needs a state_id from a counterfactual rollout")
        state_actions.setdefault(str(state_id), set()).add(row["action"])
    valid_states = {key for key, actions in state_actions.items() if len(actions) >= 2}
    rows = [row for row in rows if str(row["state_id"]) in valid_states]
    if not rows:
        raise ValueError("need at least one state with rewards for two or more legal actions")
    width = len(rows[0]["features"])
    weights = {action: [0.0] * (width + 1) for action in ACTIONS}
    scale = max(1.0, max(abs(float(r["reward"])) for r in rows))
    rng = random.Random(7)
    for epoch in range(epochs):
        rng.shuffle(rows)
        rate = learning_rate / math.sqrt(1 + epoch * 0.08)
        for row in rows:
            action = row["action"]
            if action not in weights or len(row["features"]) != width:
                continue
            x = [1.0] + [float(v) for v in row["features"]]
            target = float(row["reward"]) / scale
            prediction = sum(w * v for w, v in zip(weights[action], x))
            error = max(-3.0, min(3.0, prediction - target))
            weights[action] = [w - rate * (error * v + l2 * w) for w, v in zip(weights[action], x)]
    return {"version": 2, "feature_count": width, "reward_scale": scale, "weights": weights,
            "metadata": {"label_source": "counterfactual_rollout",
                         "states": len(valid_states), "samples": len(rows)}}


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: python train.py INPUT.jsonl OUTPUT.json")
    model = train(load_transitions(argv[1]))
    target = Path(argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)

