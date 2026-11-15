"""Live inference for the action-value model, with protocol-safe translation."""

import json
from pathlib import Path

from features import ACTIONS, action_bucket, state_features


class ActionValuePolicy:
    def __init__(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.weights = payload["weights"]
        self.feature_count = payload["feature_count"]

    def scores(self, req, hole, opponent, equity):
        values = state_features(req, hole, opponent, equity)
        if len(values) != self.feature_count:
            raise ValueError("model feature count does not match current feature extractor")
        x = [1.0] + values
        return {name: sum(w * v for w, v in zip(self.weights[name], x))
                for name in ACTIONS if name in self.weights}

    def choose(self, req, hole, opponent, equity):
        legal = set(req["legal_actions"])
        valid = [a for a in ACTIONS if (a in legal or (a.startswith("raise_") and "raise" in legal))]
        if not valid:
            return None
        best = max(valid, key=self.scores(req, hole, opponent, equity).get)
        if not best.startswith("raise_"):
            return {"type": "action", "action": best}
        lo, hi = req["min_raise"], req["max_raise"]
        fraction = {"raise_small": .33, "raise_medium": .62, "raise_large": .92}[best]
        return {"type": "action", "action": "raise", "amount": int(lo + (hi - lo) * fraction)}

