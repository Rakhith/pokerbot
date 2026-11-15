#!/usr/bin/env python3
"""Protocol-compatible ML bot.

Run it exactly as the regular bot: ``python3 ml_bot.py``.  It is deliberately
safe by default: without a *counterfactual* trained model it makes precisely
the same decisions as ``bot.py``.  A bad or missing model must never turn a
solid baseline into a losing bot.
"""

import json
import os
import sys

from bot import PokerBot
from features import action_bucket
from policy import ActionValuePolicy


class MLPokerBot(PokerBot):
    """Heuristic policy with a conservatively gated learned action-value head."""

    def __init__(self):
        super().__init__()
        self.ml_policy = None
        model_path = os.environ.get("ML_MODEL_PATH", "models/action_value.json")
        try:
            candidate = ActionValuePolicy(model_path)
            # Models trained on logged, single-action trajectories have invalid
            # action comparisons and are intentionally not allowed to override.
            if candidate.metadata.get("label_source") == "counterfactual_rollout":
                self.ml_policy = candidate
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    def choose_action(self, req):
        baseline = super().choose_action(req)
        if self.ml_policy is None or req["street"] == "preflop":
            return baseline

        # The learned head may only improve post-flop choices when it has a
        # meaningful value margin.  Calling/folding remains the baseline unless
        # rollout labels show a substantial advantage; this prevents noisy
        # model scores from producing spewy raises.
        equity = self.current_equity(req)
        scores = self.ml_policy.scores(req, self.hole, self.opp_model, equity)
        proposed = self.ml_policy.choose(req, self.hole, self.opp_model, equity)
        if proposed is None:
            return baseline
        baseline_label = action_bucket(baseline, req)
        proposed_label = action_bucket(proposed, req)
        margin = scores.get(proposed_label, float("-inf")) - scores.get(baseline_label, 0.0)
        # A margin in normalised chip-EV units.  This needs validation against
        # the baseline before it is lowered.
        if margin < 0.08:
            return baseline
        if proposed["action"] == "raise" and equity < 0.52:
            return baseline
        if baseline["action"] == "call" and proposed["action"] == "fold" and equity >= 0.42:
            return baseline
        return proposed


def main():
    bot = MLPokerBot()
    for raw_line in sys.stdin:
        try:
            msg = json.loads(raw_line)
        except (TypeError, ValueError):
            continue
        kind = msg.get("type")
        if kind == "action_request":
            bot.observe(msg)
            print(json.dumps(bot.choose_action(msg), separators=(",", ":")), flush=True)
        elif kind == "discard_request":
            print(json.dumps(bot.choose_discard(msg), separators=(",", ":")), flush=True)
        elif kind == "match_end":
            break
        else:
            bot.observe(msg)


if __name__ == "__main__":
    main()
