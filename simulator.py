"""Card-level rollout evaluator for generating offline action-value labels.

This intentionally models only showdown equity.  Add betting and stack rules
here when the tournament engine's full rules are available, then use the same
``rollout_equity`` interface to create terminal rewards.
"""

import random

from bot import DECK, best_hand


def rollout_equity(hero, board, blocked=(), samples=500, rng=None):
    """Estimate heads-up showdown equity for a retained two-card hand."""
    rng = rng or random.Random()
    known = set(hero) | set(board) | set(blocked)
    unseen = [card for card in DECK if card not in known]
    missing = 5 - len(board)
    wins = ties = 0
    for _ in range(samples):
        draw = rng.sample(unseen, 2 + missing)
        runout = list(board) + draw[2:]
        ours, theirs = best_hand(hero, runout), best_hand(draw[:2], runout)
        wins += ours > theirs
        ties += ours == theirs
    return (wins + .5 * ties) / samples

