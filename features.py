"""Feature extraction shared by data collection, training, and live play.

The representation is deliberately small and deterministic: it is cheap enough
to compute during the bot's time bank and does not require a ML dependency.
"""

from collections import Counter
from typing import Dict, Iterable, List, Sequence

from bot import RANK_VALUE, best_hand, flush_draw_info, straight_draw_outs


ACTIONS = ("fold", "check", "call", "raise_small", "raise_medium", "raise_large")


def _one_hot(index: int, width: int) -> List[float]:
    return [1.0 if i == index else 0.0 for i in range(width)]


def hand_features(hole: Sequence[str], board: Sequence[str]) -> List[float]:
    """Return card-only features for a two- or three-card private holding."""
    values = sorted((RANK_VALUE[c[0]] for c in hole), reverse=True)
    suits = [c[1] for c in hole]
    counts = Counter(values)
    padded = values + [0] * (3 - len(values))
    suit_counts = Counter(suits)
    hero_values = values
    board_values = [RANK_VALUE[c[0]] for c in board]
    fd, fd_outs, _ = flush_draw_info(list(hole), list(board))
    sd_outs = straight_draw_outs(hero_values, board_values)
    category = best_hand(list(hole), list(board))[0] if len(hole) + len(board) >= 5 else 0
    return [
        *(v / 14.0 for v in padded),
        float(max(counts.values()) >= 2),
        float(max(counts.values()) == 3),
        float(max(suit_counts.values()) >= 2),
        float(max(suit_counts.values()) == 3),
        float(max(values) - min(values) <= 4) if len(values) > 1 else 0.0,
        *_one_hot(category, 9),
        float(fd), fd_outs / 9.0, min(sd_outs, 12) / 12.0,
    ]


def state_features(req: Dict, hole: Sequence[str], opponent: object = None,
                   equity: float = 0.5) -> List[float]:
    """Convert a protocol action request into a fixed-size numeric vector."""
    board = list(req.get("board", []))
    pot = max(1, float(req.get("pot", 0)))
    to_call = float(req.get("to_call", 0))
    legal = set(req.get("legal_actions", []))
    street = req.get("street", "preflop")
    street_index = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}.get(street, 0)
    board_ranks = [RANK_VALUE[c[0]] for c in board]
    board_counts = Counter(board_ranks)
    vpip = getattr(opponent, "vpip", 0.5)
    pfr = getattr(opponent, "pfr", 0.5)
    af = min(getattr(opponent, "aggression_factor", 1.0), 5.0) / 5.0
    fold_raise = getattr(opponent, "fold_to_raise_pct", 0.5)
    discard_rank = getattr(opponent, "avg_discard_rank", 8.0) / 14.0
    return hand_features(hole, board) + [
        equity, to_call / pot, min(pot, 200) / 200.0,
        *_one_hot(street_index, 4),
        len(board) / 5.0,
        float(any(count >= 2 for count in board_counts.values())),
        float(len(set(c[1] for c in board)) <= 2) if board else 0.0,
        *(float(a in legal) for a in ("fold", "check", "call", "raise")),
        vpip, pfr, af, fold_raise, discard_rank,
    ]


def action_bucket(action: Dict, req: Dict) -> str:
    """Map protocol actions to stable labels suitable for a small classifier."""
    kind = action.get("action", "check")
    if kind != "raise":
        return kind
    minimum = float(req.get("min_raise", 0))
    maximum = max(minimum + 1.0, float(req.get("max_raise", minimum + 1)))
    ratio = (float(action.get("amount", minimum)) - minimum) / (maximum - minimum)
    return "raise_small" if ratio < 1 / 3 else "raise_medium" if ratio < 2 / 3 else "raise_large"

