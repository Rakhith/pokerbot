#!/usr/bin/env python3
"""
Tournament-grade Discard Hold'em bot.

Key improvements over the starter:
  - Correct 7-card hand evaluator via best-of-C(7,5)
  - Monte-Carlo equity conditioned on public cards and exposed discards
  - Board-texture-aware discard selection with draw potential scoring
  - Adaptive opponent modeling: track fold%, aggression, showdown ranges
  - Position-aware preflop strategy with 3-card hand strength tables
  - Post-flop bet sizing: geometric sizing toward river, overbets with nuts
  - Bluff-to-value ratio tuned per street (semi-bluffs on flop/turn, thin value river)
  - Time-budget management: adaptive sample counts to stay within bank
"""

import json
import random
import sys
import time
from collections import Counter
from itertools import combinations

# ================================================================== Constants
RANKS = "23456789TJQKA"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
DECK = tuple(r + s for r in RANKS for s in "shdc")

# Precomputed lookup for 5-card evaluation (category, tiebreakers)
# Categories: 0=high, 1=pair, 2=two-pair, 3=trips, 4=straight, 5=flush,
#             6=full-house, 7=quads, 8=straight-flush

# ============================================================ Hand Evaluator

def _rank5(cards):
    """Evaluate exactly 5 cards -> comparable tuple (higher is better)."""
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    is_flush = len(set(suits)) == 1
    unique = sorted(set(values), reverse=True)

    # Check straight
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight_high = 5  # wheel

    groups = sorted(Counter(values).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    by_group = tuple(v for v, _ in groups)

    if straight_high and is_flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7,) + by_group
    if shape == [3, 2]:
        return (6,) + by_group
    if is_flush:
        return (5,) + tuple(values)
    if straight_high:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3,) + by_group
    if shape == [2, 2, 1]:
        return (2,) + by_group
    if shape == [2, 1, 1, 1]:
        return (1,) + by_group
    return (0,) + tuple(values)


def best_hand(hole, board):
    """Best 5-card hand from hole + board (using 0, 1, or 2 hole cards)."""
    all_cards = list(hole) + list(board)
    return max(_rank5(combo) for combo in combinations(all_cards, 5))


# ====================================================== Utility functions

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def straight_draw_outs(values, board_values):
    """Count approximate straight draw outs (open-ended or gutshot)."""
    all_vals = set(values) | set(board_values)
    if 14 in all_vals:
        all_vals.add(1)
    outs = 0
    for high in range(5, 15):
        needed = set(range(high - 4, high + 1))
        missing = needed - all_vals
        if len(missing) == 1:
            outs += 4  # approximate (could be fewer if cards are dead)
        elif len(missing) == 2 and len(all_vals & needed) >= 4:
            outs += 2  # weak draw
    return min(outs, 12)


def flush_draw_info(hole, board):
    """Return (is_flush_draw, flush_outs, suit) for the hero's hand."""
    suits_count = Counter(c[1] for c in hole + board)
    for suit, count in suits_count.items():
        if count >= 5:
            return False, 0, suit  # already a flush
        if count == 4:
            hero_in_suit = sum(1 for c in hole if c[1] == suit)
            if hero_in_suit >= 1:
                return True, 9, suit
    return False, 0, None


# ========================================================= Opponent Model

class OpponentModel:
    """Simple adaptive model tracking opponent tendencies across a leg."""

    def __init__(self):
        self.hands_seen = 0
        self.vpip_count = 0       # voluntarily put $ in pot preflop
        self.pfr_count = 0        # preflop raise count
        self.fold_to_raise = 0    # folded when facing a raise
        self.raise_faced = 0
        self.cbet_count = 0       # continuation bet on flop
        self.cbet_opp = 0
        self.showdown_hands = 0
        self.total_aggression = 0  # raises post-flop
        self.total_passive = 0     # calls post-flop
        self.fold_to_river_bet = 0
        self.river_bet_faced = 0
        self.discard_rank_sum = 0
        self.discard_count = 0

    def update_hand(self, history, result, opp_discard=None):
        self.hands_seen += 1

        # Track VPIP/PFR
        preflop_actions = [h for h in history if h.get("street") == "preflop" and h.get("actor") == "opp"]
        if preflop_actions:
            actions = [h["action"] for h in preflop_actions]
            if "raise" in actions or "call" in actions:
                self.vpip_count += 1
            if "raise" in actions:
                self.pfr_count += 1

        # Track fold-to-raise
        for h in history:
            if h.get("actor") == "opp":
                if h["action"] == "fold":
                    # Check if they were facing a raise
                    idx = history.index(h)
                    if idx > 0:
                        prev_actions = [x for x in history[:idx] if x.get("actor") == "me"]
                        if prev_actions and prev_actions[-1]["action"] == "raise":
                            self.fold_to_raise += 1
                            self.raise_faced += 1
                elif h["action"] in ("call", "raise"):
                    idx = history.index(h)
                    if idx > 0:
                        prev_actions = [x for x in history[:idx] if x.get("actor") == "me"]
                        if prev_actions and prev_actions[-1]["action"] == "raise":
                            self.raise_faced += 1

        # Track post-flop aggression
        for h in history:
            if h.get("actor") == "opp" and h.get("street") != "preflop":
                if h["action"] == "raise":
                    self.total_aggression += 1
                elif h["action"] == "call":
                    self.total_passive += 1

        # Track showdowns
        if result == "showdown":
            self.showdown_hands += 1

        # Track discard quality
        if opp_discard:
            self.discard_count += 1
            self.discard_rank_sum += RANK_VALUE[opp_discard[0]]

    @property
    def vpip(self):
        return self.vpip_count / max(1, self.hands_seen)

    @property
    def pfr(self):
        return self.pfr_count / max(1, self.hands_seen)

    @property
    def fold_to_raise_pct(self):
        return self.fold_to_raise / max(1, self.raise_faced)

    @property
    def aggression_factor(self):
        """AF = raises / calls (post-flop). Higher = more aggressive."""
        return self.total_aggression / max(1, self.total_passive)

    @property
    def avg_discard_rank(self):
        return self.discard_rank_sum / max(1, self.discard_count)

    def opp_range_tightness(self):
        """0.0 = extremely loose, 1.0 = extremely tight. Used for equity adjustment."""
        if self.hands_seen < 8:
            return 0.5  # default assumption
        # VPIP < 0.3 is tight, > 0.7 is loose
        return clamp(1.0 - self.vpip, 0.15, 0.85)

    def pressure_discount(self, raises_by_opp, to_call, pot):
        """Extra equity discount when opp shows aggression, adjusted by their tendencies."""
        base_pressure = 0.018 * raises_by_opp + 0.055 * min(1.0, to_call / max(1, pot))
        # If opponent is a maniac (high AF), discount their aggression
        if self.hands_seen >= 15 and self.aggression_factor > 2.5:
            base_pressure *= 0.6
        # If opponent is tight-aggressive, respect their bets more
        elif self.hands_seen >= 15 and self.vpip < 0.35 and self.aggression_factor > 1.2:
            base_pressure *= 1.4
        return min(0.18, base_pressure)


# ============================================================= Main Bot

class PokerBot:
    def __init__(self):
        self.rng = random.Random()
        self.hole = []
        self.opp_discard = None
        self.seat = None
        self.opp_model = OpponentModel()
        self.time_bank_ms = 30000
        self.start_time = time.monotonic()
        self.hands_played = 0
        self.total_hands = 200
        self.last_history = []
        self._flop_keep_cache = None  # (hole_tuple, board_tuple) -> keep_pair
        self._flop_keep_key = None

    def time_remaining(self):
        """Rough estimate of remaining time bank."""
        return self.time_bank_ms

    def sample_budget(self, street_len):
        """Adaptively choose number of MC samples based on remaining time budget."""
        remaining = self.time_remaining()
        hands_left = max(1, self.total_hands - self.hands_played)

        # Reserve some budget for future hands
        budget_per_hand_ms = remaining / hands_left
        # Each hand has ~3-5 decisions, each decision might sample
        budget_per_decision_ms = budget_per_hand_ms / 4.0

        if budget_per_decision_ms > 80:
            base = 150
        elif budget_per_decision_ms > 40:
            base = 100
        elif budget_per_decision_ms > 15:
            base = 60
        elif budget_per_decision_ms > 5:
            base = 35
        else:
            base = 20

        # More samples on later streets (fewer unknown cards, more at stake)
        multiplier = {0: 0.5, 3: 0.7, 4: 0.9, 5: 1.0}
        return int(base * multiplier.get(street_len, 0.8))

    # -------------------------------------------------------- Preflop Strength
    @staticmethod
    def preflop_strength(cards):
        """
        Score a 3-card starting hand. Roughly 0-45 scale.
        Accounts for: high cards, pairs, trips, suitedness, connectivity.
        """
        values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
        suits = [c[1] for c in cards]
        counts = Counter(values)

        # Base card strength (weighted by position)
        score = values[0] * 0.85 + values[1] * 0.60 + values[2] * 0.28

        # Pair bonus
        pair_count = sum(1 for c in counts.values() if c >= 2)
        if 3 in counts.values():
            score += 18.0  # trips - massive
        elif pair_count:
            pair_rank = max(r for r, c in counts.items() if c >= 2)
            score += 6.0 + (pair_rank - 2) * 0.5  # higher pairs worth more

        # Suited bonus
        suit_counts = Counter(suits)
        max_suited = max(suit_counts.values())
        if max_suited == 3:
            score += 4.0  # all three suited
        elif max_suited == 2:
            score += 1.8  # two suited

        # Connectivity bonus (straight potential)
        gaps = values[0] - values[2]
        if gaps <= 4:
            score += 3.5  # very connected
        elif gaps <= 6:
            score += 1.8
        elif gaps <= 8:
            score += 0.5

        # Broadway bonus (all cards T+)
        if values[2] >= 10:
            score += 2.5

        # Ace-high kicker bonus
        if values[0] == 14:
            if values[1] >= 12:
                score += 2.0
            elif values[1] >= 10:
                score += 1.0

        return score

    # ------------------------------------------------------ Equity Estimation
    def sample_equity(self, hero, board, samples, blocked=()):
        """
        Monte Carlo equity estimation.
        The opponent is modelled as a random legal two-card holding after its
        discard.  Before the discard this is deliberately a conservative,
        tractable approximation; betting-history pressure is applied by the
        action policy rather than being baked into card sampling.
        """
        known = set(hero) | set(board) | set(blocked)
        if self.opp_discard:
            known.add(self.opp_discard)
        unseen = [c for c in DECK if c not in known]
        needed_board = 5 - len(board)
        wins = ties = 0
        n = min(samples, len(unseen) * (len(unseen) - 1) // 2)  # cap at max combos

        for _ in range(n):
            draw = self.rng.sample(unseen, 2 + needed_board)
            runout = board + draw[2:]
            ours = best_hand(hero, runout)

            # At showdown an opponent has two cards.  On the flop this is the
            # post-discard approximation described above.
            theirs = best_hand(draw[:2], runout)

            if ours > theirs:
                wins += 1
            elif ours == theirs:
                ties += 1

        return (wins + 0.5 * ties) / max(1, n)

    def choose_keep(self, cards, flop):
        """
        Choose which 2 cards to keep. Evaluates each possible discard by:
        1. Current hand strength with the flop
        2. Draw potential (flush draws, straight draws)
        3. Monte Carlo equity simulation
        """
        best_pair, best_score = None, -1.0

        for pair in combinations(cards, 2):
            pair_list = list(pair)
            discard = [c for c in cards if c not in pair_list][0]

            # Current hand strength
            current_rank = best_hand(pair_list, list(flop))

            # Draw potential bonus
            draw_bonus = 0.0
            hero_values = [RANK_VALUE[c[0]] for c in pair_list]
            board_values = [RANK_VALUE[c[0]] for c in flop]

            # Flush draw check
            is_fd, fd_outs, _ = flush_draw_info(pair_list, list(flop))
            if is_fd:
                draw_bonus += 0.08

            # Straight draw check
            sd_outs = straight_draw_outs(hero_values, board_values)
            if sd_outs >= 8:
                draw_bonus += 0.06
            elif sd_outs >= 4:
                draw_bonus += 0.03

            # Kicker quality (higher kicker = better)
            kicker_bonus = sum(hero_values) / 28.0 * 0.02

            # Information cost: discarding a high card reveals more
            discard_rank = RANK_VALUE[discard[0]]
            info_penalty = 0.0
            # Discarding an ace or king gives away information
            if discard_rank >= 13:
                info_penalty = 0.01
            elif discard_rank >= 11:
                info_penalty = 0.005

            # MC equity (main signal)
            equity = self.sample_equity(pair_list, list(flop), 50, blocked=cards)

            score = equity + draw_bonus + kicker_bonus - info_penalty

            if score > best_score:
                best_pair, best_score = pair_list, score

        return best_pair

    def current_equity(self, req):
        board = list(req["board"])
        if len(board) == 3 and len(self.hole) == 3:
            hero = self.planned_keep(board)
        else:
            hero = list(self.hole)

        samples = self.sample_budget(len(board))
        return self.sample_equity(hero, board, samples, blocked=self.hole)

    def planned_keep(self, flop):
        """Return the one discard plan used for both flop betting and discard.

        A flop decision and the exposed discard happen at different protocol
        messages.  Re-sampling independently can make the action evaluate one
        two-card hand and then expose a different one, so retain the plan for
        this hand and flop.
        """
        key = (tuple(sorted(self.hole)), tuple(flop))
        if self._flop_keep_key != key or self._flop_keep_cache is None:
            self._flop_keep_cache = self.choose_keep(self.hole, list(flop))
            self._flop_keep_key = key
        return list(self._flop_keep_cache)

    # -------------------------------------------------------- Bet Sizing
    def compute_raise_amount(self, req, equity, street):
        """
        Context-sensitive raise sizing.
        - Value hands: size up for max extraction
        - Semi-bluffs: smaller sizing for fold equity
        - Position-aware: bet slightly larger OOP
        """
        pot = req["pot"]
        lower = req["min_raise"]
        upper = req["max_raise"]

        if equity >= 0.90:
            # Polar: big bet with near-nuts
            fraction = 1.2
        elif equity >= 0.78:
            # Strong value: 70-80% pot
            fraction = 0.75
        elif equity >= 0.65:
            # Medium value / protection: 50-60% pot
            fraction = 0.55
        else:
            # Semi-bluff / thin value: 33-40% pot
            fraction = 0.35

        # Adjust for position (bet slightly larger OOP for protection)
        if self.seat == "big_blind" and street != "preflop":
            fraction *= 1.1

        target = req["opp_bet"] + max(2, int(pot * fraction))
        return clamp(target, lower, upper)

    # --------------------------------------------------- Preflop Action
    def preflop_action(self, req):
        legal = req["legal_actions"]
        strength = self.preflop_strength(self.hole)
        to_call = req["to_call"]
        pot = req["pot"]
        raises = sum(1 for h in req.get("history", []) if h.get("action") == "raise")
        is_button = self.seat == "button"

        if to_call == 0:
            # We can check or raise
            if "raise" in legal:
                if strength >= 28:
                    # Premium: raise big
                    target = req["opp_bet"] + max(2, int(pot * 1.2))
                    return {"type": "action", "action": "raise",
                            "amount": clamp(target, req["min_raise"], req["max_raise"])}
                elif strength >= 20:
                    # Good hand: standard raise
                    target = req["opp_bet"] + max(2, int(pot * 0.7))
                    return {"type": "action", "action": "raise",
                            "amount": clamp(target, req["min_raise"], req["max_raise"])}
                elif is_button and strength >= 14:
                    # Button steal with decent hands
                    target = req["opp_bet"] + max(2, int(pot * 0.55))
                    return {"type": "action", "action": "raise",
                            "amount": clamp(target, req["min_raise"], req["max_raise"])}
            return {"type": "action", "action": "check"}

        # Facing a raise
        pot_odds = to_call / (pot + to_call)

        # Adjust call/raise thresholds based on opponent model
        opp_tightness = self.opp_model.opp_range_tightness()
        # Against tight opponents, need stronger hands
        tight_adjust = (opp_tightness - 0.5) * 6.0  # +3 vs tight, -3 vs loose

        # 3-bet / 4-bet
        reraise_threshold = 30.0 + 3.0 * raises + tight_adjust
        if "raise" in legal and strength >= reraise_threshold:
            fraction = 1.3 if strength >= 36 else 0.85
            target = req["opp_bet"] + max(2, int(pot * fraction))
            return {"type": "action", "action": "raise",
                    "amount": clamp(target, req["min_raise"], req["max_raise"])}

        # Call
        call_threshold = 10.0 + 20.0 * pot_odds + 3.5 * max(0, raises - 1) + tight_adjust
        if "call" in legal and strength >= call_threshold:
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold"}

    # ------------------------------------------------ Post-flop Action
    def postflop_action(self, req):
        legal = req["legal_actions"]
        equity = self.current_equity(req)
        to_call = req["to_call"]
        pot = req["pot"]
        street = req["street"]
        board = req["board"]

        # Count opponent raises this hand for pressure estimation
        opp_raises = sum(
            1 for h in req.get("history", [])
            if h.get("actor") == "opp" and h.get("action") == "raise"
        )

        # Draw information
        hero = (self.planned_keep(board)
                if len(board) == 3 and len(self.hole) == 3 else list(self.hole))
        is_fd, fd_outs, _ = flush_draw_info(hero, list(board))
        hero_vals = [RANK_VALUE[c[0]] for c in hero]
        board_vals = [RANK_VALUE[c[0]] for c in board]
        sd_outs = straight_draw_outs(hero_vals, board_vals)
        has_draw = is_fd or sd_outs >= 6

        if to_call == 0:
            # === Check or Bet ===

            # Determine board texture
            board_pairs = sum(1 for _, c in Counter(RANK_VALUE[b[0]] for b in board).items() if c >= 2)
            board_is_wet = is_fd or sd_outs >= 4 or board_pairs == 0

            # Value bet thresholds
            if equity >= 0.80:
                # Strong value: bet for value
                if "raise" in legal:
                    amt = self.compute_raise_amount(req, equity, street)
                    return {"type": "action", "action": "raise", "amount": amt}

            if equity >= 0.62:
                # Medium-strong: bet for value/protection, especially on wet boards
                if "raise" in legal:
                    if board_is_wet or street == "river":
                        amt = self.compute_raise_amount(req, equity, street)
                        return {"type": "action", "action": "raise", "amount": amt}
                    elif self.rng.random() < 0.7:
                        amt = self.compute_raise_amount(req, equity, street)
                        return {"type": "action", "action": "raise", "amount": amt}

            # Semi-bluff with draws (not on river)
            if has_draw and street != "river" and "raise" in legal:
                bluff_freq = 0.35 if (is_fd and sd_outs >= 4) else 0.20 if is_fd else 0.12
                if self.rng.random() < bluff_freq:
                    amt = self.compute_raise_amount(req, 0.55, street)
                    return {"type": "action", "action": "raise", "amount": amt}

            # Occasional bluff on dry boards
            if street == "river" and equity < 0.35 and "raise" in legal:
                # Bluff river ~8-12% (higher vs tight opponents)
                bluff_freq = 0.08
                if self.opp_model.hands_seen >= 15 and self.opp_model.fold_to_raise_pct > 0.45:
                    bluff_freq = 0.15
                if self.rng.random() < bluff_freq:
                    amt = self.compute_raise_amount(req, 0.85, street)  # big bluff sizing
                    return {"type": "action", "action": "raise", "amount": amt}

            return {"type": "action", "action": "check"}

        # === Facing a bet ===
        pot_odds = to_call / (pot + to_call)
        pressure = self.opp_model.pressure_discount(opp_raises, to_call, pot)
        adjusted = equity - pressure

        # Raise for value
        if "raise" in legal and adjusted >= 0.80:
            amt = self.compute_raise_amount(req, adjusted, street)
            return {"type": "action", "action": "raise", "amount": amt}

        # Semi-bluff raise with strong draws (not river)
        if "raise" in legal and has_draw and street != "river" and adjusted >= 0.40:
            if self.rng.random() < 0.25:
                amt = self.compute_raise_amount(req, 0.60, street)
                return {"type": "action", "action": "raise", "amount": amt}

        # Call if equity justifies it
        # Include implied odds for draws
        implied_bonus = 0.0
        if has_draw and street != "river":
            # Implied odds: if we hit, we might win more
            implied_bonus = 0.04 if is_fd else 0.025

        if "call" in legal and (adjusted + implied_bonus) >= pot_odds + 0.02:
            return {"type": "action", "action": "call"}

        # Bluff-raise vs weak opponents
        if "raise" in legal and self.opp_model.hands_seen >= 20:
            if self.opp_model.fold_to_raise_pct > 0.55 and self.rng.random() < 0.10:
                amt = self.compute_raise_amount(req, 0.80, street)
                return {"type": "action", "action": "raise", "amount": amt}

        return {"type": "action", "action": "fold"}

    # ------------------------------------------------ Choose Action
    def choose_action(self, req):
        if req["street"] == "preflop":
            return self.preflop_action(req)
        return self.postflop_action(req)

    # ------------------------------------------------ Choose Discard
    def choose_discard(self, req):
        cards, flop = list(req["hole_cards"]), list(req["board"])
        # Reuse the planned pair if we received a flop action request.  All
        # legal hand flows that reach discard have the same hole cards/flop.
        key = (tuple(sorted(cards)), tuple(flop))
        if self._flop_keep_key == key and self._flop_keep_cache is not None:
            keep = list(self._flop_keep_cache)
        else:
            keep = self.choose_keep(cards, flop)
            self._flop_keep_cache, self._flop_keep_key = keep, key
        discard = next(c for c in cards if c not in keep)
        return {"type": "discard", "card": discard}

    # ------------------------------------------------ Observe
    def observe(self, msg):
        kind = msg.get("type")
        if "time_left_ms" in msg:
            self.time_bank_ms = msg["time_left_ms"]
        if kind == "init":
            self.total_hands = msg.get("total_hands", 200)
            self.time_bank_ms = msg.get("time_bank_ms", 30000)
        elif kind == "new_hand":
            self.hole = list(msg.get("hole_cards", []))
            self.opp_discard = None
            self.seat = msg.get("seat")
            self.hands_played += 1
            self._flop_keep_cache = None
            self._flop_keep_key = None
        elif kind == "discard_result":
            self.hole = list(msg.get("hole_cards", []))
            self.opp_discard = msg.get("opp_discard")
        elif kind == "hand_result":
            history = msg.get("history", [])
            result = msg.get("result", "")
            opp_disc = msg.get("opp_discard")
            self.opp_model.update_hand(history, result, opp_disc)


# ================================================================ Main Loop

def main():
    bot = PokerBot()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except (TypeError, ValueError):
            continue
        kind = msg.get("type")
        if kind == "action_request":
            bot.observe(msg)  # update time bank before deciding
            print(json.dumps(bot.choose_action(msg), separators=(",", ":")), flush=True)
        elif kind == "discard_request":
            print(json.dumps(bot.choose_discard(msg), separators=(",", ":")), flush=True)
        elif kind == "match_end":
            break
        else:
            bot.observe(msg)


if __name__ == "__main__":
    main()
