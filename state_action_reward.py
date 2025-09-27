# State - Action - Reward for Q-learning

import pandas as pd
import numpy as np

# -------------------------
# States
# -------------------------

# --- State feature rules ---
state_rules = {
    "wasabi_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "wasabi"),
    "egg_nigiri_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "nigiri" and c.subtype == 1),
    "salmon_nigiri_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "nigiri" and c.subtype == 2),
    "squid_nigiri_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "nigiri" and c.subtype == 3),
    "tempura_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "tempura"),
    "sashimi_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "sashimi"),
    "dumpling_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "dumpling"),
    "pudding_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "pudding"),
    "maki_1_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "maki" and c.subtype == 1),
    "maki_2_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "maki" and c.subtype == 2),
    "maki_3_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "maki" and c.subtype == 3),
    "chopsticks_in_hand": lambda hand, table: sum(1 for c in hand if c.type == "chopsticks"),
    "free_wasabi_on_table": lambda hand, table: count_free_wasabi(table),
    "free_tempura_on_table": lambda hand, table: sum(1 for c in table if c.type == "tempura") % 2,
    "free_sashimi_on_table": lambda hand, table: sum(1 for c in table if c.type == "sashimi") % 3,
    "dumpling_on_table": lambda hand, table: sum(1 for c in table if c.type == "dumpling"),
    "pudding_on_table": lambda hand, table: sum(1 for c in table if c.type == "pudding"),
    "maki_points_on_table": lambda hand, table: sum(c.subtype for c in table if c.type == "maki"),
    "cards_in_hand": lambda hand, table: len(hand),
}

def state_vars():
    """Return the list of all state feature keys."""
    return list(state_rules.keys())

def build_state_dict(cards_in_hand, cards_on_table):
    """Compute state features using state_rules."""
    return {key: rule(cards_in_hand, cards_on_table) for key, rule in state_rules.items()}

# -------------------------
# Actions
# -------------------------

# Map each action to a lambda that evaluates availability given a state_dict
action_rules = {
    "play_wasabi": lambda s: s["wasabi_in_hand"] > 0,
    "play_highest_nigiri": lambda s: (
        s["egg_nigiri_in_hand"] +
        s["salmon_nigiri_in_hand"] +
        s["squid_nigiri_in_hand"]
    ) > 0,
    "play_tempura": lambda s: s["tempura_in_hand"] > 0,
    "play_sashimi": lambda s: s["sashimi_in_hand"] > 0,
    "play_dumpling": lambda s: s["dumpling_in_hand"] > 0,
    "play_pudding": lambda s: s["pudding_in_hand"] > 0,
    "play_highest_maki": lambda s: (
        s["maki_1_in_hand"] +
        s["maki_2_in_hand"] +
        s["maki_3_in_hand"]
    ) > 0,
    "play_chopsticks": lambda s: s["chopsticks_in_hand"] > 0,
}

def actions():
    """
    Define the set of possible actions (keys only).
    """
    return list(action_rules.keys())

def build_actions_dict(state_dict):
    """Given a state dictionary, return a dictionary of possible actions (True/False)."""
    return {action: rule(state_dict) for action, rule in action_rules.items()}

# -------------------------
# Rewards
# -------------------------

def rewards(states, actions):
    """
    Initialize a reward table with zeros.
    Reward policy is updated during gameplay.
    """
    R = pd.DataFrame(
        data=np.zeros((len(states), len(actions))),
        columns=actions,
        index=states
    )
    return R

# -------------------------
# Miscellaneous
# -------------------------

def count_free_wasabi(cards_on_table):
    """Return the number of wasabi cards on table that have not yet been paired with a nigiri."""

    free_wasabi = 0
    pending_wasabi = 0

    for card in cards_on_table:
        if card.type == "wasabi":
            # new wasabi waiting for a nigiri
            pending_wasabi += 1
        elif card.type == "nigiri" and pending_wasabi > 0:
            # first pending wasabi gets paired with this nigiri
            pending_wasabi -= 1
        # else: ignore (either nigiri without wasabi, or other card)

    free_wasabi = pending_wasabi
    return free_wasabi
