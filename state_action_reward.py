# State - Action - Reward for Q-learning

import pandas as pd
import numpy as np
import itertools

def states():
    """
    Build all possible state combinations for Sushi Go based on card counts.
    Each element in the state vector is an integer in the specified range.
    """

    # Define ranges for each component of the state vector
    state_components = [
        range(3),  # wasabi_in_hand (0-2)
        range(3),  # egg_nigiri_in_hand (0-2)
        range(3),  # salmon_nigiri_in_hand (0-2)
        range(3),  # squid_nigiri_in_hand (0-2)
        range(3),  # tempura_in_hand (0-2)
        range(3),  # sashimi_in_hand (0-2)
        range(3),  # dumpling_in_hand (0-2)
        range(3),  # pudding_in_hand (0-2)
        range(3),  # maki_1_in_hand (0-2)
        range(3),  # maki_2_in_hand (0-2)
        range(3),  # maki_3_in_hand (0-2)
        range(3),  # chopsticks_in_hand (0-2)
        range(3),  # free_wasabi_on_table (0-2)
        range(2),  # free_tempura_on_table (0-1)
        range(3),  # free_sashimi_on_table (0-2)
        range(6),  # dumpling_on_table (0-5)
        range(3),  # pudding_on_table (0-2)
        range(7),  # maki_points_on_table (0-6)
        range(11), # cards_in_hand (0-10)
    ]

    # # Temporary small state vector for prototyping
    # state_components = [
    #     range(3),  # wasabi_in_hand (0-2)
    #     range(3),  # egg_nigiri_in_hand (0-2)
    # ]

    # Generate Cartesian product of all state components
    states_all = list(itertools.product(*state_components))

    return states_all


def actions():
    """
    Define the set of possible actions.
    """
    actions_all = [
        "play_wasabi",
        "play_highest_nigiri",
        "play_tempura",
        "play_sashimi",
        "play_dumpling",
        "play_pudding",
        "play_highest_maki",
        "play_chopsticks"
    ]
    return actions_all


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
