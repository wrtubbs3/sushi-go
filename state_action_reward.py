# State - Action - Reward for Q-learning

import pandas as pd
import numpy as np

# =======================================================================================
# Machine Learning - Reinforcement Learning
# =======================================================================================
# 
# Monte Carlo Agent:
#
# q(s,a) = q(s,a) + alpha*(R - q(s,a))
#
# (Brunton version)
# q(s,a) = q(s,a) + (1/n)*(R - q(s,a))
# 
# where s is the state vector, a is the action vector, R is the reward for this episode,
# and alpha is the step size parameter. Using the epsilon-greedy algorithm, given a state
# the agent chooses an action as follows:
#   -- With epsilon probability: random action
#   -- With (1-epsilon) probability: action with maximum q value
#
#
# Q-Learning Agent:
# 
# q(s,a) = q(s,a) + alpha*(r + q(s_hat, a_hat) - q(s,a))
#
# where s is the state vector, a is the action vector, r is the reward for the next step,
# and alpha is the step size parameter. s_hat and a_hat refer to the vectors for the next
# # step. Using the epsilon-greedy algorithm, given a state
# the agent chooses an action as follows:
#   -- With epsilon probability: random action
#   -- With (1-epsilon) probability: action with maximum q value
#
#

def states():
    
    # State vector:
    # {counting cards off}: [my cards in hand, my cards on table, total qty. of cards in hand]
    #
    # Other ideas to include: score prior to hand, opponent's cards on table
    # 
    # Detailed state vector (# of possible states) (states)
    # [
    # wasabi_in_hand (3) (0 thru 2),
    # egg_nigiri_in_hand (3) (0 thru 2),
    # salmon_nigiri_in_hand (3) (0 thru 2),
    # squid_nigiri_in_hand (3) (0 thru 2),
    # tempura_in_hand (3) (0 thru 2),
    # sashimi_in_hand (3) (0 thru 2),
    # dumpling_in_hand (3) (0 thru 2),
    # pudding_in_hand (3) (0 thru 2),
    # maki_1_in_hand (3) (0 thru 2),
    # maki_2_in_hand (3) (0 thru 2),
    # maki_3_in_hand (3) (0 thru 2),
    # chopsticks_in_hand (3) (0 thru 2),
    # free_wasabi_on_table (3) (0 thru 2),
    # free_tempura_on_table (2) (0 thru 1),
    # free_sashimi_on_table (3) (0 thru 2),
    # dumpling_on_table (6) (0 thru 5),
    # pudding_on_table (3) (0 thru 2),
    # maki_points_on_table (7) (0 thru 6),
    # cards_in_hand (11) (0 thru 10),
    # ]
    #
    # Total possible states = 1.3 x 1E10
    #
    # {counting cards on}: same states as above, plus cards on table for other players, plus cards played in previous hands



    # Normal cards
    norm_cards = {"RED":2,"GRE":2,"BLU":2,"YEL":2}
    spec_cards = {"SKI":1,"REV":1,"PL2":1}
    wild_cards = {"PL4":1,"COL":1}

    # Special cards
    norm_cards_play = {"RED#":1,"GRE#":1,"BLU#":1,"YEL#":1}
    spec_cards_play = {"SKI#":1,"REV#":1,"PL2#":1}

    # Combine dictionaries
    states_dict  = {
        **norm_cards, 
        **spec_cards, 
        **wild_cards, 
        **norm_cards_play, 
        **spec_cards_play
    }
    states = [["RED", "GRE", "BLU", "YEL"]]

    for val in states_dict.values():
        aux = range(0,val+1)
        states.append(aux)

    # Conduct all combinations
    states = list(itertools.product(*states))
    states_all = list()

    for i in range(len(states)):
        if (
            states[i][1] >= states[i][10] and
            states[i][2] >= states[i][11] and
            states[i][3] >= states[i][12] and
            states[i][4] >= states[i][13] and
            states[i][5] >= states[i][14] and
            states[i][6] >= states[i][15] and
            states[i][7] >= states[i][16]
        ): 
            states_all.append(states[i])




    states_all = ["wasabi_in_hand", "egg_nigiri_in_hand", "salmon_nigiri_in_hand", 
             "squid_nigiri_in_hand", "tempura_in_hand", "sashimi_in_hand", "dumpling_in_hand",
             "pudding_in_hand", "maki_1_in_hand", "maki_2_in_hand", "maki_3_in_hand", 
             "chopsticks_in_hand", "free_wasabi_on_table", "free_tempura_on_table",
             "free_sashimi_on_table", "dumpling_on_table", "pudding_on_table",
             "maki_points_on_table", "cards_in_hand"]

    return states_all

def actions():
    '''Return list of all possible actions (written as strings)'''
    
    actions_all = ["play_wasabi", "play_highest_nigiri", "play_tempura", "play_sashimi", "play_dumpling",
                   "play_pudding", "play_highest_maki", "play_chopsticks"]
    
    return actions_all

def rewards(states, actions):

    # Monte Carlo agent:
    # 
    # Reward policy (per hand):
    # 
    # R = points_earned_during_hand - highest_points_earned_by_any_other_player
    #
    # Q-Learning agent:
    #
    # Reward policy (per turn):
    # 
    # r = points_earned_during_turn + winning_bonus_at_end + margin_of_victory_bonus_at_end
    
    n_states = len(states)
    n_actions = len(actions)

    # Create array of zeros
    R = np.zeros((n_states, n_actions))

    # Create dataframe from array
    R = pd.DataFrame(
        data=R, 
        columns=actions, 
        index=states)
    
    return R



state_labels = states()
action_labels = actions()
rewards_df = rewards(state_labels, action_labels)

# print(state_labels)
# print(action_labels)

print(type(rewards_df))
print(rewards_df)