import pandas as pd
import numpy as np
import random
import os
import state_action_reward as sar


class Agent(object):
    def __init__(self, agent_info: dict):
        """Initialize the agent, set parameters, and create empty Q-table."""
        self.epsilon   = agent_info.get("epsilon", 0.1)
        self.step_size = agent_info.get("step_size", 0.5)
        self.states    = sar.states()
        self.actions   = sar.actions()
        self.R         = sar.rewards(self.states, self.actions)

        # Q-table and visit counts
        self.q = pd.DataFrame(
            data=np.zeros((len(self.states), len(self.actions))),
            columns=self.actions,
            index=self.states
        )
        self.visit = self.q.copy()

    # -----------------------------
    # Persistence (save/load)
    # -----------------------------
    def save(self, filename_base="q_table"):
        """Save Q-table both as binary (.npy) and human-readable (.csv)."""
        np.save(filename_base + ".npy", self.q.values)
        self.q.to_csv(filename_base + ".csv")
        print(f"Saved Q-table to {filename_base}.npy and {filename_base}.csv")

    def load(self, filename_base="q_table"):
        """Load Q-table from .npy (preferred) or .csv."""
        npy_file = filename_base + ".npy"
        csv_file = filename_base + ".csv"
        if os.path.exists(npy_file):
            values = np.load(npy_file)
            self.q.loc[:, :] = values
            print(f"Loaded Q-table from {npy_file}")
        elif os.path.exists(csv_file):
            self.q = pd.read_csv(csv_file, index_col=0)
            print(f"Loaded Q-table from {csv_file}")


# ==========================================================
# Q-Learning Agent
# ==========================================================

class QLearningAgent(Agent):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        # Q-learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration

        # initialize Q-table
        self.states_all = sar.states()
        self.actions_all = sar.actions()
        self.q = sar.rewards(self.states_all, self.actions_all)

        # previous state/action for updates
        self.prev_state = None
        self.prev_action = None

    def _state_to_tuple(self, state_dict):
        """
        Convert state_dict into the ordered tuple expected by the Q-table index.
        Missing keys default to 0.
        """
        return (
            state_dict.get("wasabi_in_hand", 0),
            state_dict.get("egg_nigiri_in_hand", 0),
            state_dict.get("salmon_nigiri_in_hand", 0),
            state_dict.get("squid_nigiri_in_hand", 0),
            state_dict.get("tempura_in_hand", 0),
            state_dict.get("sashimi_in_hand", 0),
            state_dict.get("dumpling_in_hand", 0),
            state_dict.get("pudding_in_hand", 0),
            state_dict.get("maki_1_in_hand", 0),
            state_dict.get("maki_2_in_hand", 0),
            state_dict.get("maki_3_in_hand", 0),
            state_dict.get("chopsticks_in_hand", 0),
            state_dict.get("free_wasabi_on_table", 0),
            state_dict.get("free_tempura_on_table", 0),
            state_dict.get("free_sashimi_on_table", 0),
            state_dict.get("dumpling_on_table", 0),
            state_dict.get("pudding_on_table", 0),
            state_dict.get("maki_points_on_table", 0),
            state_dict.get("cards_in_hand", 0),
        )

    def step(self, state_dict, actions_dict):
        """
        Choose the next action using epsilon-greedy policy.
        state_dict: dict of card counts/features
        actions_dict: dict {action: 0/1} indicating valid actions
        """
        state = self._state_to_tuple(state_dict)

        # Exploration
        if random.random() < self.epsilon:
            actions_possible = [a for a, v in actions_dict.items() if v != 0]
            action = random.choice(actions_possible)
        else:
            # Exploitation: pick best action
            actions_possible = [a for a, v in actions_dict.items() if v != 0]
            random.shuffle(actions_possible)
            best_val, action = -np.inf, None
            for a in actions_possible:
                val = self.q.loc[state, a]
                if val >= best_val:
                    best_val, action = val, a

        return action

    def update(self, state_dict, action):
        """
        Update Q-values using Bellman equation.
        state_dict: dict of card counts/features
        action: action string taken
        """
        state = self._state_to_tuple(state_dict)

        if self.prev_state is not None:
            prev_q = self.q.loc[self.prev_state, self.prev_action]
            this_q = self.q.loc[state, action]
            reward = self.R.loc[state, action]

            # Q-learning update rule
            new_val = prev_q + self.step_size * ((reward + this_q) - prev_q)
            self.q.loc[self.prev_state, self.prev_action] = new_val

            self.visit.loc[self.prev_state, self.prev_action] += 1

        # Save current step for next update
        self.prev_state  = state
        self.prev_action = action


# ==========================================================
# Monte Carlo Agent
# ==========================================================

class MonteCarloAgent(Agent):
    def __init__(self, agent_info: dict):
        super().__init__(agent_info)
        self.state_seen  = []
        self.action_seen = []
        self.q_seen      = []

    def _state_to_tuple(self, state_dict):
        return (
            state_dict.get("wasabi_in_hand", 0),
            state_dict.get("egg_nigiri_in_hand", 0),
            state_dict.get("salmon_nigiri_in_hand", 0),
            state_dict.get("squid_nigiri_in_hand", 0),
            state_dict.get("tempura_in_hand", 0),
            state_dict.get("sashimi_in_hand", 0),
            state_dict.get("dumpling_in_hand", 0),
            state_dict.get("pudding_in_hand", 0),
            state_dict.get("maki_1_in_hand", 0),
            state_dict.get("maki_2_in_hand", 0),
            state_dict.get("maki_3_in_hand", 0),
            state_dict.get("chopsticks_in_hand", 0),
            state_dict.get("free_wasabi_on_table", 0),
            state_dict.get("free_tempura_on_table", 0),
            state_dict.get("free_sashimi_on_table", 0),
            state_dict.get("dumpling_on_table", 0),
            state_dict.get("pudding_on_table", 0),
            state_dict.get("maki_points_on_table", 0),
            state_dict.get("cards_in_hand", 0),
        )

    def step(self, state_dict, actions_dict):
        state = self._state_to_tuple(state_dict)

        if random.random() < self.epsilon:
            actions_possible = [a for a, v in actions_dict.items() if v != 0]
            action = random.choice(actions_possible)
        else:
            actions_possible = [a for a, v in actions_dict.items() if v != 0]
            random.shuffle(actions_possible)
            best_val, action = -np.inf, None
            for a in actions_possible:
                val = self.q.loc[state, a]
                if val >= best_val:
                    best_val, action = val, a

        if ((state), action) not in self.q_seen:
            self.state_seen.append(state)
            self.action_seen.append(action)
        self.q_seen.append(((state), action))
        self.visit.loc[state, action] += 1

        return action

    def update(self, state_dict, action):
        state = self._state_to_tuple(state_dict)
        reward = self.R.loc[state, action]

        for s, a in zip(self.state_seen, self.action_seen):
            self.q.loc[s, a] += self.step_size * (reward - self.q.loc[s, a])

        self.state_seen, self.action_seen, self.q_seen = [], [], []
