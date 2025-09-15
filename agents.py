# agents.py
import random
import numpy as np
import pandas as pd
import state_action_reward as sar

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2, train=True):
        """
        Dictionary-based Q-learning agent.
        alpha = learning rate
        gamma = discount factor
        epsilon = exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.train = train

        # Action space
        self.actions = sar.actions()

        # Dictionary for Q-values: keys are (state_tuple, action), values are floats
        self.q = {}

        # Track last state/action for updates
        self.prev_state = None
        self.prev_action = None

    # -------------------------
    # State handling
    # -------------------------
    def _state_to_tuple(self, state_dict):
        """
        Convert ordered state_dict into a tuple (used as dict key).
        Must match the feature order defined in state_action_reward.states().
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

    # -------------------------
    # Q-value helpers
    # -------------------------
    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def set_q(self, state, action, value):
        self.q[(state, action)] = value

    # -------------------------
    # Action selection
    # -------------------------
    def step(self, state_dict, actions_dict):
        """
        Choose the next action using epsilon-greedy policy.
        state_dict: dict of card counts/features
        actions_dict: dict {action: True/False} indicating valid actions
        """
        state = self._state_to_tuple(state_dict)
        possible_actions = [a for a, ok in actions_dict.items() if ok]

        if not possible_actions:
            return random.choice(self.actions)  # fallback

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
        else:
            # Exploit: choose action with highest Q
            q_vals = [(a, self.get_q(state, a)) for a in possible_actions]
            random.shuffle(q_vals)  # break ties randomly
            action = max(q_vals, key=lambda x: x[1])[0]

        # Store for update
        self.prev_state = state
        self.prev_action = action
        return action

    # -------------------------
    # Q-learning update
    # -------------------------
    def update(self, reward, state_dict, actions_dict):
        """
        Perform Q-learning update after observing a transition.
        """
        if not self.train:
            return
        
        if self.prev_state is None or self.prev_action is None:
            return

        state = self._state_to_tuple(state_dict)
        possible_actions = [a for a, ok in actions_dict.items() if ok]

        if possible_actions:
            max_future_q = max(self.get_q(state, a) for a in possible_actions)
        else:
            max_future_q = 0.0

        old_q = self.get_q(self.prev_state, self.prev_action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.set_q(self.prev_state, self.prev_action, new_q)

        # Reset previous state/action
        self.prev_state, self.prev_action = None, None

    # -------------------------
    # Save & Load
    # -------------------------
    def save(self, filename_base="q_table"):
        """Save Q-table as .npy (fast) and .csv (human-readable)."""
        # Convert dict → DataFrame for CSV
        if not self.q:
            print("[WARN] No Q-values to save.")
            return

        states, actions, values = zip(*[(s, a, v) for (s, a), v in self.q.items()])
        df = pd.DataFrame({"state": states, "action": actions, "value": values})
        df.to_csv(filename_base + ".csv", index=False)

        # Save numpy version too
        np.save(filename_base + ".npy", self.q, allow_pickle=True)
        print(f"[INFO] Q-table saved to {filename_base}.csv and {filename_base}.npy")

    def load(self, filename_base="q_table"):
        """Load Q-table from .npy if available."""
        try:
            self.q = np.load(filename_base + ".npy", allow_pickle=True).item()
            print(f"[INFO] Q-table loaded from {filename_base}.npy")
        except FileNotFoundError:
            print(f"[WARN] No saved Q-table found at {filename_base}.npy — starting fresh.")
