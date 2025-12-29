# agents.py
import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import state_action_reward as sar

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

        # Track training progress
        self.games_trained = 0

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
        if (state, action) not in self.q:
            self.q[(state, action)] = 0.0
        return self.q[(state, action)]

    def set_q(self, state, action, value):
        self.q[(state, action)] = value

    # -------------------------
    # Action selection
    # -------------------------
    def step(self, state_dict, actions_dict):
        """
        Choose the next action using epsilon-greedy policy when training,
        or on-policy (pure exploitation) when not training.
        """
        state = self._state_to_tuple(state_dict)
        possible_actions = [a for a, ok in actions_dict.items() if ok]

        if not possible_actions:
            return random.choice(self.actions)  # fallback

        if not self.train:
            # On-policy: always pick the best action
            q_vals = [(a, self.get_q(state, a)) for a in possible_actions]
            random.shuffle(q_vals)  # break ties randomly
            action = max(q_vals, key=lambda x: x[1])[0]
            return action
        else:
            # Training: epsilon-greedy
            if random.random() < self.epsilon:
                action = random.choice(possible_actions)
            else:
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

    def update_from_observation(self, state_dict, action, reward, next_state_dict, next_actions_dict, alpha_obs=None):
        """
        Q-learning update for observing other players' transitions.
        Uses a smaller learning rate (alpha_obs).
        """
        if not self.train:
            return

        state = self._state_to_tuple(state_dict)
        next_state = self._state_to_tuple(next_state_dict)

        if any(next_actions_dict.values()):
            max_future_q = max(self.get_q(next_state, a) for a, ok in next_actions_dict.items() if ok)
        else:
            max_future_q = 0.0

        old_q = self.get_q(state, action)
        lr = alpha_obs if alpha_obs is not None else self.alpha * 0.2  # default to 20% of alpha
        new_q = old_q + lr * (reward + self.gamma * max_future_q - old_q)
        self.set_q(state, action, new_q)

    # -------------------------
    # Save & Load
    # -------------------------
    def save(self, filename_base="q_table"):
        """Save Q-table + metadata as .pkl and .csv."""
        if not self.q:
            print("[WARN] No Q-values to save.")
            return
        
        if not self.train:
            return

        # Prepare metadata
        data = {
            "q": self.q,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "games_trained": self.games_trained,
        }

        # Save pickle (full agent state)
        with open(filename_base + ".pkl", "wb") as f:
            pickle.dump(data, f)

        # Save human-readable CSV
        states, actions, values = zip(*[(s, a, v) for (s, a), v in self.q.items()])
        df = pd.DataFrame({
            "state": states,
            "action": actions,
            "value": values,
            "games_trained": self.games_trained,  # repeated in each row
        })
        df.to_csv(filename_base + ".csv", index=False)

        print(f"[INFO] Agent saved to {filename_base}.pkl, {filename_base}.csv"
              f" (games_trained={self.games_trained})")

    def load(self, filename="q_table"):
        """Load Q-table + metadata from .pkl (preferred)."""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            self.q = data.get("q", {})
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.games_trained = data.get("games_trained", 0)
            print(f"[INFO] Agent loaded from {filename}"
                  f" (games_trained={self.games_trained}, q_size={len(self.q)})")
        except FileNotFoundError:
            print(f"[WARN] No saved agent found at {filename} — starting fresh.")

# -------------------------
# Deep Q-Learning Agent
# -------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class DeepQLearningAgent:
    def __init__(self, state_dim=19, action_dim=8, gamma=0.99, lr=1e-3, epsilon=0.2, train=True, buffer_size=50000, 
                 batch_size=64, target_update=1000):
        """
        Deep Q-Learning agent.
        state_dim = length of state vector
        action_dim = number of possible action types
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.train = train
        self.batch_size = batch_size
        self.target_update = target_update

        # State space (from SAR helper)
        if state_dim is None:
            self.states = sar.states()
            self.state_dim = len(self.states)
        else:
            self.state_dim = state_dim

        # Action space (from SAR helper)
        if action_dim is None:
            self.actions = sar.actions()
            self.action_dim = len(self.actions)
        else:
            self.action_dim = action_dim

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Networks - use the computed self.state_dim (not the raw parameter)
        self.q_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Training bookkeeping
        self.steps_done = 0
        self.games_trained = 0

    # -------------------------
    # State helpers
    # -------------------------
    def _state_to_vector(self, state_dict):
        """Convert ordered state_dict to vector of counts (float32 tensor)."""
        state_vec = np.array([
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
        ], dtype=np.float32)
        return torch.tensor(state_vec)

    # -------------------------
    # Action selection
    # -------------------------
    def step(self, state_dict, actions_dict):
        """Epsilon-greedy selection with masking of invalid actions."""
        state = self._state_to_vector(state_dict).unsqueeze(0)  # shape [1, state_dim]
        valid_actions = [i for i, a in enumerate(actions_dict.keys()) if actions_dict.get(a, False)]

        if not valid_actions:
            # No valid actions: pick a random action index (but set prev_state/prev_action consistently)
            chosen_idx = random.randrange(self.action_dim)
        else:
            # Exploration
            if self.train and random.random() < self.epsilon:
                chosen_idx = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = self.q_net(state).squeeze(0)  # shape [action_dim]
                    mask = torch.full_like(q_values, float("-inf"))
                    mask[valid_actions] = 0
                    chosen_idx = torch.argmax(q_values + mask).item()

        self.prev_state = state
        self.prev_action = chosen_idx
        return self.actions[chosen_idx]

    # -------------------------
    # Store transition
    # -------------------------
    def update(self, reward, state_dict, actions_dict, done=False):
        """Store transition and trigger learning."""
        if not self.train:
            return

        next_state = self._state_to_vector(state_dict)
        transition = Transition(self.prev_state.squeeze(0),
                                self.prev_action,
                                reward,
                                next_state,
                                done)
        self.memory.append(transition)

        self.steps_done += 1
        if len(self.memory) >= self.batch_size:
            self.learn()

        # Target net update
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # -------------------------
    # Training step
    # -------------------------
    def learn(self):
        """Sample a minibatch and perform Bellman update."""
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))

        states = torch.stack(batch.state)                       # [batch, state_dim]
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)   # [batch, 1] (long)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_states = torch.stack(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)  # [batch, 1]

        # Q_target(s,a)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * (1 - dones) * max_next_q

        loss = nn.MSELoss()(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -------------------------
    # Save & Load
    # -------------------------
    def save(self, filename_base="dqn_agent"):
        """Save model + optimizer state."""
        if not self.train:
            return

        data = {
            "model_state_dict": self.q_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "steps_done": self.steps_done,
            "games_trained": self.games_trained,
        }

        torch.save(data, filename_base + ".pt")
        print(f"[INFO] DQN agent saved to {filename_base}.pt "
              f"(games_trained={self.games_trained}, steps_done={self.steps_done})")

    def load(self, filename="dqn_agent.pt"):
        """Load model + optimizer state."""
        try:
            data = torch.load(filename)
            self.q_net.load_state_dict(data["model_state_dict"])
            self.target_net.load_state_dict(data["target_state_dict"])
            self.optimizer.load_state_dict(data["optimizer_state_dict"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.gamma = data.get("gamma", self.gamma)
            self.steps_done = data.get("steps_done", 0)
            self.games_trained = data.get("games_trained", 0)
            print(f"[INFO] DQN agent loaded from {filename} "
                  f"(games_trained={self.games_trained}, steps_done={self.steps_done})")
        except FileNotFoundError:
            print(f"[WARN] No saved DQN agent found at {filename} — starting fresh.")
