# agents.py
import os
import csv
import time
import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import state_action_reward as sar
from torch.nn.utils import clip_grad_norm_

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
    def update(self, reward, state_dict, actions_dict, done=False):
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
    def __init__(self, state_dim=None, action_dim=None, gamma=0.99, lr=1e-4, epsilon=0.2, epsilon_decay = 0.999,
                 epsilon_min = 0.05, train=True, buffer_size=50000, batch_size=64, target_update=5000, 
                 min_replay_size=None, device=None):
        """
        Deep Q-Learning agent with explicit, canonical action->index mapping to avoid
        action-order mismatches between the network and environment.
        - state_dim / action_dim: if None, derived from sar.states() and sar.actions().
        - gamma: discount factor for future rewards
        - lr: learning rate
        - epsilon: exploration rate for epsilon-greedy action selection
        - train: whether the agent is in training mode (enables exploration and learning)
        - buffer_size: replay buffer size
        - batch_size: minibatch size for learning
        - target_update: how often (in steps) to update the target network
        - min_replay_size: how many transitions before learning starts (defaults to max(1000, batch_size)).
        - device: torch device to use (cpu by default).
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.train = train
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[INFO] Using device: {self.device}")

        # State and action spaces (canonicalized)
        if state_dim is None:
            self.states = sar.states()
            self.state_dim = len(self.states)
        else:
            self.state_dim = state_dim

        # canonical action list
        self.actions = sar.actions()
        self.action_dim = len(self.actions) if action_dim is None else action_dim

        # mapping action_name -> index and index -> action_name
        self.action_to_index = {a: i for i, a in enumerate(self.actions)}
        self.index_to_action = {i: a for i, a in enumerate(self.actions)}

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.min_replay_size = min_replay_size if min_replay_size is not None else max(1000, batch_size)

        # Networks
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Training stats logging
        self.stats_file = "dqn_stats.csv"   # set to None to disable logging
        # create header if needed
        if self.stats_file is not None and not os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "steps_done",
                        "games_trained",
                        "loss",
                        "td_error",
                        "avg_q",
                        "epsilon",
                        "replay_size",
                        "batch_size",
                        "grad_norm"
                    ])
            except Exception as e:
                print(f"[WARN] Could not create stats file {self.stats_file}: {e}")

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
        return torch.tensor(state_vec, dtype=torch.float32, device=self.device)

    # -------------------------
    # Action selection
    # -------------------------
    def step(self, state_dict, actions_dict):
        """Epsilon-greedy selection with masking of invalid actions.
        actions_dict: mapping action_name -> boolean (available)
        Returns action_name (string) consistent with self.actions list.
        """
        state = self._state_to_vector(state_dict).unsqueeze(0)  # shape [1, state_dim]

        # Build list of valid action indices using the canonical mapping.
        valid_actions = [self.action_to_index[a] for a, ok in actions_dict.items() if ok and a in self.action_to_index]

        # If no valid actions
        if not valid_actions:
            chosen_idx = random.randrange(self.action_dim)
        else:
            # Exploration
            if self.train and random.random() < self.epsilon:
                chosen_idx = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = self.q_net(state).squeeze(0)  # shape [action_dim]
                    # mask out invalid action indices by setting them to -inf so argmax never picks them
                    mask = torch.full_like(q_values, float("-inf"))
                    mask[valid_actions] = 0.0
                    chosen_idx = torch.argmax(q_values + mask).item()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Store last state/action for the environment -> update call
        # keep the state tensor (not moved to numpy) so transitions can be stacked later
        self.prev_state = state.detach()  # shape [1, state_dim]
        self.prev_action = int(chosen_idx)
        return self.index_to_action[chosen_idx]

    # -------------------------
    # Store transition
    # -------------------------

    def update(self, reward, state_dict, actions_dict, done=False):
        """Store transition and trigger learning. Intended to be called after environment step."""
        if not self.train:
            return

        next_state = self._state_to_vector(state_dict)
        # prev_state stored as [1, state_dim]; squeeze to [state_dim] to make stacking easier
        if self.prev_state is None or self.prev_action is None:
            # if no previous stored state/action, skip (happens at beginning)
            return

        transition = Transition(self.prev_state.squeeze(0).cpu(),  # store on cpu to reduce GPU memory pressure
                                self.prev_action,
                                float(reward),
                                next_state.cpu(),
                                bool(done))
        self.memory.append(transition)

        self.steps_done += 1
        # Only learn after we have a reasonable buffer size (avoid overfitting to tiny buffer)
        if len(self.memory) >= max(self.batch_size, self.min_replay_size):
            self.learn()

        # Target net hard update on schedule
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def update_from_observation(self, state_dict, action, reward, next_state_dict, next_actions_dict):
        """
        Allow the Deep Q agent to learn from observing other players' transitions.

        This stores an observed transition in the replay buffer with a scaled reward
        so observational updates have smaller effect (similar in spirit to the tabular
        agent's smaller learning rate for observations).

        Parameters:
        - state_dict: observed state (before the other player's action)
        - action: observed action name (string) or None for terminal observation
        - reward: observed reward (float)
        - next_state_dict: observed next state
        - next_actions_dict: observed valid actions in next state (dict action->bool)
        """
        if not self.train:
            return

        # If no action provided (terminal observation), skip: the terminal full-update should already be applied
        if action is None:
            return

        # Ensure action exists in canonical mapping
        if action not in self.action_to_index:
            # unknown action (shouldn't happen) — skip
            return

        # Prepare tensors (store on CPU like normal transitions)
        state = self._state_to_vector(state_dict).cpu()       # shape [state_dim]
        next_state = self._state_to_vector(next_state_dict).cpu()

        action_idx = self.action_to_index[action]

        # determine done from next_actions_dict: if no valid actions, treat as terminal
        done = not any(next_actions_dict.values()) if next_actions_dict is not None else False

        # Append an observed transition to replay buffer
        transition = Transition(state, action_idx, reward, next_state, bool(done))
        self.memory.append(transition)

        # Only learn after we have a reasonable buffer size (avoid overfitting to tiny buffer)
        if len(self.memory) >= max(self.batch_size, self.min_replay_size):
            self.learn()

    # -------------------------
    # Training step
    # -------------------------
    def learn(self):
        """Sample a minibatch and perform Bellman update (uses Huber loss and gradient clipping)."""
        # Sample
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))

        # Convert to tensors on device
        states = torch.stack(batch.state).to(self.device)                       # [batch, state_dim]
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)   # [batch, 1]
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor([1.0 if d else 0.0 for d in batch.done], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a) for taken actions
        q_values = self.q_net(states).gather(1, actions)  # [batch, 1]

        # Targets computed with target network (detached)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            q_targets = rewards + self.gamma * (1.0 - dones) * max_next_q

        # Use Huber (SmoothL1) loss for robustness to outliers
        loss = nn.SmoothL1Loss()(q_values, q_targets)

        # Compute TD error (mean absolute) for diagnostics
        with torch.no_grad():
            td_error = (q_values - q_targets).abs().mean().item()

        # Batch diagnostics
        with torch.no_grad():
            rewards_tensor = rewards  # [batch,1]
            batch_reward_mean = float(rewards_tensor.mean().item())
            batch_reward_max = float(rewards_tensor.max().item())
            n_terminals = int(dones.sum().item())
            # stats of the next-state target Q (before any update)
            max_next_q_stat = float(max_next_q.mean().item()) if 'max_next_q' in locals() else float(self.target_net(next_states).max(1, keepdim=True)[0].mean().item())
            q_targets_mean = float(q_targets.mean().item())
            # detect whether a hard target update just happened (useful for spike correlation)
            is_target_update = bool(self.target_update and (self.steps_done % self.target_update == 0))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm (L2) before clipping for diagnostics
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += (param_norm.item() ** 2)
        grad_norm = total_norm ** 0.5 if total_norm > 0.0 else 0.0

        # gradient clipping to avoid exploding updates
        clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Optional diagnostics logging to CSV
        if getattr(self, "stats_file", None):
            try:
                # compute avg q for batch for logging
                avg_q = q_values.mean().item()
                replay_size = len(self.memory)
                batch_size = self.batch_size
                row = [
                    int(time.time()),
                    int(self.steps_done),
                    int(self.games_trained),
                    float(loss.item()),
                    float(td_error),
                    float(avg_q),
                    float(self.epsilon),
                    int(replay_size),
                    int(batch_size),
                    float(grad_norm),
                    float(batch_reward_mean),
                    float(batch_reward_max),
                    int(n_terminals),
                    float(max_next_q_stat),
                    float(q_targets_mean),
                    int(is_target_update)
                ]
                with open(self.stats_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                # don't crash training on logging failure
                print(f"[WARN] Failed to write DQN stats: {e}")
    
    # -------------------------
    # Save & Load
    # -------------------------
    def save(self, filename_base="dqn_agent"):
        """Save model + optimizer state."""
        if not self.train:
            return

        data = {
            "model_state_dict": self.q_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "games_trained": self.games_trained,
            "epsilon": self.epsilon,
            # action mapping included so loading later preserves canonical order
            "actions": self.actions
        }
        with open(filename_base + ".pkl", "wb") as f:
            pickle.dump(data, f)

        print(f"[INFO] Deep Q agent saved to {filename_base}.pkl (steps={self.steps_done})")

    def load(self, filename="dqn_agent"):
        """Load model + optimizer state if file exists."""
        try:
            with open(filename + ".pkl", "rb") as f:
                data = pickle.load(f)
            self.q_net.load_state_dict(data.get("model_state_dict", {}))
            opt_state = data.get("optimizer_state_dict", None)
            if opt_state is not None:
                try:
                    self.optimizer.load_state_dict(opt_state)
                except Exception:
                    # optimizer load can fail if device or pytorch versions differ; ignore safely
                    pass
            self.steps_done = data.get("steps_done", self.steps_done)
            self.games_trained = data.get("games_trained", self.games_trained)
            self.epsilon = data.get("epsilon", self.epsilon)
            saved_actions = data.get("actions", None)
            if saved_actions:
                # if saved action ordering differs, rebuild mappings but prefer the saved order
                self.actions = saved_actions
                self.action_to_index = {a: i for i, a in enumerate(self.actions)}
                self.index_to_action = {i: a for i, a in enumerate(self.actions)}
            print(f"[INFO] DQN agent loaded from {filename}.pkl (steps_done={self.steps_done})")
        except FileNotFoundError:
            print(f"[WARN] No saved DQN agent found at {filename}.pkl — starting fresh.")
