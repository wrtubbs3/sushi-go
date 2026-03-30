params = {
    # Total number of full Sushi Go games to simulate in one training/eval run.
    "iterations": int(1e5),

    # Save trainable agents every N games during a run.
    "save_every": int(1e3),

    # Multiplier applied to terminal margin reward:
    # terminal_reward = terminal_margin_scale * (own_score - best_opponent_score)
    "terminal_margin_scale": 1.0,

    # When True, agents also learn from other players' observed transitions.
    # This increases training signal but can slow runs down substantially.
    "enable_observation_learning": False,

    # Scales the learning rate for tabular Q-learning observation updates.
    # Only relevant when observation learning is enabled.
    "alpha_obs_factor": 0.2,

    # Master switch for writing DQN diagnostics to dqn_stats.csv.
    "logging": True,

    # Write one diagnostics row every N DQN optimizer steps.
    "dqn_log_every_updates": 100,

    # Flush buffered DQN diagnostics rows to disk every N logged rows.
    "dqn_stats_flush_every": 1000,

    # For tabular Q-learning checkpoints, skip the large CSV during periodic saves.
    # The pickle file is still written.
    "q_table_save_csv": False,

    # For the final tabular Q-learning save, also export the large human-readable CSV.
    "q_table_save_csv_on_final": True,

    # Hyperparameters for the tabular Q-learning agent.
    "q_learning": {
        # Learning rate for Q-value updates.
        "alpha": 0.1,

        # Discount factor for future rewards.
        "gamma": 0.9,

        # Exploration probability for epsilon-greedy action selection.
        "epsilon": 0.2,
    },

    # Hyperparameters for the deep Q-learning agent.
    "dqn": {
        # Width of each hidden layer in the neural network.
        "hidden": 128,

        # Discount factor for future rewards.
        "gamma": 0.99,

        # Adam optimizer learning rate.
        "lr": 1e-5,

        # Initial exploration probability for epsilon-greedy action selection.
        "epsilon": 0.1,

        # Multiplicative epsilon decay applied after each action selection.
        "epsilon_decay": 0.99999,

        # Lower bound for epsilon after decay.
        "epsilon_min": 0.01,

        # Replay buffer capacity in transitions.
        "buffer_size": 50000,

        # Minibatch size sampled from replay during each learning step.
        "batch_size": 64,

        # Frequency, in environment steps, for hard target-network syncs.
        "target_update": 5000,

        # Polyak averaging coefficient for soft target-network updates.
        "tau": 0.005,

        # Minimum replay-buffer size before gradient updates begin.
        # Use None to fall back to max(10000, batch_size).
        "min_replay_size": None,

        # Torch device override, e.g. "cpu" or "cuda". Use None for auto-detect.
        "device": None,
    },
}
