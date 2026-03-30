params = {
    "iterations": int(1e5),  # number of games to simulate
    "save_every": int(1e3),  # how often to checkpoint trainable agents
    "winner_bonus": 10,      # flat reward to winners
    "enable_observation_learning": False,  # when True, agents also learn from other players' transitions
    "alpha_obs_factor": 0.2, # observation learning factor
    "logging": True,         # master switch for DQN diagnostics CSV logging
    "dqn_log_every_updates": 100,   # log one DQN diagnostics row every N optimizer steps
    "dqn_stats_flush_every": 1000,  # flush buffered DQN diagnostics rows every N logged rows
    "q_table_save_csv": False,      # periodic Q-table checkpoints skip the huge CSV by default
    "q_table_save_csv_on_final": True,  # export the giant human-readable CSV only at final save
    "q_learning": {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.2,
    },
    "dqn": {
        "hidden": 128,
        "gamma": 0.99,
        "lr": 1e-5,
        "epsilon": 0.1,
        "epsilon_decay": 0.99999,
        "epsilon_min": 0.01,
        "buffer_size": 50000,
        "batch_size": 64,
        "target_update": 5000,
        "tau": 0.005,
        "min_replay_size": None,
        "device": None,
    },
}
