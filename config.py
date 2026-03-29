params = {
    "iterations": int(1e5),  # number of games to simulate
    "save_every": int(1e3),  # how often to checkpoint trainable agents
    "winner_bonus": 10,      # flat reward to winners
    "alpha_obs_factor": 0.2, # observation learning factor
    "algorithm": "q-learning", # ["q-learning", "monte-carlo"]
    "logging": False,        # master switch for DQN diagnostics CSV logging
    "dqn_log_every_updates": 100,   # log one DQN diagnostics row every N optimizer steps
    "dqn_stats_flush_every": 1000,  # flush buffered DQN diagnostics rows every N logged rows
    "q_table_save_csv": False,      # periodic Q-table checkpoints skip the huge CSV by default
    "q_table_save_csv_on_final": True,  # export the giant human-readable CSV only at final save
    "model": {
        "epsilon": 0.4,
        "step_size": 0.2,
    }
}
