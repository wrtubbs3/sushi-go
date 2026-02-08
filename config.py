params = {
    "iterations": int(1e4),  # number of games to simulate
    "save_every": int(1e3),  # how often to save Q-table
    "winner_bonus": 10,     # flat reward to winners
    "alpha_obs_factor": 0.2,  # observation learning factor
    "algorithm": "q-learning", # ["q-learning", "monte-carlo"]
    "logging": False,
    "model": {
        "epsilon": 0.4,
        "step_size": 0.2,
    }
}