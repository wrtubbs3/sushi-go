import config
from agents import QLearningAgent, DeepQLearningAgent

class Player:
    def __init__(self, name, strategy, q_table_file=None, agent=None):
        """
        Player name is a string.
        Strategy is one of: "random", "sequential", "user choice", "hierarchy", "q-learning", "deep q-learning".
        If strategy == "q-learning", you can optionally provide q_table_file to load a saved Q-table.
        If agent is provided, reuse that already-instantiated agent instead of creating/loading one here.
        """
        self.points = 0
        self.cards_in_hand = []
        self.cards_on_table = []
        self.name = name

        # Default strategy: random
        if strategy not in ["random", "sequential", "user choice", "hierarchy", "q-learning", "deep q-learning"]:
            self.strategy = "random"
        else:
            self.strategy = strategy

        # Reuse an existing agent when the caller wants one persistent learner
        # to participate in many freshly-created games.
        if agent is not None:
            self.agent = agent
            return

        # If applicable, determine agent
        if self.strategy == "q-learning":
            q_config = config.params.get("q_learning", {})
            self.agent = QLearningAgent(
                alpha=q_config.get("alpha", 0.1),
                gamma=q_config.get("gamma", 0.9),
                epsilon=q_config.get("epsilon", 0.2),
            )

            # Load Q-table if provided
            if q_table_file is not None:
                try:
                    self.agent.load(q_table_file)
                    print(f"[INFO] {self.name} loaded Q-table from {q_table_file}")
                except Exception as e:
                    print(f"[WARN] Could not load Q-table for {self.name} from {q_table_file}: {e}")

        elif self.strategy == "deep q-learning":
            # Instantiate a deep Q-learning agent
            dqn_config = config.params.get("dqn", {})
            self.agent = DeepQLearningAgent(
                hidden=dqn_config.get("hidden", 128),
                gamma=dqn_config.get("gamma", 0.99),
                lr=dqn_config.get("lr", 1e-5),
                epsilon=dqn_config.get("epsilon", 0.1),
                epsilon_decay=dqn_config.get("epsilon_decay", 0.99999),
                epsilon_min=dqn_config.get("epsilon_min", 0.01),
                buffer_size=dqn_config.get("buffer_size", 50000),
                batch_size=dqn_config.get("batch_size", 64),
                target_update=dqn_config.get("target_update", 5000),
                tau=dqn_config.get("tau", 0.005),
                min_replay_size=dqn_config.get("min_replay_size"),
                device=dqn_config.get("device"),
            )

            # Load agent file if provided
            if q_table_file is not None:
                try:
                    # `DeepQLearningAgent.load` appends ".pkl", so accept either
                    # a name with or without the suffix from the caller.
                    fname = q_table_file
                    if fname.endswith('.pkl'):
                        fname = fname[:-4]
                    self.agent.load(fname)
                    print(f"[INFO] {self.name} loaded DQN agent from {q_table_file}")
                except Exception as e:
                    print(f"[WARN] Could not load DQN agent for {self.name} from {q_table_file}: {e}")

        else:
            self.agent = None

    def __str__(self):
        return f"Player Name: {self.name}\n"
