from agents import QLearningAgent, DeepQLearningAgent

class Player:
    def __init__(self, name, strategy, q_table_file=None):
        """
        Player name is a string.
        Strategy is one of: "random", "sequential", "user choice", "hierarchy", "q-learning", "deep q-learning".
        If strategy == "q-learning", you can optionally provide q_table_file to load a saved Q-table.
        """
        self.points = 0
        self.cards_in_hand = []
        self.cards_on_table = []
        self.name = name

        # Default strategy: random
        if strategy not in ["random", "sequential", "user choice", "hierarchy", "q-learning", "deep-q-learning"]:
            self.strategy = "random"
        else:
            self.strategy = strategy

        # If applicable, determine agent
        if self.strategy == "q-learning":
            self.agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

            # Load Q-table if provided
            if q_table_file is not None:
                try:
                    self.agent.load(q_table_file)
                    print(f"[INFO] {self.name} loaded Q-table from {q_table_file}")
                except Exception as e:
                    print(f"[WARN] Could not load Q-table for {self.name} from {q_table_file}: {e}")

        elif self.strategy == "deep q-learning":
            # Instantiate a deep Q-learning agent
            self.agent = DeepQLearningAgent()

        else:
            self.agent = None

    def __str__(self):
        return f"Player Name: {self.name}\n"
