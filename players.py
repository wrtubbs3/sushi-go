from agents import QLearningAgent

class Player:
    def __init__(self, name, strategy):
        """
        Player name is a string.
        Strategy is one of: "random", "sequential", "user choice", "hierarchy", "q-learning"
        """
        self.points = 0
        self.cards_in_hand = []
        self.cards_on_table = []
        self.name = name

        # Default strategy: random
        if strategy not in ["random", "sequential", "user choice", "hierarchy", "q-learning"]:
            self.strategy = "random"
        else:
            self.strategy = strategy

        # If applicable, determine agent
        if self.strategy == "q-learning":
            self.agent = QLearningAgent
        else:
            self.agent = []

    def __str__(self):
        return f"Player Name: {self.name}\n"
