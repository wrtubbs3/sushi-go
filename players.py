class Player:
    def __init__(self, name, strategy):
        """Player name is a string. Strategy is a string, options are: "random", "sequential", "user choice", "hierarchy" """
        self.points = 0
        self.cards_in_hand = []
        self.cards_on_table = []
        self.name = name
        
        # Default strategy: random
        if not ((strategy == "random") or (strategy == "sequential") or (strategy == "user choice") or ("hierarchy")):
            self.strategy = "random"
        else:
            self.strategy = strategy

    def __str__(self):
        return f"Player Name: {self.name}\n"