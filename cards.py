# Imports
from random import shuffle

class Card:

    def __init__(self, type, subtype):
        """Card [type] is a string, options are:
        "wasabi", "nigiri", "tempura", "sashimi", "dumpling", "pudding", "maki", "chopsticks"
        \nCard [subtype] must be 1, 2, or 3 if type is "nigiri" or "maki", otherwise [subtype] is "n/a" """
        self.type = type
        self.subtype = subtype

        if ((type == "wasabi") or (type == "tempura") or (type == "sashimi") or (type == "dumpling") 
            or (type == "pudding") or (type == "chopsticks")):
            self.subtype = "n/a"
        elif (type == "nigiri"):
            if not ((subtype == 1) or (subtype == 2) or (subtype == 3)):
                print('Error! "Nigiri" cards are only allowed a subtype of 1, 2, or 3.')
                exit()
        elif (type == "maki"):
            if not ((subtype == 1) or (subtype == 2) or (subtype == 3)):
                print('Error! "Maki" cards are only allowed a subtype of 1, 2, or 3.')
                exit()
        else:
            print('Error! Invalid card type.')
            exit()
    
    def __str__(self):
        return f"card type: {self.type}\ncard subtype: {self.subtype}\n"

class Deck:
    # Quantity of cards in Deck:
    #
    # (6) wasabi (Value: 3x multiplier for next nigiri)
    # (5) egg nigiri (a.k.a. nigiri-1) (Value: 1 point)
    # (10) salmon nigiri (a.k.a. nigiri-2) (Value: 2 points)
    # (5) squid nigiri (a.k.a. nigiri-3) (Value: 3 points)
    # (14) tempura (Value: 5 points for 2 cards)
    # (14) sashimi (Value: 10 points for 3 cards)
    # (14) dumpling (Value: 1 point for 1 card, 3 points for 2 cards, 6 points for 3 cards, 
    #                10 points for 4 cards, 15 points for 5 cards)
    # (10) pudding (Value: 6 points for most, -6 points for least)
    # (6) maki-1 (Value: 6 points for most total maki, 3 points for second most)
    # (12) maki-2 (Value: 6 points for most total maki, 3 points for second most)
    # (8) maki-3 (Value: 6 points for most total maki, 3 points for second most)
    # (4) chopsticks (Value: no value, can be used to swap later)
    #
    # Total deck: (108) cards

    def __init__(self):
        """A Sushi Go deck consists of 108 cards."""

        self.cards = []

        for i in range(6):
            self.cards.append(Card("wasabi", "n/a"))
        for i in range(5):
            self.cards.append(Card("nigiri", 1)) # egg nigiri
        for i in range(10):
            self.cards.append(Card("nigiri", 2)) # salmon nigiri
        for i in range(5):
            self.cards.append(Card("nigiri", 3)) # squid nigiri
        for i in range(14):
            self.cards.append(Card("tempura", "n/a"))
        for i in range(14):
            self.cards.append(Card("sashimi", "n/a"))
        for i in range(14):
            self.cards.append(Card("dumpling", "n/a"))
        for i in range(10):
            self.cards.append(Card("pudding", "n/a"))
        for i in range(6):
            self.cards.append(Card("maki", 1)) # maki-1
        for i in range(12):
            self.cards.append(Card("maki", 2)) # maki-2
        for i in range(8):
            self.cards.append(Card("maki", 3)) # maki-3
        for i in range(4):
            self.cards.append(Card("chopsticks", "n/a"))
        
        shuffle(self.cards)

    def rm_card(self):
        """Draw card from deck"""
        if len(self.cards) == 0:
            return
        return self.cards.pop() # returns card from last (highest) index position
    
    def reset(self):
        """Return all cards to deck and shuffle deck"""
        'Note: In practice, this method restores the deck to an initial state'

        self.cards = []

        for i in range(6):
            self.cards.append(Card("wasabi", "n/a"))
        for i in range(5):
            self.cards.append(Card("nigiri", 1)) # egg nigiri
        for i in range(10):
            self.cards.append(Card("nigiri", 2)) # salmon nigiri
        for i in range(5):
            self.cards.append(Card("nigiri", 3)) # squid nigiri
        for i in range(14):
            self.cards.append(Card("tempura", "n/a"))
        for i in range(14):
            self.cards.append(Card("sashimi", "n/a"))
        for i in range(14):
            self.cards.append(Card("dumpling", "n/a"))
        for i in range(10):
            self.cards.append(Card("pudding", "n/a"))
        for i in range(6):
            self.cards.append(Card("maki", 1)) # maki-1
        for i in range(12):
            self.cards.append(Card("maki", 2)) # maki-2
        for i in range(8):
            self.cards.append(Card("maki", 3)) # maki-3
        for i in range(4):
            self.cards.append(Card("chopsticks", "n/a"))
        
        shuffle(self.cards)