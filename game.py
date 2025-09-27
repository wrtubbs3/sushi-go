# Imports
import random
import time
from math import floor
from sys import exit
from cards import *
from state_action_reward import *
from players import Player
import config

class SushiGo:
    def __init__(self, n, player_names, player_strategies, player_qtables=None):
        """[n] is an integer representing the number of players. [player_names] is a length-n list 
        of strings containing each player's name. [player_strategies] is a lengh-n list containing
        each player's strategy."""

        # Check that each player has a name
        if not (len(player_names) == n):
            print('Error! The list of player names must contain exactly one name for every player.')
            exit()
        
        # Check that each player has a defined strategy
        if not (len(player_strategies) == n):
            print('Error! The list of player strategies must contain exactly one strategy for every player.')
            exit()

        self.n_players = n
        self.deck = Deck()

        # From 2-5 players allowed
        # Number of cards dealt to each player depends on number of players
        if self.n_players == 2:
            self.n_cards_dealt_per_player = 10
        elif self.n_players == 3:
            self.n_cards_dealt_per_player = 9
        elif self.n_players == 4:
            self.n_cards_dealt_per_player = 8
        elif self.n_players == 5:
            self.n_cards_dealt_per_player = 7
        else:
            print('Error! This game is only for 2-5 players.')
            return
        
        if player_qtables is None:
            player_qtables = [None] * n
        
        # Create players
        self.players = []
        for i in range(self.n_players):
            # j = i + 1
            # name = input(f"Input Player {j} Name:  ")
            # strategy = input(f"Input Player {j} Strategy: ")
            # self.players.append(Player(name, strategy))
            self.players.append(Player(player_names[i], player_strategies[i], player_qtables[i]))

    def play_game(self):
        
        # Play three hands in each game
        for hand in range(3):
            # Deal cards to each player
            for i in range(self.n_cards_dealt_per_player):
                for j in range(self.n_players):
                    # Draw card
                    self.players[j].cards_in_hand.append(self.deck.rm_card())

                    # if i == (self.n_cards_dealt_per_player - 1):
                    #     print(f"{self.players[j].name} has the following cards in hand: ")
                    #     for k in range(self.n_cards_dealt_per_player):
                    #         print(self.players[j].cards_in_hand[k])
            
            # Play hand, until all cards are on table
            for i in range(self.n_cards_dealt_per_player):
                self.players = self.play_hand(self.players)

            # for i in range(self.n_players):
            #     print(f"{self.players[i].name} has the following cards on the table: ")
            #     for j in range(len(self.players[i].cards_on_table)):
            #         print(self.players[i].cards_on_table[j])

            # Count points on table for each player and discard cards in hand and on table
            points_on_table = self.count_points(self.players)
            for i in range(self.n_players):
                self.players[i].points = self.players[i].points + points_on_table[i]
                self.players[i].cards_in_hand = []
                self.players[i].cards_on_table = []
        
        # --- END OF GAME BONUS UPDATE ---

        # Compute winners
        final_points = [p.points for p in self.players]
        max_points = max(final_points)
        winners = [i for i, pts in enumerate(final_points) if pts == max_points]

        # Config parameters
        winner_bonus = getattr(config.params, "winner_bonus", 10)
        alpha_obs_factor = getattr(config.params, "alpha_obs_factor", 0.2)

        for i, player in enumerate(self.players):
            if player.strategy == "q-learning" and player.agent is not None:
                # Terminal next state (no cards left, no valid actions)
                next_state_dict = {"cards_in_hand": 0}
                next_actions_dict = {a: False for a in player.agent.actions}

                # Assign reward
                final_reward = winner_bonus if i in winners else 0

                # Full update for this player's own final reward
                player.agent.update(
                    reward=final_reward,
                    state_dict=next_state_dict,
                    actions_dict=next_actions_dict
                )

                # Observational updates for other Q-learning players
                for j, other_player in enumerate(self.players):
                    if j == i:
                        continue
                    if other_player.strategy == "q-learning" and other_player.agent is not None:
                        observed_reward = winner_bonus if j in winners else 0
                        other_player.agent.update_from_observation(
                            state_dict=next_state_dict,
                            action=None,  # terminal observation
                            reward=observed_reward,
                            next_state_dict=next_state_dict,
                            next_actions_dict=next_actions_dict,
                            alpha_obs=other_player.agent.alpha * alpha_obs_factor
                        )

        # Create list of each player's points total, and reset player points to zero for next game
        total_points = []
        for i in range(self.n_players):
            total_points.append(self.players[i].points)
            self.players[i].points = 0

        # Reset deck at end of game
        self.deck.reset()

        return total_points

    def count_points(self, players):
        
        n_players = len(players)
        n_cards_on_table = len(self.players[0].cards_on_table)

        cards_on_table = []
        n_pudding = []
        n_maki_total = []

        # Initialize list of points to give each player for various card types
        pts_pudding = [0]*n_players
        pts_maki = [0]*n_players
        pts_nigiri = [0]*n_players
        pts_tempura = [0]*n_players
        pts_sashimi = [0]*n_players
        pts_dumpling = [0]*n_players
        pts_total = [0]*n_players

        # Create list of all players hands
        for i in range(n_players):
            cards_on_table.append(self.players[i].cards_on_table)

            # Count puddings for all players
            n_pudding.append(sum(1 for card in cards_on_table[i] if card.type == "pudding"))       

            # Count maki for all players
            n_maki_1 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 1)))
            n_maki_2 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 2)))
            n_maki_3 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 3)))
            n_maki_total.append(n_maki_1*1 + n_maki_2*2 + n_maki_3*3)

        # Award points for pudding

        most_pudding = max(n_pudding)
        least_pudding = min(n_pudding)

        if not (most_pudding == least_pudding): # award 0 points to all players if all have equal # of puddings
            
            most_pudding_player_idx = [i for i in range(n_players) if n_pudding[i] == most_pudding]
            most_pudding_value = floor(6/len(most_pudding_player_idx)) # divide 6 points evenly among players (round down)
            for idx in most_pudding_player_idx:
                pts_pudding[idx] = most_pudding_value

            
            least_pudding_player_idx = [i for i in range(n_players) if n_pudding[i] == least_pudding]
            least_pudding_value = (-1)*floor(6/len(least_pudding_player_idx)) # divide -6 points evenly among players (round down)
            for idx in least_pudding_player_idx:
                pts_pudding[idx] = least_pudding_value
        
        # Award points for maki

        most_maki = max(n_maki_total)

        if not (most_maki == 0): # award 0 points to all players if nobody has maki

            most_maki_player_idx = [i for i in range(n_players) if n_maki_total[i] == most_maki]
            n_players_with_most_maki = len(most_maki_player_idx)

            most_maki_value = floor(6/n_players_with_most_maki) # divide 6 points evenly among players (round down)

            for idx in most_maki_player_idx:
                pts_maki[idx] = most_maki_value

            if n_players_with_most_maki == 1: # if one person has most with no ties, divide 3 points amongst second most
                # Modfiy n_maki_total list to zero out maximum value, so 2nd-highest value can be easily evaluated
                n_maki_total[most_maki_player_idx[0]] = 0
                second_most_maki = max(n_maki_total)
                second_most_maki_player_idx = [i for i in range(n_players) if n_maki_total[i] == second_most_maki]
                n_players_with_second_most_maki = len(second_most_maki_player_idx)

                second_most_maki_value = floor(3/n_players_with_second_most_maki) # divide 3 points evenly among players (round down)

                for idx in second_most_maki_player_idx:
                    pts_maki[idx] = second_most_maki_value

        # Compute points from nigiri, tempura, sashimi, and dumpling for each player
        for i in range(n_players):

            # Award points for nigiri
            n_wasabi = sum(1 for card in cards_on_table[i] if card.type == "wasabi")
            n_egg_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 1)))
            n_salmon_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 2)))
            n_squid_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 3)))
            if n_wasabi == 0:
                pts_nigiri[i] = n_egg_nigiri*1 + n_salmon_nigiri*2 + n_squid_nigiri*3
            else:
                wasabi_idx_list = [j for j in range(n_cards_on_table) if cards_on_table[i][j].type == "wasabi"]
                nigiri_idx_list = [j for j in range(n_cards_on_table) if cards_on_table[i][j].type == "nigiri"]

                # for each wasabi
                for wasabi_idx in wasabi_idx_list:
                    # if there is a next nigiri
                    if (len(nigiri_idx_list) != 0) and (nigiri_idx_list[-1] > wasabi_idx):
                        # find index (position) of next nigiri
                        next_nigiri_idx = next(x for x, val in enumerate(nigiri_idx_list) if val > wasabi_idx)  
                        # wasabi card applies a 3x multiplier to value of next nigiri
                        next_nigiri_value = cards_on_table[i][nigiri_idx_list[next_nigiri_idx]].subtype * 3
                        # remove next nigiri from list to avoid duplicating
                        nigiri_idx_list.pop(next_nigiri_idx)
                        # add next nigiri value to nigiri points subtotal
                        pts_nigiri[i] = pts_nigiri[i] + next_nigiri_value
                    else:
                        break

                # add value of all remaining nigiri cards
                for nigiri_idx in nigiri_idx_list:
                    pts_nigiri[i] = pts_nigiri[i] + cards_on_table[i][nigiri_idx].subtype
            
            # Award points for tempura
            n_tempura = sum(1 for card in cards_on_table[i] if card.type == "tempura")
            pts_tempura[i] = floor(n_tempura/2)*5

            # Award points for sashimi
            n_sashimi = sum(1 for card in cards_on_table[i] if card.type == "sashimi")
            pts_sashimi[i] = floor(n_sashimi/3)*10

            # Award points for dumplings
            n_dumpling = sum(1 for card in cards_on_table[i] if card.type == "dumpling")          
            if n_dumpling == 1:
                pts_dumpling[i] = 1
            elif n_dumpling == 2:
                pts_dumpling[i] = 3
            elif n_dumpling == 3:
                pts_dumpling[i] = 6
            elif n_dumpling == 4:
                pts_dumpling[i] = 10
            elif n_dumpling >= 5:
                pts_dumpling[i] = 15

            # Sum together points from all types
            pts_total[i] = pts_pudding[i] + pts_maki[i] + pts_nigiri[i] + pts_tempura[i] + pts_sashimi[i] + pts_dumpling[i]

        return pts_total
    
    def play_hand(self, players):
        n_players = len(players)

        # --- BEFORE ACTION: points baseline ---
        old_points = self.count_points(players)

        cards_in_hand = []
        cards_on_table = []
        cards_passed_left = []

        # Create list of all players' hands
        for i in range(n_players):
            cards_in_hand.append(players[i].cards_in_hand)
            cards_on_table.append(players[i].cards_on_table)

        # Each player selects one card to keep
        actions_taken = []  # track state/action/reward transitions for observation
        for i in range(n_players):
            player = players[i]

            # Build current state_dict (pre-action)
            state_dict = build_state_dict(cards_in_hand[i], cards_on_table[i])

            # Build dictionary of possible actions from current state
            actions_dict = build_actions_dict(state_dict)
            
            # --- SELECT CARD ---
            card_to_keep_idx = self.select_card(cards_in_hand[i], cards_on_table[i], player)
            card_to_keep = cards_in_hand[i].pop(card_to_keep_idx)
            player.cards_on_table.append(card_to_keep)

            # Record state/action for later reward assignment
            chosen_action = None
            if player.strategy == "q-learning" and player.agent is not None:
                chosen_action = player.agent.prev_action  # stored inside step()
            elif chosen_action is None:
                # fallback: infer from card type
                chosen_action = "play_" + card_to_keep.type

            actions_taken.append((player, state_dict, actions_dict, chosen_action))

        # --- AFTER ACTION: new points ---
        new_points = self.count_points(players)

        # Update agent for each player as applicable
        for i in range(n_players):
            player = players[i]
            state_dict, actions_dict, chosen_action = actions_taken[i][1], actions_taken[i][2], actions_taken[i][3]
            reward = new_points[i] - old_points[i]  # incremental reward

            # Construct next state dictionary
            next_state_dict = build_state_dict(player.cards_in_hand, player.cards_on_table)         

            # Construct dictionary of possible actions from next state
            next_actions_dict = build_actions_dict(next_state_dict)

            # --- SELF UPDATE ---
            if player.strategy == "q-learning" and player.agent is not None:
                player.agent.update(reward, next_state_dict, next_actions_dict)

            # --- OBSERVATION UPDATES ---
            if chosen_action is not None:
                for other_player in players:
                    if other_player is not player and other_player.strategy == "q-learning" and other_player.agent is not None:
                        other_player.agent.update_from_observation(
                            state_dict=state_dict,
                            action=chosen_action,
                            reward=reward,
                            next_state_dict=next_state_dict,
                            next_actions_dict=next_actions_dict,
                            alpha_obs=other_player.agent.alpha * 0.2  # smaller learning rate for observers
                        )

            # Track remaining cards to pass left
            cards_passed_left.append(cards_in_hand[i])

        # Pass remaining cards to the left
        for i in range(n_players):
            if i == 0:
                players[0].cards_in_hand = cards_in_hand[-1]
            else:
                players[i].cards_in_hand = cards_in_hand[i-1]
        
        return players
    
    def select_card(self, cards_in_hand, cards_on_table, player):
        """Select card to keep based on the strategy selected. 'cards_in_hand' and 'cards_on_table' 
        are both lists of cards. 'player' is a Player object."""

        n_cards_in_hand = len(cards_in_hand)
        n_cards_on_table = len(cards_on_table)
        strategy = player.strategy

        # For each strategy, identify index of card to keep
        if n_cards_in_hand == 1:
            card_to_keep_idx = 0
        
        elif strategy == "random":
            card_to_keep_idx = random.randint(0, n_cards_in_hand - 1)
        
        elif strategy == "sequential":
            card_to_keep_idx = 0
        
        elif strategy == "user choice":
            print("\nUser must select which card to keep.")
            
            # Display cards on table
            print("Cards currently on table: ")
            for i in range(n_cards_on_table):
                print(cards_on_table[i])                    

            # Display cards in hand
            print("Cards currently in hand: ")
            for i in range(n_cards_in_hand):
                print(cards_in_hand[i])  

            # Prompt user to choose
            card_to_keep_idx = int(input(f"Enter integer index of card to keep. (Note that indices start at 0.)  "))
        
        elif strategy == "hierarchy":
            # Simple hierarchy (in descending order): pudding, squid nigiri (nigiri-3), 
            # sashimi, wasabi, tempura, salmon nigiri (nigiri-2), dumpling, maki-3, 
            # egg nigiri (nigiri-1), maki-2, maki-1, chopsticks

            # Initialize hierarchy level for while loop
            level = 0
            level_max = 12 # There are 12 types of cards and thus 12 levels in hierarchy

            card_to_keep_idx = 0 # Initialize default value for index

            # Select index of first pudding in hand. If no pudding, move on.
            while level < level_max:
                level = level + 1
                if level == 1:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "pudding":
                            card_to_keep_idx = i
                            break
                    break
                elif level == 2:
                    for i in range(n_cards_in_hand):
                        if ((cards_in_hand[i].type == "nigiri") and (cards_in_hand[i].subtype == 3)):
                            card_to_keep_idx = i
                            break
                    break
                elif level == 3:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "sashimi":
                            card_to_keep_idx = i
                            break
                    break
                elif level == 4:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "wasabi":
                            card_to_keep_idx = i
                            break
                    break
                elif level == 5:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "tempura":
                            card_to_keep_idx = i
                            break
                    break
                elif level == 6:
                    for i in range(n_cards_in_hand):
                        if ((cards_in_hand[i].type == "nigiri") and (cards_in_hand[i].subtype == 2)):
                            card_to_keep_idx = i
                            break
                    break
                elif level == 7:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "dumpling":
                            card_to_keep_idx = i
                            break
                    break
                elif level == 8:
                    for i in range(n_cards_in_hand):
                        if ((cards_in_hand[i].type == "maki") and (cards_in_hand[i].subtype == 3)):
                            card_to_keep_idx = i
                            break
                    break
                elif level == 9:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "nigiri": # egg nigiri (nigiri-1)
                            card_to_keep_idx = i
                            break
                    break
                elif level == 10:
                    for i in range(n_cards_in_hand):
                        if ((cards_in_hand[i].type == "maki") and (cards_in_hand[i].subtype == 2)):
                            card_to_keep_idx = i
                            break
                    break
                elif level == 11:
                    for i in range(n_cards_in_hand):
                        if cards_in_hand[i].type == "maki": # maki-1
                            card_to_keep_idx = i
                            break
                    break
                else: # all remaining cards are chopsticks
                    card_to_keep_idx = 0
                    break
        
        elif strategy == "q-learning":
            
            # Construct current state dictionary
            state_dict = build_state_dict(cards_in_hand, cards_on_table)

            # Construct dictionary of possible actions
            actions_dict = build_actions_dict(state_dict)

            # Call the player’s agent
            action = player.agent.step(state_dict, actions_dict)

            # Map chosen action back to card index
            card_to_keep_idx = 0
            for i, c in enumerate(cards_in_hand):
                if action.endswith(c.type):  # match "play_tempura" to card.type == "tempura"
                    card_to_keep_idx = i
                    break

        else: # default
            card_to_keep_idx = 0

        return card_to_keep_idx
    