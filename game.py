# Imports
import random
from math import floor
from sys import exit

from cards import *
from players import Player
from state_action_reward import *
import config


def card_action_name(card):
    """Map a concrete card back to the abstract action space used by the agents."""
    if card.type == "nigiri":
        return "play_highest_nigiri"
    if card.type == "maki":
        return "play_highest_maki"

    action_map = {
        "wasabi": "play_wasabi",
        "tempura": "play_tempura",
        "sashimi": "play_sashimi",
        "dumpling": "play_dumpling",
        "pudding": "play_pudding",
        "chopsticks": "play_chopsticks",
    }
    return action_map.get(card.type)


def card_label(card):
    """Human-friendly label for simple text-based card rendering."""
    if card.type == "nigiri":
        return f"Nigiri-{card.subtype}"
    if card.type == "maki":
        return f"Maki-{card.subtype}"
    return card.type.replace("_", " ").title()


class SushiGo:
    def __init__(self, n, player_names, player_strategies, player_qtables=None, player_agents=None):
        """[n] is an integer representing the number of players. [player_names] is a length-n list
        of strings containing each player's name. [player_strategies] is a length-n list containing
        each player's strategy. [player_agents] optionally contains pre-built agent objects to
        attach to each player, which is useful when reusing a single training agent across many
        episode-specific seatings."""

        if not (len(player_names) == n):
            print("Error! The list of player names must contain exactly one name for every player.")
            exit()

        if not (len(player_strategies) == n):
            print("Error! The list of player strategies must contain exactly one strategy for every player.")
            exit()

        self.n_players = n
        self.deck = Deck()
        self.total_hands = 3

        if self.n_players == 2:
            self.n_cards_dealt_per_player = 10
        elif self.n_players == 3:
            self.n_cards_dealt_per_player = 9
        elif self.n_players == 4:
            self.n_cards_dealt_per_player = 8
        elif self.n_players == 5:
            self.n_cards_dealt_per_player = 7
        else:
            print("Error! This game is only for 2-5 players.")
            return

        if player_qtables is None:
            player_qtables = [None] * n
        if player_agents is None:
            player_agents = [None] * n

        self.players = []
        for i in range(self.n_players):
            self.players.append(Player(player_names[i], player_strategies[i], player_qtables[i], player_agents[i]))

        self.game_started = False
        self.game_finished = False
        self.awaiting_next_hand = False
        self.current_hand_number = 0
        self.current_turn_number = 0
        self.last_hand_summary = None
        self.last_game_summary = None

    def _reset_agent_runtime_state(self, player):
        """Clear per-game transient decision state while preserving learned parameters."""
        if player.agent is None:
            return

        for attr in ("prev_state", "prev_action", "last_state", "last_action"):
            if hasattr(player.agent, attr):
                setattr(player.agent, attr, None)

    def _clear_player_runtime_state(self, reset_points=True):
        for player in self.players:
            if reset_points:
                player.points = 0
            player.cards_in_hand = []
            player.cards_on_table = []
            self._reset_agent_runtime_state(player)

    def reset_after_game(self):
        """Return the final scores, then restore a clean pre-game state."""
        total_points = [player.points for player in self.players]

        self._clear_player_runtime_state(reset_points=True)
        self.deck.reset()
        self.game_started = False
        self.game_finished = False
        self.awaiting_next_hand = False
        self.current_hand_number = 0
        self.current_turn_number = 0
        self.last_hand_summary = None
        self.last_game_summary = None

        return total_points

    def start_game(self):
        """Reset all runtime state and deal the opening hand."""
        self._clear_player_runtime_state(reset_points=True)
        self.deck.reset()
        self.game_started = True
        self.game_finished = False
        self.awaiting_next_hand = False
        self.current_hand_number = 0
        self.current_turn_number = 0
        self.last_hand_summary = None
        self.last_game_summary = None
        self.deal_next_hand()

    def deal_next_hand(self):
        """Deal a fresh hand to each player. Call once at game start or after acknowledging hand scoring."""
        if self.game_finished:
            return False

        if self.current_hand_number >= self.total_hands:
            return False

        for player in self.players:
            if player.cards_in_hand or player.cards_on_table:
                raise RuntimeError("Cannot deal a new hand while cards are still in play.")

        self.current_hand_number += 1
        self.current_turn_number = 0
        self.awaiting_next_hand = False

        for _ in range(self.n_cards_dealt_per_player):
            for player in self.players:
                player.cards_in_hand.append(self.deck.rm_card())

        return True

    def advance_to_next_hand(self):
        """Move from an end-of-hand pause into the next hand."""
        if self.game_finished or not self.awaiting_next_hand:
            return False

        self.last_hand_summary = None
        return self.deal_next_hand()

    def get_game_state(self):
        """Return a lightweight snapshot for renderers or alternate runners."""
        return {
            "game_started": self.game_started,
            "game_finished": self.game_finished,
            "awaiting_next_hand": self.awaiting_next_hand,
            "current_hand_number": self.current_hand_number,
            "current_turn_number": self.current_turn_number,
            "players": [
                {
                    "name": player.name,
                    "strategy": player.strategy,
                    "points": player.points,
                    "cards_in_hand": list(player.cards_in_hand),
                    "cards_on_table": list(player.cards_on_table),
                }
                for player in self.players
            ],
        }

    def play_game(self):
        """Play a full game start-to-finish, preserving the legacy training/eval API."""
        self.start_game()

        while not self.game_finished:
            turn_summary = self.play_turn()
            if turn_summary["hand_complete"] and not turn_summary["game_complete"]:
                self.advance_to_next_hand()

        return self.reset_after_game()

    def finish_game(self):
        """Finalize terminal rewards and produce a game summary without resetting scores yet."""
        if self.last_game_summary is not None:
            return self.last_game_summary

        final_points = [p.points for p in self.players]
        terminal_margin_scale = config.params.get("terminal_margin_scale", 1.0)

        for i, player in enumerate(self.players):
            if player.agent is not None and player.strategy in {"q-learning", "deep q-learning"}:
                best_opponent_score = max(score for j, score in enumerate(final_points) if j != i)
                final_reward = terminal_margin_scale * (final_points[i] - best_opponent_score)
                player.agent.apply_terminal_reward(final_reward)

        winning_score = max(final_points)
        winner_indices = [i for i, score in enumerate(final_points) if score == winning_score]
        winner_names = [self.players[i].name for i in winner_indices]

        self.game_finished = True
        self.awaiting_next_hand = False
        self.last_game_summary = {
            "final_points": final_points,
            "winning_score": winning_score,
            "winner_indices": winner_indices,
            "winner_names": winner_names,
            "is_tie": len(winner_indices) > 1,
            "players": [
                {
                    "name": player.name,
                    "strategy": player.strategy,
                    "points": player.points,
                }
                for player in self.players
            ],
        }
        return self.last_game_summary

    def score_current_hand(self):
        """Count points, update totals, and clear the table in preparation for the next hand."""
        points_on_table = self.count_points(self.players)
        player_summaries = []
        table_snapshot = [
            {
                "player_index": i,
                "name": player.name,
                "strategy": player.strategy,
                "cards_in_hand": list(player.cards_in_hand),
                "cards_on_table": list(player.cards_on_table),
                "points_before_hand_score": player.points,
            }
            for i, player in enumerate(self.players)
        ]

        for i, player in enumerate(self.players):
            player.points += points_on_table[i]
            player_summaries.append(
                {
                    "name": player.name,
                    "strategy": player.strategy,
                    "hand_points": points_on_table[i],
                    "total_points": player.points,
                }
            )
            player.cards_in_hand = []
            player.cards_on_table = []

        self.last_hand_summary = {
            "hand_number": self.current_hand_number,
            "points_earned": points_on_table,
            "totals": [player.points for player in self.players],
            "players": player_summaries,
            "table_snapshot": table_snapshot,
        }
        return self.last_hand_summary

    def play_turn(self, manual_card_indices=None):
        """Play exactly one draft step for every player.

        `manual_card_indices` can override the selected card index for one or more
        players, which is how the GUI supplies the user's click choice.
        """
        if not self.game_started:
            self.start_game()

        if self.game_finished:
            raise RuntimeError("Cannot play another turn after the game has finished.")

        if self.awaiting_next_hand:
            raise RuntimeError("This hand is over. Call advance_to_next_hand() before playing again.")

        manual_card_indices = manual_card_indices or {}
        n_players = len(self.players)
        observation_learning_enabled = config.params.get("enable_observation_learning", False)
        alpha_obs_factor = config.params.get("alpha_obs_factor", 0.2)

        old_points = self.count_points(self.players)
        actions_taken = []

        for i in range(n_players):
            player = self.players[i]
            state_dict = build_state_dict(player.cards_in_hand, player.cards_on_table)
            actions_dict = build_actions_dict(state_dict)

            if i in manual_card_indices:
                card_to_keep_idx = int(manual_card_indices[i])
                if not 0 <= card_to_keep_idx < len(player.cards_in_hand):
                    raise IndexError(f"Manual choice {card_to_keep_idx} is out of range for player {player.name}.")
            else:
                card_to_keep_idx = self.select_card(player.cards_in_hand, player.cards_on_table, player)

            card_to_keep = player.cards_in_hand.pop(card_to_keep_idx)
            player.cards_on_table.append(card_to_keep)

            chosen_action = None
            if player.agent is not None and player.strategy in {"q-learning", "deep q-learning"}:
                chosen_action = getattr(player.agent, "prev_action", None)
                if isinstance(chosen_action, int) and hasattr(player.agent, "index_to_action"):
                    chosen_action = player.agent.index_to_action.get(chosen_action)

            if chosen_action is None:
                chosen_action = card_action_name(card_to_keep)

            actions_taken.append(
                {
                    "player": player,
                    "player_index": i,
                    "state_dict": state_dict,
                    "actions_dict": actions_dict,
                    "chosen_action": chosen_action,
                    "chosen_card": card_to_keep,
                    "chosen_card_label": card_label(card_to_keep),
                    "manual": i in manual_card_indices,
                }
            )

        new_points = self.count_points(self.players)
        rewards = []
        for i in range(n_players):
            reward = new_points[i] - old_points[i]
            rewards.append(reward)
            actions_taken[i]["reward"] = reward

        passed_hands = [player.cards_in_hand for player in self.players]
        for i, player in enumerate(self.players):
            player.cards_in_hand = passed_hands[i - 1]

        for i in range(n_players):
            player = self.players[i]
            turn_record = actions_taken[i]
            state_dict = turn_record["state_dict"]
            actions_dict = turn_record["actions_dict"]
            chosen_action = turn_record["chosen_action"]
            reward = rewards[i]

            next_state_dict = build_state_dict(player.cards_in_hand, player.cards_on_table)
            next_actions_dict = build_actions_dict(next_state_dict)

            if player.strategy == "q-learning" and player.agent is not None:
                player.agent.update(reward, next_state_dict, next_actions_dict)
            elif player.strategy == "deep q-learning" and player.agent is not None:
                player.agent.update(reward, next_state_dict, next_actions_dict)

            if observation_learning_enabled and chosen_action is not None:
                for other_player in self.players:
                    if other_player is player or other_player.agent is None:
                        continue
                    if other_player.strategy == "q-learning":
                        other_player.agent.update_from_observation(
                            state_dict=state_dict,
                            action=chosen_action,
                            reward=reward,
                            next_state_dict=next_state_dict,
                            next_actions_dict=next_actions_dict,
                            alpha_obs=other_player.agent.alpha * alpha_obs_factor,
                        )
                    elif other_player.strategy == "deep q-learning":
                        other_player.agent.update_from_observation(
                            state_dict=state_dict,
                            action=chosen_action,
                            reward=reward,
                            next_state_dict=next_state_dict,
                            next_actions_dict=next_actions_dict,
                        )

        self.current_turn_number += 1
        hand_complete = len(self.players[0].cards_in_hand) == 0

        turn_summary = {
            "hand_number": self.current_hand_number,
            "turn_number": self.current_turn_number,
            "selected_cards": [
                {
                    "player_index": record["player_index"],
                    "player_name": record["player"].name,
                    "card_label": record["chosen_card_label"],
                    "reward": record["reward"],
                    "manual": record["manual"],
                }
                for record in actions_taken
            ],
            "hand_complete": hand_complete,
            "game_complete": False,
        }

        if hand_complete:
            turn_summary["hand_summary"] = self.score_current_hand()
            if self.current_hand_number >= self.total_hands:
                turn_summary["game_complete"] = True
                turn_summary["game_summary"] = self.finish_game()
            else:
                self.awaiting_next_hand = True

        return turn_summary

    def count_points(self, players):
        n_players = len(players)
        n_cards_on_table = len(self.players[0].cards_on_table)

        cards_on_table = []
        n_pudding = []
        n_maki_total = []

        pts_pudding = [0] * n_players
        pts_maki = [0] * n_players
        pts_nigiri = [0] * n_players
        pts_tempura = [0] * n_players
        pts_sashimi = [0] * n_players
        pts_dumpling = [0] * n_players
        pts_total = [0] * n_players

        for i in range(n_players):
            cards_on_table.append(self.players[i].cards_on_table)

            n_pudding.append(sum(1 for card in cards_on_table[i] if card.type == "pudding"))

            n_maki_1 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 1)))
            n_maki_2 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 2)))
            n_maki_3 = sum(1 for card in cards_on_table[i] if ((card.type == "maki") and (card.subtype == 3)))
            n_maki_total.append(n_maki_1 * 1 + n_maki_2 * 2 + n_maki_3 * 3)

        most_pudding = max(n_pudding)
        least_pudding = min(n_pudding)

        if not (most_pudding == least_pudding):
            most_pudding_player_idx = [i for i in range(n_players) if n_pudding[i] == most_pudding]
            most_pudding_value = floor(6 / len(most_pudding_player_idx))
            for idx in most_pudding_player_idx:
                pts_pudding[idx] = most_pudding_value

            least_pudding_player_idx = [i for i in range(n_players) if n_pudding[i] == least_pudding]
            least_pudding_value = (-1) * floor(6 / len(least_pudding_player_idx))
            for idx in least_pudding_player_idx:
                pts_pudding[idx] = least_pudding_value

        most_maki = max(n_maki_total)

        if not (most_maki == 0):
            most_maki_player_idx = [i for i in range(n_players) if n_maki_total[i] == most_maki]
            n_players_with_most_maki = len(most_maki_player_idx)

            most_maki_value = floor(6 / n_players_with_most_maki)

            for idx in most_maki_player_idx:
                pts_maki[idx] = most_maki_value

            if n_players_with_most_maki == 1:
                n_maki_total[most_maki_player_idx[0]] = 0
                second_most_maki = max(n_maki_total)
                second_most_maki_player_idx = [i for i in range(n_players) if n_maki_total[i] == second_most_maki]
                n_players_with_second_most_maki = len(second_most_maki_player_idx)

                second_most_maki_value = floor(3 / n_players_with_second_most_maki)

                for idx in second_most_maki_player_idx:
                    pts_maki[idx] = second_most_maki_value

        for i in range(n_players):
            n_wasabi = sum(1 for card in cards_on_table[i] if card.type == "wasabi")
            n_egg_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 1)))
            n_salmon_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 2)))
            n_squid_nigiri = sum(1 for card in cards_on_table[i] if ((card.type == "nigiri") and (card.subtype == 3)))
            if n_wasabi == 0:
                pts_nigiri[i] = n_egg_nigiri * 1 + n_salmon_nigiri * 2 + n_squid_nigiri * 3
            else:
                wasabi_idx_list = [j for j in range(n_cards_on_table) if cards_on_table[i][j].type == "wasabi"]
                nigiri_idx_list = [j for j in range(n_cards_on_table) if cards_on_table[i][j].type == "nigiri"]

                for wasabi_idx in wasabi_idx_list:
                    if (len(nigiri_idx_list) != 0) and (nigiri_idx_list[-1] > wasabi_idx):
                        next_nigiri_idx = next(x for x, val in enumerate(nigiri_idx_list) if val > wasabi_idx)
                        next_nigiri_value = cards_on_table[i][nigiri_idx_list[next_nigiri_idx]].subtype * 3
                        nigiri_idx_list.pop(next_nigiri_idx)
                        pts_nigiri[i] = pts_nigiri[i] + next_nigiri_value
                    else:
                        break

                for nigiri_idx in nigiri_idx_list:
                    pts_nigiri[i] = pts_nigiri[i] + cards_on_table[i][nigiri_idx].subtype

            n_tempura = sum(1 for card in cards_on_table[i] if card.type == "tempura")
            pts_tempura[i] = floor(n_tempura / 2) * 5

            n_sashimi = sum(1 for card in cards_on_table[i] if card.type == "sashimi")
            pts_sashimi[i] = floor(n_sashimi / 3) * 10

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

            pts_total[i] = (
                pts_pudding[i]
                + pts_maki[i]
                + pts_nigiri[i]
                + pts_tempura[i]
                + pts_sashimi[i]
                + pts_dumpling[i]
            )

        return pts_total

    def play_hand(self, players):
        """Backward-compatible helper that plays one full hand in one call."""
        while len(players[0].cards_in_hand) > 0:
            self.play_turn()
        return players

    def select_card(self, cards_in_hand, cards_on_table, player):
        """Select card to keep based on the strategy selected. 'cards_in_hand' and 'cards_on_table'
        are both lists of cards. 'player' is a Player object."""

        n_cards_in_hand = len(cards_in_hand)
        n_cards_on_table = len(cards_on_table)
        strategy = player.strategy

        if n_cards_in_hand == 1:
            card_to_keep_idx = 0

        elif strategy == "random":
            card_to_keep_idx = random.randint(0, n_cards_in_hand - 1)

        elif strategy == "sequential":
            card_to_keep_idx = 0

        elif strategy == "user choice":
            print("\nUser must select which card to keep.")

            print("Cards currently on table: ")
            for i in range(n_cards_on_table):
                print(cards_on_table[i])

            print("Cards currently in hand: ")
            for i in range(n_cards_in_hand):
                print(cards_in_hand[i])

            card_to_keep_idx = int(input("Enter integer index of card to keep. (Note that indices start at 0.)  "))

        elif strategy == "hierarchy":
            level = 0
            level_max = 12
            card_to_keep_idx = 0

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
                        if cards_in_hand[i].type == "nigiri":
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
                        if cards_in_hand[i].type == "maki":
                            card_to_keep_idx = i
                            break
                    break
                else:
                    card_to_keep_idx = 0
                    break

        elif strategy == "q-learning":
            state_dict = build_state_dict(cards_in_hand, cards_on_table)
            actions_dict = build_actions_dict(state_dict)
            action = player.agent.step(state_dict, actions_dict)
            card_to_keep_idx = self.map_action_to_card_index(cards_in_hand, action)

        elif strategy == "deep q-learning":
            state_dict = build_state_dict(cards_in_hand, cards_on_table)
            actions_dict = build_actions_dict(state_dict)
            action = player.agent.step(state_dict, actions_dict)
            card_to_keep_idx = self.map_action_to_card_index(cards_in_hand, action)

        else:
            card_to_keep_idx = 0

        return card_to_keep_idx

    def map_action_to_card_index(self, cards_in_hand, action):
        """Map an abstract action name to a concrete card index in the current hand."""
        if not cards_in_hand:
            return 0

        if action == "play_highest_nigiri":
            best_idx = None
            best_subtype = -1
            for i, card in enumerate(cards_in_hand):
                if card.type == "nigiri" and card.subtype > best_subtype:
                    best_idx = i
                    best_subtype = card.subtype
            return best_idx if best_idx is not None else 0

        if action == "play_highest_maki":
            best_idx = None
            best_subtype = -1
            for i, card in enumerate(cards_in_hand):
                if card.type == "maki" and card.subtype > best_subtype:
                    best_idx = i
                    best_subtype = card.subtype
            return best_idx if best_idx is not None else 0

        action_to_type = {
            "play_wasabi": "wasabi",
            "play_tempura": "tempura",
            "play_sashimi": "sashimi",
            "play_dumpling": "dumpling",
            "play_pudding": "pudding",
            "play_chopsticks": "chopsticks",
        }
        target_type = action_to_type.get(action)
        if target_type is not None:
            for i, card in enumerate(cards_in_hand):
                if card.type == target_type:
                    return i

        return 0
