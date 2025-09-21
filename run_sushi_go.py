# Sushi Go card game

# Imports9
import datetime
import statistics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from game import SushiGo
import config

# From 2-5 players allowed
player_names = ['Al', 'Bob', 'Charlie', 'Doug']
n_players = len(player_names)
player_strategies = ['hierarchy', 'hierarchy', 'q-learning', 'q-learning']
# player_qtables = [None, 'q_table.pkl', 'q_table.pkl', 'q_table.pkl']

game = SushiGo(n_players, player_names, player_strategies)
# game = SushiGo(n_players, player_names, player_strategies, player_qtables=player_qtables)

# Keep a reference to the Q-learning agent (if present)
q_agent = None
for p in game.players:
    if p.strategy == "q-learning":
        q_agent = p.agent
        break

# Number of games to simulate
n_games = config.params['iterations']

# Initialize game log for statistical tracking
game_score_log = [[] for _ in range(n_players)]

# Simulate games
for i in tqdm(range(n_games)):
    game_score = game.play_game()

    for j in range(n_players):
        game_score_log[j].append(game_score[j])

    if q_agent:
        q_agent.games_trained += 1

    # Periodically save Q-table if applicable
    save_interval = config.params['save_every']
    if q_agent and (i + 1) % save_interval == 0:
        fname = f"q_table"
        q_agent.save(fname)

# run = tournament(
#         iterations = config.params['iterations'],
#         algo       = config.params['algorithm'],
#         comment    = config.params['logging'],
#         agent_info = config.params['model']
#                 )

# Save final Q-table if applicable
if q_agent:
    q_agent.save("q_table")

# Compute statistics for each player
total_score = []
avg_game_score = []
stdev_game_score = []
for i in range(n_players):
    total_score.append(sum(game_score_log[i]))
    avg_game_score.append(round(total_score[i] / n_games, 2))
    stdev_game_score.append(round(statistics.pstdev(game_score_log[i]), 2))

# Print results for each player
for i in range(n_players):
    print(player_names[i], 'finished with', total_score[i], 'points using the', player_strategies[i], 'strategy.')
    # print('Game log: ', game_score_log[i])
    print(player_names[i], 'finished with an average score of', avg_game_score[i], 'points using the', player_strategies[i], 'strategy.')    
    print(player_names[i], 'finished with standard deviation of', stdev_game_score[i], 'points using the', player_strategies[i], 'strategy.')    

# ---------------------------------------------------
# Plotting raw and smoothed scores
# ---------------------------------------------------

plt.figure(figsize=(12, 6))
for i in range(n_players):
    raw_scores = np.array(game_score_log[i])

    # Smoothed (rolling average with window)
    window = 100
    smoothed = np.convolve(raw_scores, np.ones(window)/window, mode='valid')

    plt.plot(range(1, len(raw_scores)+1), raw_scores, alpha=0.2, label=f"{player_names[i]} (raw)")
    plt.plot(range(window, len(raw_scores)+1), smoothed, label=f"{player_names[i]} (avg)")

plt.xlabel("Game Number")
plt.ylabel("Score")
plt.title("Sushi Go Scores During Training")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Add timestamp to filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"sushi_go_training_{timestamp}.png"

# Save plot as PNG
plt.savefig(filename, dpi=300)

# Show on screen
plt.show()

print(f"[INFO] Plot saved as {filename}")

# # Determine which player(s) have the most points and print output
# most_points = max(game_score)
# winner_idx = [i for i in range(n_players) if (game_score[i] == most_points)]
# n_winners = len(winner_idx)
# if n_winners == 1:
#     winner = player_names[winner_idx[0]]
#     print(f"{winner} is the winner!")
# else:
#     winner_list = []
#     print('There are multiple players who tied for the most points:')
#     for i in range(n_winners):
#         winner_list.append(player_names[winner_idx[i]])
#         print(winner_list[i])
