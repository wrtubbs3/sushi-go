# Sushi Go card game

# Imports
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
player_strategies = ['deep q-learning', 'hierarchy', 'hierarchy', 'hierarchy']
player_qtables = [None, None, None, None]  # only used for q-learning

# Create game
game = SushiGo(n_players, player_names, player_strategies, player_qtables=player_qtables)

# Collect references to any trainable agents
trainable_agents = []
for p in game.players:
    if p.strategy == "q-learning":
        p.agent.train = False
        if p.agent.train == True:
            trainable_agents.append(p.agent)
    elif p.strategy == "deep q-learning":
        p.agent.train = True
        if p.agent.train == True:
            trainable_agents.append(p.agent)

# Number of games to simulate
n_games = config.params['iterations']

# Initialize game log for statistical tracking
game_score_log = [[] for _ in range(n_players)]

# Filenames for saving agents
q_table_filename = "q_table"
dqn_filename = "dqn_agent"

# Simulate games
for i in tqdm(range(n_games)):
    game_score = game.play_game()

    for j in range(n_players):
        game_score_log[j].append(game_score[j])

    for agent in trainable_agents:
        agent.games_trained += 1

    # Periodically save agents
    save_interval = config.params['save_every']
    if (i + 1) % save_interval == 0:
        for agent in trainable_agents:
            if agent.__class__.__name__ == "QLearningAgent":
                agent.save(q_table_filename)
            elif agent.__class__.__name__ == "DeepQLearningAgent":
                agent.save(dqn_filename)

# Save final versions
for agent in trainable_agents:
    if agent.__class__.__name__ == "QLearningAgent":
        agent.save(q_table_filename)
    elif agent.__class__.__name__ == "DeepQLearningAgent":
        agent.save(dqn_filename)

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
    window = int(n_games/100) if n_games >= 100 else 1
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
