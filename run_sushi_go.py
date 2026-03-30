"""Run Sushi Go training/evaluation with configurable episode lineups.

This script now supports sampling opponents from a pool and optionally
randomizing seat order every game. That makes training less brittle than
repeating one fixed set of neighbors for every episode.
"""

import datetime
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config
from game import SushiGo
from players import Player


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

# The persistent learner that we want to train or evaluate across many games.
# Set "train" to False when you want to evaluate a saved model without updates.
TRAINING_PLAYER = {
    "name": "DQN Agent",
    "strategy": "deep q-learning",
    "agent_file": None,
    "train": True,
}

# Candidate opponents. A fresh player is created from one of these specs each
# episode, so non-learning opponents do not accumulate score/card state from
# game to game. Learned opponents can still reuse one cached loaded agent so
# large model files are only read from disk once per run.
OPPONENT_POOL = [
    {"name": "Random Bot", "strategy": "random", "agent_file": None, "train": False},
    {"name": "Hierarchy Bot", "strategy": "hierarchy", "agent_file": None, "train": False},
    {"name": "Sequential Bot", "strategy": "sequential", "agent_file": None, "train": False},
    {"name": "Q Bot", "strategy": "q-learning", "agent_file": "q_table_4_players.pkl", "train": False},
]

# Total seats at the table, including the training player.
N_PLAYERS = 4

# When True, sample a new subset of opponents from OPPONENT_POOL each game.
# When False, the first (N_PLAYERS - 1) entries from OPPONENT_POOL are reused.
RANDOMIZE_OPPONENT_SELECTION = True

# When True, shuffle seat order every episode after selecting the lineup.
RANDOMIZE_SEATING_EACH_GAME = True

# Optional seed for reproducible lineup sampling. Set to None for full
# randomness from run to run.
RANDOM_SEED = None

# Base filenames for saving trainable agents.
Q_TABLE_FILENAME = "q_table"
DQN_FILENAME = "dqn_agent"


def build_persistent_training_spec(training_player):
    """Create one reusable learner instance from the training config."""
    training_stub = Player(
        training_player["name"],
        training_player["strategy"],
        training_player.get("agent_file"),
    )
    if training_stub.agent is None:
        raise ValueError("TRAINING_PLAYER must use a trainable strategy.")

    training_stub.agent.train = training_player.get("train", True)

    return {
        "name": training_player["name"],
        "strategy": training_player["strategy"],
        "agent_file": training_player.get("agent_file"),
        "train": training_player.get("train", True),
        "agent": training_stub.agent,
        "persistent": True,
    }


def normalize_player_spec(spec):
    """Return a copy of a player spec with the keys expected by this script."""
    return {
        "name": spec["name"],
        "strategy": spec["strategy"],
        "agent_file": spec.get("agent_file"),
        "train": spec.get("train", False),
        "agent": spec.get("agent"),
        "persistent": spec.get("persistent", False),
    }


def build_cached_opponent_pool(opponent_pool):
    """Preload any fixed learned opponents once so large files are not re-read."""
    cached_specs = []
    for spec in opponent_pool:
        normalized = normalize_player_spec(spec)

        # Heuristic opponents are already cheap, so keep them lightweight.
        if normalized["strategy"] not in {"q-learning", "deep q-learning"}:
            cached_specs.append(normalized)
            continue

        opponent_stub = Player(
            normalized["name"],
            normalized["strategy"],
            normalized.get("agent_file"),
        )
        if opponent_stub.agent is not None:
            opponent_stub.agent.train = normalized.get("train", False)
            normalized["agent"] = opponent_stub.agent
            normalized["persistent"] = True

        cached_specs.append(normalized)

    return cached_specs


def sample_opponents(opponent_pool, n_opponents, rng, randomize_selection):
    """Choose the non-training players for one episode."""
    if len(opponent_pool) < n_opponents:
        raise ValueError(
            f"Need at least {n_opponents} opponents in OPPONENT_POOL for a {n_opponents + 1}-player game."
        )

    if randomize_selection:
        chosen = rng.sample(opponent_pool, n_opponents)
    else:
        chosen = opponent_pool[:n_opponents]

    return [normalize_player_spec(spec) for spec in chosen]


def build_episode_lineup(training_spec, opponent_pool, n_players, rng):
    """Assemble one game's full lineup, optionally with randomized seating."""
    lineup = [normalize_player_spec(training_spec)]
    lineup.extend(
        sample_opponents(
            opponent_pool=opponent_pool,
            n_opponents=n_players - 1,
            rng=rng,
            randomize_selection=RANDOMIZE_OPPONENT_SELECTION,
        )
    )

    if RANDOMIZE_SEATING_EACH_GAME:
        rng.shuffle(lineup)

    return lineup


def build_game_from_lineup(lineup):
    """Create a fresh game while preserving any injected persistent agents."""
    player_names = [spec["name"] for spec in lineup]
    player_strategies = [spec["strategy"] for spec in lineup]
    player_qtables = [spec.get("agent_file") for spec in lineup]
    player_agents = [spec.get("agent") for spec in lineup]

    game = SushiGo(
        len(lineup),
        player_names,
        player_strategies,
        player_qtables=player_qtables,
        player_agents=player_agents,
    )

    # Make the intended training/evaluation mode explicit every episode.
    for spec, player in zip(lineup, game.players):
        if player.agent is not None:
            player.agent.train = spec.get("train", False)

    return game


def collect_trainable_agents(lineup):
    """Return any persistent agents that should be updated and saved."""
    agents = []
    for spec in lineup:
        agent = spec.get("agent")
        if agent is not None and spec.get("train", False):
            agents.append(agent)
    return agents


def save_agent(agent, final=False):
    """Save a trainable agent to the appropriate filename base."""
    if agent.__class__.__name__ == "QLearningAgent":
        print(f"[INFO] Saving Q-learning agent to {os.path.abspath(Q_TABLE_FILENAME)}.pkl")
        include_csv = config.params.get("q_table_save_csv_on_final", True) if final else config.params.get("q_table_save_csv", False)
        agent.save(Q_TABLE_FILENAME, include_csv=include_csv)
    elif agent.__class__.__name__ == "DeepQLearningAgent":
        print(f"[INFO] Saving Deep Q agent to {os.path.abspath(DQN_FILENAME)}.pkl")
        agent.save(DQN_FILENAME)


def format_lineup(lineup):
    """Human-readable lineup summary for logging sampled seat order."""
    return " | ".join(f"{spec['name']} ({spec['strategy']})" for spec in lineup)


def main():
    """Run training/evaluation across many sampled Sushi Go games."""
    if not 2 <= N_PLAYERS <= 5:
        raise ValueError("Sushi Go supports between 2 and 5 players.")

    rng = random.Random(RANDOM_SEED)
    n_games = config.params["iterations"]
    save_interval = config.params["save_every"]

    training_spec = build_persistent_training_spec(TRAINING_PLAYER)
    cached_opponent_pool = build_cached_opponent_pool(OPPONENT_POOL)
    trainable_agents = collect_trainable_agents([training_spec])

    # The training player appears every episode, so this per-game log is useful
    # for a learning curve even when seat order changes.
    training_score_log = []

    # Aggregate score distributions by participant name and by strategy. Opponents
    # may appear in only a subset of games, so we track counts separately.
    scores_by_name = defaultdict(list)
    scores_by_strategy = defaultdict(list)

    print("[INFO] Persistent training player:")
    print(f"  {training_spec['name']} ({training_spec['strategy']}), train={training_spec['train']}")
    print("[INFO] Opponent pool:")
    for opponent in cached_opponent_pool:
        suffix = " [cached]" if opponent.get("persistent", False) else ""
        print(f"  {opponent['name']} ({opponent['strategy']}){suffix}")

    if config.params.get("logging", False):
        stats_path = Path("dqn_stats.csv").resolve()
        print(f"[INFO] DQN diagnostics logging is ON. Stats will be written to {stats_path}")

    if RANDOM_SEED is not None:
        print(f"[INFO] Random seed: {RANDOM_SEED}")

    first_lineup = build_episode_lineup(training_spec, cached_opponent_pool, N_PLAYERS, rng)
    print(f"[INFO] Example sampled lineup: {format_lineup(first_lineup)}")

    for i in tqdm(range(n_games)):
        lineup = build_episode_lineup(training_spec, cached_opponent_pool, N_PLAYERS, rng)
        game = build_game_from_lineup(lineup)
        game_score = game.play_game()

        for spec, score in zip(lineup, game_score):
            scores_by_name[spec["name"]].append(score)
            scores_by_strategy[spec["strategy"]].append(score)
            if spec["name"] == training_spec["name"]:
                training_score_log.append(score)

        for agent in trainable_agents:
            agent.games_trained += 1

        if (i + 1) % save_interval == 0:
            for agent in trainable_agents:
                try:
                    save_agent(agent, final=False)
                except Exception as exc:
                    print(f"[ERROR] Failed to save agent {agent}: {exc}")

    for agent in trainable_agents:
        try:
            save_agent(agent, final=True)
        except Exception as exc:
            print(f"[ERROR] Failed final save for agent {agent}: {exc}")

    print()
    print("[INFO] Score summary by participant:")
    for name, raw_scores in scores_by_name.items():
        avg_score = round(sum(raw_scores) / len(raw_scores), 2)
        stdev_score = round(statistics.pstdev(raw_scores), 2)
        print(
            f"{name} played {len(raw_scores)} games, averaged {avg_score} points, "
            f"and had a score standard deviation of {stdev_score}."
        )

    print()
    print("[INFO] Aggregate score summary by strategy:")
    for strategy, raw_scores in scores_by_strategy.items():
        avg_score = round(sum(raw_scores) / len(raw_scores), 2)
        stdev_score = round(statistics.pstdev(raw_scores), 2)
        print(
            f"{strategy} appeared in {len(raw_scores)} seats, averaged {avg_score} points, "
            f"and had a score standard deviation of {stdev_score}."
        )

    # Plot only the persistent training player's scores. This remains meaningful
    # even when opponents and seat order vary from one episode to the next.
    plt.figure(figsize=(12, 6))
    raw_scores = np.array(training_score_log)
    window = max(1, int(len(raw_scores) / 100))
    smoothed = np.convolve(raw_scores, np.ones(window) / window, mode="valid")

    plt.plot(range(1, len(raw_scores) + 1), raw_scores, alpha=0.2, label=f"{training_spec['name']} (raw)")
    plt.plot(range(window, len(raw_scores) + 1), smoothed, label=f"{training_spec['name']} (avg)")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.title("Sushi Go Training Player Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sushi_go_training_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"[INFO] Plot saved as {filename}")


if __name__ == "__main__":
    main()
