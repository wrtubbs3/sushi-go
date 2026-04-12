"""Shared helpers for the Sushi Go user-facing Tkinter tools."""

from pathlib import Path

from players import Player


PLAYER_STRATEGIES = ["random", "sequential", "hierarchy", "q-learning", "deep q-learning"]
ADVISOR_STRATEGIES = ["none", "q-learning", "deep q-learning"]
REALTIME_STRATEGIES = ["q-learning", "deep q-learning"]
SETTINGS_PATH = Path(__file__).with_name("user_play_settings.json")


def resolve_agent_path(path_text):
    """Resolve a saved or user-entered agent path against the repo root when possible."""
    raw = (path_text or "").strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate.resolve()

    repo_candidate = (Path.cwd() / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return candidate.resolve()


def normalize_agent_path_for_save(path_text):
    """Store paths relative to the repo when possible so the setup file stays portable."""
    resolved = resolve_agent_path(path_text)
    if resolved is None:
        return ""

    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def scan_agent_files():
    """Return pickle files in the repo root for agent selection controls."""
    return sorted(path.name for path in Path.cwd().glob("*.pkl"))


def matching_agent_files(available_agent_files, strategy):
    """Return candidate checkpoint names that match the selected strategy."""
    if strategy == "q-learning":
        matches = [name for name in available_agent_files if "q_table" in name.lower()]
    elif strategy == "deep q-learning":
        matches = [name for name in available_agent_files if "dqn" in name.lower()]
    else:
        matches = []

    return matches


def build_inference_agent(strategy, agent_file):
    """Instantiate a non-training agent for advice or inference-only opponents."""
    stub = Player("Inference Agent", strategy, agent_file)
    if stub.agent is None:
        raise ValueError(f"Could not build an agent for strategy '{strategy}'.")
    stub.agent.train = False
    return stub.agent
