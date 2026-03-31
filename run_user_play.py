"""Interactive Sushi Go runner with a simple Tkinter GUI for manual play."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import config
from game import SushiGo, card_action_name, card_label
from players import Player
from state_action_reward import build_actions_dict, build_state_dict


PLAYER_STRATEGIES = ["random", "sequential", "hierarchy", "q-learning", "deep q-learning"]
ADVISOR_STRATEGIES = ["none", "q-learning", "deep q-learning"]
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


def build_inference_agent(strategy, agent_file):
    """Instantiate a non-training agent for advice or inference-only opponents."""
    stub = Player("Inference Agent", strategy, agent_file)
    if stub.agent is None:
        raise ValueError(f"Could not build an agent for strategy '{strategy}'.")
    stub.agent.train = False
    return stub.agent


class UserPlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sushi Go User Play")
        self.root.geometry("1280x900")

        # User-play mode is always inference-only.
        config.params["logging"] = False

        self.game = None
        self.human_index = 0
        self.setup_frame = None
        self.game_frame = None

        self.saved_settings = self.load_settings()
        self.available_agent_files = self.scan_agent_files()

        self.n_players_var = tk.IntVar(value=self.saved_settings.get("n_players", 4))
        self.human_name_var = tk.StringVar(value=self.saved_settings.get("human_name", "You"))
        self.advisor_strategy_var = tk.StringVar(value=self.saved_settings.get("advisor_strategy", "none"))
        self.advisor_file_var = tk.StringVar(value=self.saved_settings.get("advisor_agent_file", ""))

        self.opponent_rows = []

        self.status_var = tk.StringVar(value="Choose a setup to begin.")
        self.recommendation_var = tk.StringVar(value="No recommendation available yet.")

        self.show_setup_screen()

    def load_settings(self):
        if not SETTINGS_PATH.exists():
            return {}

        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def save_settings(self, data):
        with open(SETTINGS_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        self.saved_settings = data

    def scan_agent_files(self):
        return sorted(path.name for path in Path.cwd().glob("*.pkl"))

    def matching_agent_files(self, strategy):
        if strategy == "q-learning":
            matches = [name for name in self.available_agent_files if "q_table" in name.lower()]
        elif strategy == "deep q-learning":
            matches = [name for name in self.available_agent_files if "dqn" in name.lower()]
        else:
            matches = []

        return matches

    def show_setup_screen(self):
        if self.setup_frame is not None:
            self.setup_frame.destroy()

        if self.game_frame is not None:
            self.game_frame.destroy()
            self.game_frame = None

        self.available_agent_files = self.scan_agent_files()
        self.setup_frame = ttk.Frame(self.root, padding=16)
        self.setup_frame.pack(fill="both", expand=True)

        title = ttk.Label(self.setup_frame, text="Sushi Go User Play", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            self.setup_frame,
            text=(
                "Set the number of players, choose each opponent's strategy, and optionally select a "
                "Q-learning or DQN checkpoint to act as your advisor."
            ),
            wraplength=1000,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(6, 16))

        general_frame = ttk.Frame(self.setup_frame)
        general_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(general_frame, text="Players").grid(row=0, column=0, sticky="w", padx=(0, 8))
        player_spinbox = tk.Spinbox(
            general_frame,
            from_=2,
            to=5,
            textvariable=self.n_players_var,
            width=5,
            command=self.update_visible_player_rows,
        )
        player_spinbox.grid(row=0, column=1, sticky="w")

        ttk.Button(general_frame, text="Refresh Agent Files", command=self.refresh_agent_files).grid(
            row=0, column=2, sticky="w", padx=(16, 0)
        )

        human_frame = ttk.LabelFrame(self.setup_frame, text="Your Seat", padding=12)
        human_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(human_frame, text="Name").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(human_frame, textvariable=self.human_name_var, width=18).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(human_frame, text="Advisor Strategy").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        advisor_combo = ttk.Combobox(
            human_frame,
            textvariable=self.advisor_strategy_var,
            values=ADVISOR_STRATEGIES,
            width=18,
            state="readonly",
        )
        advisor_combo.grid(row=0, column=3, sticky="w", pady=4)
        advisor_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_advisor_controls())

        ttk.Label(human_frame, text="Advisor Agent File").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.advisor_file_combo = ttk.Combobox(human_frame, textvariable=self.advisor_file_var, width=60)
        self.advisor_file_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=4)
        self.advisor_file_combo.bind("<<ComboboxSelected>>", lambda _event: self.maybe_browse_agent_file(self.advisor_file_var))
        self.advisor_browse_button = ttk.Button(
            human_frame,
            text="Browse...",
            command=lambda: self.browse_agent_file(self.advisor_file_var),
        )
        self.advisor_browse_button.grid(row=1, column=4, sticky="w", padx=(8, 0), pady=4)
        human_frame.columnconfigure(3, weight=1)

        opponents_frame = ttk.LabelFrame(self.setup_frame, text="Opponents", padding=12)
        opponents_frame.pack(fill="both", expand=True, pady=(0, 12))

        saved_opponents = self.saved_settings.get("opponents", [])
        self.opponent_rows = []
        for index in range(4):
            seat_number = index + 2
            saved_row = saved_opponents[index] if index < len(saved_opponents) else {}
            row_frame = ttk.Frame(opponents_frame)
            row_frame.grid(row=index, column=0, sticky="ew", pady=6)
            opponents_frame.columnconfigure(0, weight=1)

            ttk.Label(row_frame, text=f"Player {seat_number}").grid(row=0, column=0, sticky="w", padx=(0, 8))

            name_var = tk.StringVar(value=saved_row.get("name", f"Player {seat_number}"))
            ttk.Entry(row_frame, textvariable=name_var, width=16).grid(row=0, column=1, sticky="w", padx=(0, 12))

            strategy_var = tk.StringVar(value=saved_row.get("strategy", "random"))
            strategy_combo = ttk.Combobox(
                row_frame,
                textvariable=strategy_var,
                values=PLAYER_STRATEGIES,
                width=18,
                state="readonly",
            )
            strategy_combo.grid(row=0, column=2, sticky="w", padx=(0, 12))

            agent_file_var = tk.StringVar(value=saved_row.get("agent_file", ""))
            agent_file_combo = ttk.Combobox(row_frame, textvariable=agent_file_var, width=55)
            agent_file_combo.grid(row=0, column=3, sticky="ew", padx=(0, 8))
            agent_file_combo.bind("<<ComboboxSelected>>", lambda _event, var=agent_file_var: self.maybe_browse_agent_file(var))

            browse_button = ttk.Button(
                row_frame,
                text="Browse...",
                command=lambda var=agent_file_var: self.browse_agent_file(var),
            )
            browse_button.grid(row=0, column=4, sticky="w")
            row_frame.columnconfigure(3, weight=1)

            row = {
                "frame": row_frame,
                "name_var": name_var,
                "strategy_var": strategy_var,
                "strategy_combo": strategy_combo,
                "agent_file_var": agent_file_var,
                "agent_file_combo": agent_file_combo,
                "browse_button": browse_button,
            }
            strategy_combo.bind("<<ComboboxSelected>>", lambda _event, current_row=row: self.update_player_row_controls(current_row))
            self.opponent_rows.append(row)

        buttons_frame = ttk.Frame(self.setup_frame)
        buttons_frame.pack(fill="x")

        ttk.Button(buttons_frame, text="Start Game", command=self.start_game).pack(side="left")
        ttk.Button(buttons_frame, text="Quit", command=self.root.destroy).pack(side="right")

        self.update_advisor_controls()
        self.update_visible_player_rows()

    def refresh_agent_files(self):
        self.available_agent_files = self.scan_agent_files()
        self.update_advisor_controls()
        for row in self.opponent_rows:
            self.update_player_row_controls(row)

    def set_combo_values(self, combo, strategy, current_value):
        values = self.matching_agent_files(strategy)
        if current_value and current_value not in values:
            values = values + [current_value]
        combo["values"] = values + (["Browse..."] if strategy in {"q-learning", "deep q-learning"} else [])

    def update_advisor_controls(self):
        strategy = self.advisor_strategy_var.get()
        self.set_combo_values(self.advisor_file_combo, strategy, self.advisor_file_var.get())

        enabled = strategy in {"q-learning", "deep q-learning"}
        state = "normal" if enabled else "disabled"
        self.advisor_file_combo.configure(state=state)
        self.advisor_browse_button.configure(state=state)

    def update_player_row_controls(self, row):
        strategy = row["strategy_var"].get()
        self.set_combo_values(row["agent_file_combo"], strategy, row["agent_file_var"].get())

        enabled = strategy in {"q-learning", "deep q-learning"}
        state = "normal" if enabled else "disabled"
        row["agent_file_combo"].configure(state=state)
        row["browse_button"].configure(state=state)

    def update_visible_player_rows(self):
        n_players = max(2, min(5, self.n_players_var.get()))
        self.n_players_var.set(n_players)

        for index, row in enumerate(self.opponent_rows):
            visible = index < (n_players - 1)
            if visible:
                row["frame"].grid()
                self.update_player_row_controls(row)
            else:
                row["frame"].grid_remove()

    def browse_agent_file(self, target_var):
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose agent checkpoint",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if selected:
            target_var.set(normalize_agent_path_for_save(selected))

    def maybe_browse_agent_file(self, target_var):
        if target_var.get() == "Browse...":
            self.browse_agent_file(target_var)

    def build_saved_settings_payload(self):
        n_players = self.n_players_var.get()
        return {
            "n_players": n_players,
            "human_name": self.human_name_var.get().strip() or "You",
            "advisor_strategy": self.advisor_strategy_var.get(),
            "advisor_agent_file": normalize_agent_path_for_save(self.advisor_file_var.get()),
            "opponents": [
                {
                    "name": row["name_var"].get().strip() or f"Player {index + 2}",
                    "strategy": row["strategy_var"].get(),
                    "agent_file": normalize_agent_path_for_save(row["agent_file_var"].get()),
                }
                for index, row in enumerate(self.opponent_rows[: n_players - 1])
            ],
        }

    def validate_learned_agent_choice(self, strategy, file_value, owner_label):
        if strategy not in {"q-learning", "deep q-learning"}:
            return None

        if not file_value.strip():
            raise ValueError(f"{owner_label} uses {strategy}, so an agent file is required.")

        resolved = resolve_agent_path(file_value)
        if resolved is None or not resolved.exists():
            raise ValueError(f"{owner_label} agent file was not found: {file_value}")

        return str(resolved)

    def start_game(self):
        try:
            settings = self.build_saved_settings_payload()
            self.save_settings(settings)

            n_players = settings["n_players"]
            human_name = settings["human_name"]
            advisor_strategy = settings["advisor_strategy"]
            advisor_file = settings["advisor_agent_file"]

            player_names = [human_name]
            player_strategies = ["user choice"]
            player_qtables = [None]
            player_agents = [None]

            if advisor_strategy in {"q-learning", "deep q-learning"}:
                advisor_agent_path = self.validate_learned_agent_choice(advisor_strategy, advisor_file, "Your advisor")
                player_agents[0] = build_inference_agent(advisor_strategy, advisor_agent_path)

            for index, opponent in enumerate(settings["opponents"], start=2):
                strategy = opponent["strategy"]
                agent_file = opponent["agent_file"]

                if strategy in {"q-learning", "deep q-learning"}:
                    agent_file = self.validate_learned_agent_choice(strategy, agent_file, opponent["name"])

                player_names.append(opponent["name"])
                player_strategies.append(strategy)
                player_qtables.append(agent_file)
                player_agents.append(None)

            self.game = SushiGo(
                n_players,
                player_names,
                player_strategies,
                player_qtables=player_qtables,
                player_agents=player_agents,
            )

            for player in self.game.players:
                if player.agent is not None:
                    player.agent.train = False

            self.setup_frame.destroy()
            self.setup_frame = None
            self.show_game_screen()
            self.game.start_game()
            self.refresh_board()

        except Exception as exc:
            messagebox.showerror("Cannot start game", str(exc), parent=self.root)

    def show_game_screen(self):
        self.game_frame = ttk.Frame(self.root, padding=16)
        self.game_frame.pack(fill="both", expand=True)

        header_frame = ttk.Frame(self.game_frame)
        header_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(header_frame, textvariable=self.status_var, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(header_frame, textvariable=self.recommendation_var, justify="left", wraplength=1100).pack(
            anchor="w",
            pady=(6, 0),
        )

        ttk.Button(header_frame, text="Return to Setup", command=self.return_to_setup).pack(anchor="e", pady=(8, 0))

        self.opponents_frame = ttk.Frame(self.game_frame)
        self.opponents_frame.pack(fill="both", expand=True)

        self.human_frame = ttk.LabelFrame(self.game_frame, text="Your Area", padding=12)
        self.human_frame.pack(fill="x", pady=(12, 0))

        self.human_table_frame = ttk.Frame(self.human_frame)
        self.human_table_frame.pack(fill="x", pady=(0, 10))

        self.human_hand_frame = ttk.Frame(self.human_frame)
        self.human_hand_frame.pack(fill="x")

    def return_to_setup(self):
        if self.game is not None and not self.game.game_finished:
            confirmed = messagebox.askyesno(
                "Leave current game?",
                "Return to setup and abandon the current game?",
                parent=self.root,
            )
            if not confirmed:
                return

        self.game = None
        self.status_var.set("Choose a setup to begin.")
        self.recommendation_var.set("No recommendation available yet.")
        self.show_setup_screen()

    def clear_children(self, frame):
        for child in frame.winfo_children():
            child.destroy()

    def compute_human_advice(self):
        player = self.game.players[self.human_index]
        state_dict = build_state_dict(player.cards_in_hand, player.cards_on_table)
        actions_dict = build_actions_dict(state_dict)

        if player.agent is None:
            return {
                "summary": "No advisor configured. Click any card in your hand to play it.",
                "per_card": [
                    {
                        "index": index,
                        "card_label": card_label(card),
                        "q_value": None,
                        "action": card_action_name(card),
                        "recommended": False,
                    }
                    for index, card in enumerate(player.cards_in_hand)
                ],
            }

        recommended_action, action_values = player.agent.recommend_action(state_dict, actions_dict)
        per_card = []
        recommended_labels = []
        for index, card in enumerate(player.cards_in_hand):
            action = card_action_name(card)
            q_value = action_values.get(action)
            recommended = action == recommended_action and q_value is not None
            if recommended:
                recommended_labels.append(f"{index}: {card_label(card)}")
            per_card.append(
                {
                    "index": index,
                    "card_label": card_label(card),
                    "q_value": q_value,
                    "action": action,
                    "recommended": recommended,
                }
            )

        summary_lines = []
        advisor_type = player.agent.__class__.__name__.replace("LearningAgent", "")
        summary_lines.append(f"Advisor: {advisor_type or 'Agent'}")
        if recommended_action is not None:
            summary_lines.append(f"Recommended action: {recommended_action}")
        if recommended_labels:
            summary_lines.append("Recommended card(s): " + ", ".join(recommended_labels))
        if action_values:
            details = ", ".join(f"{action}={value:.3f}" for action, value in action_values.items())
            summary_lines.append("Legal action Q-values: " + details)

        return {
            "summary": "\n".join(summary_lines),
            "per_card": per_card,
        }

    def render_cards(self, parent, cards, face_down=False):
        if not cards:
            ttk.Label(parent, text="None").pack(anchor="w")
            return

        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill="x")
        for index, card in enumerate(cards):
            text = f"Face Down {index + 1}" if face_down else card_label(card)
            row = index // 5
            column = index % 5
            tk.Label(
                cards_frame,
                text=text,
                relief="ridge",
                borderwidth=1,
                padx=8,
                pady=6,
                width=14,
            ).grid(row=row, column=column, padx=3, pady=3, sticky="w")

    def refresh_board(self):
        if self.game is None:
            return

        self.clear_children(self.opponents_frame)
        self.clear_children(self.human_table_frame)
        self.clear_children(self.human_hand_frame)

        self.status_var.set(
            f"Hand {self.game.current_hand_number}/3 | Pick {self.game.current_turn_number + 1}/{self.game.n_cards_dealt_per_player}"
        )

        advice = self.compute_human_advice()
        self.recommendation_var.set(advice["summary"])

        opponent_players = [player for index, player in enumerate(self.game.players) if index != self.human_index]
        for index, player in enumerate(opponent_players):
            panel = ttk.LabelFrame(
                self.opponents_frame,
                text=f"{player.name} | Score {player.points} | {player.strategy}",
                padding=10,
            )
            row = index // 2
            column = index % 2
            panel.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
            self.opponents_frame.columnconfigure(column, weight=1)

            ttk.Label(panel, text="Hand").pack(anchor="w")
            self.render_cards(panel, player.cards_in_hand, face_down=True)

            ttk.Label(panel, text="Table").pack(anchor="w", pady=(8, 0))
            self.render_cards(panel, player.cards_on_table, face_down=False)

        human_player = self.game.players[self.human_index]
        self.human_frame.configure(text=f"{human_player.name} | Score {human_player.points} | Manual Play")

        ttk.Label(self.human_table_frame, text="Your Table").pack(anchor="w")
        self.render_cards(self.human_table_frame, human_player.cards_on_table, face_down=False)

        ttk.Label(self.human_hand_frame, text="Your Hand").pack(anchor="w")
        hand_buttons_frame = ttk.Frame(self.human_hand_frame)
        hand_buttons_frame.pack(fill="x", pady=(6, 0))

        for card_info in advice["per_card"]:
            q_text = "Q=n/a" if card_info["q_value"] is None else f"Q={card_info['q_value']:.3f}"
            button_text = f"{card_info['index']}: {card_info['card_label']}\n{q_text}"
            bg_color = "#d8f0d2" if card_info["recommended"] else "#f8f8f8"
            row = card_info["index"] // 5
            column = card_info["index"] % 5
            button = tk.Button(
                hand_buttons_frame,
                text=button_text,
                command=lambda idx=card_info["index"]: self.play_human_card(idx),
                relief="raised",
                borderwidth=2,
                padx=8,
                pady=8,
                width=16,
                height=3,
                bg=bg_color,
                wraplength=120,
            )
            button.grid(row=row, column=column, padx=4, pady=4, sticky="w")

    def show_modal_dialog(self, title, message, button_text):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(frame, text=message, justify="left", wraplength=520).pack(anchor="w", pady=(10, 14))
        ttk.Button(frame, text=button_text, command=dialog.destroy).pack(anchor="e")

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)

    def format_hand_summary(self, hand_summary, button_text):
        lines = [f"Hand {hand_summary['hand_number']} complete."]
        lines.append("")
        for player_summary in hand_summary["players"]:
            lines.append(
                f"{player_summary['name']}: {player_summary['hand_points']:+d} this hand, "
                f"{player_summary['total_points']} total"
            )
        self.show_modal_dialog(f"Hand {hand_summary['hand_number']} Results", "\n".join(lines), button_text)

    def format_game_summary(self, game_summary):
        lines = ["Final Scores:"]
        lines.append("")
        for player_summary in game_summary["players"]:
            lines.append(f"{player_summary['name']}: {player_summary['points']}")
        lines.append("")
        if game_summary["is_tie"]:
            winner_text = ", ".join(game_summary["winner_names"])
            lines.append(f"It's a tie between: {winner_text}")
        else:
            lines.append(f"Winner: {game_summary['winner_names'][0]}")
        self.show_modal_dialog("Game Over", "\n".join(lines), "Return to Setup")

    def play_human_card(self, card_index):
        if self.game is None or self.game.game_finished:
            return

        try:
            turn_summary = self.game.play_turn({self.human_index: card_index})
        except Exception as exc:
            messagebox.showerror("Turn failed", str(exc), parent=self.root)
            return

        if turn_summary["hand_complete"]:
            button_text = "Show Final Scores" if turn_summary["game_complete"] else "Continue to Next Hand"
            self.format_hand_summary(turn_summary["hand_summary"], button_text)

            if turn_summary["game_complete"]:
                self.format_game_summary(turn_summary["game_summary"])
                self.return_to_setup_after_game()
                return

            self.game.advance_to_next_hand()

        self.refresh_board()

    def return_to_setup_after_game(self):
        self.game = None
        if self.game_frame is not None:
            self.game_frame.destroy()
            self.game_frame = None
        self.show_setup_screen()


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    UserPlayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
