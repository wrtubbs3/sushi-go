"""Separate real-time strategy helper UI for physical Sushi Go play."""

from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from cards import Card
from game import card_action_name, card_label
from state_action_reward import build_actions_dict, build_state_dict
from user_play_common import (
    REALTIME_STRATEGIES,
    build_inference_agent,
    matching_agent_files,
    normalize_agent_path_for_save,
    resolve_agent_path,
    scan_agent_files,
)


CARD_SPECS = [
    {"key": "wasabi", "label": "Wasabi", "type": "wasabi", "subtype": "n/a", "max_count": 6},
    {"key": "nigiri_1", "label": "Nigiri-1", "type": "nigiri", "subtype": 1, "max_count": 5},
    {"key": "nigiri_2", "label": "Nigiri-2", "type": "nigiri", "subtype": 2, "max_count": 10},
    {"key": "nigiri_3", "label": "Nigiri-3", "type": "nigiri", "subtype": 3, "max_count": 5},
    {"key": "tempura", "label": "Tempura", "type": "tempura", "subtype": "n/a", "max_count": 14},
    {"key": "sashimi", "label": "Sashimi", "type": "sashimi", "subtype": "n/a", "max_count": 14},
    {"key": "dumpling", "label": "Dumpling", "type": "dumpling", "subtype": "n/a", "max_count": 14},
    {"key": "pudding", "label": "Pudding", "type": "pudding", "subtype": "n/a", "max_count": 10},
    {"key": "maki_1", "label": "Maki-1", "type": "maki", "subtype": 1, "max_count": 6},
    {"key": "maki_2", "label": "Maki-2", "type": "maki", "subtype": 2, "max_count": 12},
    {"key": "maki_3", "label": "Maki-3", "type": "maki", "subtype": 3, "max_count": 8},
    {"key": "chopsticks", "label": "Chopsticks", "type": "chopsticks", "subtype": "n/a", "max_count": 4},
]

CARD_SPEC_BY_KEY = {spec["key"]: spec for spec in CARD_SPECS}


def create_card_from_key(card_key):
    """Build a concrete Card object from a validated catalog key."""
    spec = CARD_SPEC_BY_KEY[card_key]
    return Card(spec["type"], spec["subtype"])


def card_to_key(card):
    """Map a Card object back to the catalog key used by the count editors."""
    if card.type in {"nigiri", "maki"}:
        return f"{card.type}_{card.subtype}"
    return card.type


def cards_to_counts(cards):
    """Summarize a list of Card objects into the UI count structure."""
    counts = {spec["key"]: 0 for spec in CARD_SPECS}
    for card in cards:
        counts[card_to_key(card)] += 1
    return counts


def dealt_cards_per_player(n_players):
    """Return the standard Sushi Go opening hand size for the player count."""
    return {
        2: 10,
        3: 9,
        4: 8,
        5: 7,
    }.get(n_players)


class RealTimeStrategyMode:
    def __init__(self, root, saved_settings, save_settings_callback, on_back):
        self.root = root
        self.saved_settings = saved_settings or {}
        self.save_settings_callback = save_settings_callback
        self.on_back = on_back

        self.frame = None
        self.mode_content_frame = None
        self.agent = None
        self.last_analysis = None
        self.available_agent_files = scan_agent_files()

        self.n_players_var = tk.IntVar(value=self.saved_settings.get("n_players", 4))
        self.strategy_var = tk.StringVar(value=self.saved_settings.get("strategy", "deep q-learning"))
        self.agent_file_var = tk.StringVar(value=self.saved_settings.get("agent_file", ""))
        self.status_var = tk.StringVar(value="Load a strategy, enter your current hand, then analyze.")
        self.loaded_agent_var = tk.StringVar(value="No strategy loaded.")
        self.hand_hint_var = tk.StringVar(value="")

        self.hand_count_vars = {spec["key"]: tk.IntVar(value=0) for spec in CARD_SPECS}
        self.table_count_vars = {spec["key"]: tk.IntVar(value=0) for spec in CARD_SPECS}

        for var in list(self.hand_count_vars.values()) + list(self.table_count_vars.values()):
            var.trace_add("write", self.on_card_counts_changed)

        self.agent_file_combo = None
        self.agent_browse_button = None
        self.editor_frame = None
        self.advice_frame = None

        self.update_hand_hint()

    def show(self):
        self.available_agent_files = scan_agent_files()
        self.frame = ttk.Frame(self.root, padding=16)
        self.frame.pack(fill="both", expand=True)

        header_row = ttk.Frame(self.frame)
        header_row.pack(fill="x")
        ttk.Label(header_row, text="Sushi Go Real Time Strategy", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Button(header_row, text="Quit", command=self.root.destroy).pack(side="right")
        ttk.Button(header_row, text="Back to Modes", command=self.on_back).pack(side="right")

        ttk.Label(
            self.frame,
            text=(
                "Use this mode alongside a physical game. Load an agent, enter the cards currently in your hand "
                "and on your table, inspect per-card Q-values, then click the card you actually decide to play."
            ),
            wraplength=1080,
            justify="left",
        ).pack(anchor="w", pady=(6, 14))

        strategy_frame = ttk.LabelFrame(self.frame, text="Strategy Setup", padding=12)
        strategy_frame.pack(fill="x", pady=(0, 12))
        strategy_frame.columnconfigure(3, weight=1)

        ttk.Label(strategy_frame, text="Players").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        player_spinbox = tk.Spinbox(
            strategy_frame,
            from_=2,
            to=5,
            textvariable=self.n_players_var,
            width=5,
            command=self.update_hand_hint,
        )
        player_spinbox.grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(strategy_frame, text="Strategy").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=self.strategy_var,
            values=REALTIME_STRATEGIES,
            width=18,
            state="readonly",
        )
        strategy_combo.grid(row=0, column=3, sticky="w", pady=4)
        strategy_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_agent_controls())

        ttk.Button(strategy_frame, text="Refresh Agent Files", command=self.refresh_agent_files).grid(
            row=0,
            column=4,
            sticky="w",
            padx=(16, 0),
            pady=4,
        )

        ttk.Label(strategy_frame, text="Agent File").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.agent_file_combo = ttk.Combobox(strategy_frame, textvariable=self.agent_file_var, width=70)
        self.agent_file_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=4)
        self.agent_file_combo.bind("<<ComboboxSelected>>", lambda _event: self.maybe_browse_agent_file())
        self.agent_browse_button = ttk.Button(strategy_frame, text="Browse...", command=self.browse_agent_file)
        self.agent_browse_button.grid(row=1, column=4, sticky="w", padx=(8, 0), pady=4)

        ttk.Button(strategy_frame, text="Load Strategy", command=self.load_strategy).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(8, 4),
        )
        ttk.Label(strategy_frame, textvariable=self.loaded_agent_var, wraplength=840, justify="left").grid(
            row=2,
            column=1,
            columnspan=4,
            sticky="w",
            pady=(8, 4),
        )
        ttk.Label(strategy_frame, textvariable=self.hand_hint_var).grid(
            row=3,
            column=0,
            columnspan=5,
            sticky="w",
            pady=(4, 0),
        )

        status_frame = ttk.Frame(self.frame)
        status_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=1080,
            justify="left",
        ).pack(anchor="w")

        self.mode_content_frame = ttk.Frame(self.frame)
        self.mode_content_frame.pack(fill="both", expand=True)

        self.show_editor_view()
        self.update_agent_controls()

    def destroy(self):
        if self.frame is not None:
            self.frame.destroy()
            self.frame = None
            self.mode_content_frame = None
            self.editor_frame = None
            self.advice_frame = None

    def clear_mode_content(self):
        if self.mode_content_frame is None:
            return
        for child in self.mode_content_frame.winfo_children():
            child.destroy()
        self.editor_frame = None
        self.advice_frame = None

    def show_editor_view(self):
        self.clear_mode_content()
        self.editor_frame = ttk.Frame(self.mode_content_frame)
        self.editor_frame.pack(fill="both", expand=True)

        editors_frame = ttk.Frame(self.editor_frame)
        editors_frame.pack(fill="both", expand=True)
        editors_frame.columnconfigure(0, weight=1)
        editors_frame.columnconfigure(1, weight=1)

        hand_frame = ttk.LabelFrame(editors_frame, text="Your Hand", padding=12)
        hand_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(
            hand_frame,
            text="Enter the cards currently in your hand. After you record a play, this section resets to blank.",
            wraplength=500,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        self.build_card_editor(hand_frame, self.hand_count_vars)

        table_frame = ttk.LabelFrame(editors_frame, text="Your Table", padding=12)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(
            table_frame,
            text=(
                "Keep your table state here. When you click a play in the advice view, the chosen card is added "
                "to this table by default for the next decision."
            ),
            wraplength=500,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))
        self.build_card_editor(table_frame, self.table_count_vars)

        controls_frame = ttk.Frame(self.editor_frame)
        controls_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(controls_frame, text="Analyze Hand", command=self.analyze_current_state).pack(side="left")
        ttk.Button(controls_frame, text="Clear Hand", command=self.clear_hand).pack(side="left", padx=(8, 0))
        ttk.Button(controls_frame, text="Start Next Hand", command=self.start_next_hand).pack(side="left", padx=(8, 0))
        ttk.Button(controls_frame, text="Clear Table", command=self.clear_table).pack(side="left", padx=(8, 0))

    def show_advice_view(self):
        self.clear_mode_content()
        self.advice_frame = ttk.Frame(self.mode_content_frame)
        self.advice_frame.pack(fill="both", expand=True)

        header_row = ttk.Frame(self.advice_frame)
        header_row.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header_row,
            text="Card Advice",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        ttk.Button(header_row, text="Back to Edit", command=self.return_to_editor).pack(side="right")

        ttk.Label(
            self.advice_frame,
            text="Click a card tile to record it as your actual play.",
        ).pack(anchor="w", pady=(0, 8))

        table_frame = ttk.LabelFrame(self.advice_frame, text="Cards On Table", padding=8)
        table_frame.pack(fill="x", pady=(0, 10))
        self.render_card_tiles(table_frame, self.last_analysis["table_cards"], face_down=False, max_columns=8, tile_width=10)

        hand_frame = ttk.LabelFrame(self.advice_frame, text="Cards In Hand", padding=8)
        hand_frame.pack(fill="x")
        buttons_frame = ttk.Frame(hand_frame)
        buttons_frame.pack(fill="x")

        q_values = [card_info["q_value"] for card_info in self.last_analysis["per_card"] if card_info["q_value"] is not None]
        best_q_value = max(q_values) if q_values else None

        for card_info in self.last_analysis["per_card"]:
            q_text = "Q=n/a" if card_info["q_value"] is None else f"Q={card_info['q_value']:.3f}"
            button_text = f"{card_info['index'] + 1}: {card_info['card_label']}\n{q_text}"
            is_best_card = best_q_value is not None and card_info["q_value"] == best_q_value
            bg_color = "#d8f0d2" if is_best_card else "#f8f8f8"
            row = card_info["index"] // 6
            column = card_info["index"] % 6
            tk.Button(
                buttons_frame,
                text=button_text,
                command=lambda idx=card_info["index"]: self.play_selected_card(idx),
                relief="raised",
                borderwidth=2,
                padx=6,
                pady=6,
                width=14,
                height=2,
                bg=bg_color,
                wraplength=120,
            ).grid(row=row, column=column, padx=4, pady=4, sticky="w")

    def return_to_editor(self):
        self.show_editor_view()

    def build_card_editor(self, parent, variable_map):
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        for row_index, spec in enumerate(CARD_SPECS):
            ttk.Label(grid, text=f"{spec['label']} (max {spec['max_count']})").grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=3,
            )
            tk.Spinbox(
                grid,
                from_=0,
                to=spec["max_count"],
                textvariable=variable_map[spec["key"]],
                width=6,
            ).grid(row=row_index, column=1, sticky="w", pady=3)

    def refresh_agent_files(self):
        self.available_agent_files = scan_agent_files()
        self.update_agent_controls()

    def update_agent_controls(self):
        strategy = self.strategy_var.get()
        current_value = self.agent_file_var.get()
        values = matching_agent_files(self.available_agent_files, strategy)
        if current_value and current_value not in values:
            values = values + [current_value]

        self.agent_file_combo["values"] = values + ["Browse..."]
        self.agent_file_combo.configure(state="normal")
        self.agent_browse_button.configure(state="normal")

    def browse_agent_file(self):
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose agent checkpoint",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=".",
        )
        if selected:
            self.agent_file_var.set(normalize_agent_path_for_save(selected))

    def maybe_browse_agent_file(self):
        if self.agent_file_var.get() == "Browse...":
            self.browse_agent_file()

    def update_hand_hint(self, *_args):
        n_players = max(2, min(5, self.safe_get_int(self.n_players_var, 4)))
        self.n_players_var.set(n_players)
        dealt_cards = dealt_cards_per_player(n_players)
        self.hand_hint_var.set(
            f"Reference: a {n_players}-player hand uses {dealt_cards} total cards in your current hand plus your current table."
        )

    def validate_state_size(self, hand_cards, table_cards):
        """Ensure the entered state matches the draft size for the selected player count."""
        n_players = max(2, min(5, self.safe_get_int(self.n_players_var, 4)))
        expected_total = dealt_cards_per_player(n_players)
        actual_total = len(hand_cards) + len(table_cards)
        if actual_total != expected_total:
            raise ValueError(
                f"For a {n_players}-player game, hand + table must total {expected_total} cards. "
                f"You entered {len(hand_cards)} in hand and {len(table_cards)} on table ({actual_total} total)."
            )

    def build_saved_settings_payload(self):
        return {
            "n_players": max(2, min(5, self.safe_get_int(self.n_players_var, 4))),
            "strategy": self.strategy_var.get(),
            "agent_file": normalize_agent_path_for_save(self.agent_file_var.get()),
        }

    def validate_agent_choice(self):
        strategy = self.strategy_var.get()
        if strategy not in REALTIME_STRATEGIES:
            raise ValueError("Choose either q-learning or deep q-learning.")

        file_value = self.agent_file_var.get().strip()
        if not file_value:
            raise ValueError("Choose a .pkl agent file before loading the strategy.")

        resolved = resolve_agent_path(file_value)
        if resolved is None or not resolved.exists():
            raise ValueError(f"Agent file was not found: {file_value}")

        return strategy, str(resolved)

    def load_strategy(self):
        try:
            strategy, agent_path = self.validate_agent_choice()
            self.agent = build_inference_agent(strategy, agent_path)
            self.save_settings_callback("real_time_strategy", self.build_saved_settings_payload())
            self.loaded_agent_var.set(f"Loaded {strategy} from {agent_path}")
            self.status_var.set("Strategy loaded. Enter your hand and analyze when ready.")
        except Exception as exc:
            messagebox.showerror("Cannot load strategy", str(exc), parent=self.root)

    def safe_get_int(self, variable, fallback=0):
        try:
            return int(variable.get())
        except (tk.TclError, ValueError):
            return fallback

    def build_cards_from_editor(self, variable_map, owner_label):
        cards = []
        for spec in CARD_SPECS:
            count = self.safe_get_int(variable_map[spec["key"]], 0)
            if count < 0:
                raise ValueError(f"{owner_label} cannot contain a negative count for {spec['label']}.")
            if count > spec["max_count"]:
                raise ValueError(
                    f"{owner_label} cannot contain {count} copies of {spec['label']} "
                    f"(deck maximum is {spec['max_count']})."
                )
            cards.extend(create_card_from_key(spec["key"]) for _ in range(count))

        return cards

    def set_card_editor_counts(self, variable_map, cards):
        counts = cards_to_counts(cards)
        for spec in CARD_SPECS:
            variable_map[spec["key"]].set(counts[spec["key"]])

    def clear_card_editor(self, variable_map):
        for spec in CARD_SPECS:
            variable_map[spec["key"]].set(0)

    def on_card_counts_changed(self, *_args):
        self.last_analysis = None

    def analyze_current_state(self):
        if self.agent is None:
            messagebox.showerror("Strategy not loaded", "Load a strategy before analyzing a hand.", parent=self.root)
            return

        try:
            hand_cards = self.build_cards_from_editor(self.hand_count_vars, "Your hand")
            table_cards = self.build_cards_from_editor(self.table_count_vars, "Your table")
            self.validate_state_size(hand_cards, table_cards)
        except Exception as exc:
            messagebox.showerror("Invalid card entry", str(exc), parent=self.root)
            return

        if not hand_cards:
            messagebox.showerror("No hand entered", "Enter at least one card in your hand before analyzing.", parent=self.root)
            return

        state_dict = build_state_dict(hand_cards, table_cards)
        actions_dict = build_actions_dict(state_dict)
        recommended_action, action_values = self.agent.recommend_action(state_dict, actions_dict)

        per_card = []
        for index, card in enumerate(hand_cards):
            action = card_action_name(card)
            q_value = action_values.get(action)
            per_card.append(
                {
                    "index": index,
                    "card_label": card_label(card),
                    "q_value": q_value,
                    "recommended": action == recommended_action and q_value is not None,
                }
            )

        self.last_analysis = {
            "hand_cards": hand_cards,
            "table_cards": table_cards,
            "per_card": per_card,
        }
        self.status_var.set(
            "Q-values are shown below for each card in your hand. Click the card you decide to play to "
            "carry it onto the table and reset the hand editor for the next pass."
        )
        self.show_advice_view()

    def render_card_tiles(self, parent, cards, face_down=False, max_columns=8, tile_width=12):
        """Render static card tiles similar to the main user-play board."""
        if not cards:
            ttk.Label(parent, text="None").pack(anchor="w")
            return

        cards_frame = ttk.Frame(parent)
        cards_frame.pack(fill="x")
        for index, card in enumerate(cards):
            text = "Back" if face_down else card_label(card)
            row = index // max_columns
            column = index % max_columns
            tk.Label(
                cards_frame,
                text=text,
                relief="ridge",
                borderwidth=1,
                padx=5,
                pady=3,
                width=tile_width,
            ).grid(row=row, column=column, padx=3, pady=3, sticky="w")

    def play_selected_card(self, card_index):
        if not self.last_analysis:
            return

        hand_cards = list(self.last_analysis["hand_cards"])
        table_cards = list(self.last_analysis["table_cards"])

        if not 0 <= card_index < len(hand_cards):
            return

        played_card = hand_cards.pop(card_index)
        table_cards.append(played_card)
        self.set_card_editor_counts(self.table_count_vars, table_cards)
        self.clear_card_editor(self.hand_count_vars)
        self.last_analysis = None
        self.status_var.set(
            f"Recorded play: {card_label(played_card)}. Your table was updated. Enter the next passed hand to continue."
        )
        self.show_editor_view()

    def clear_hand(self):
        self.clear_card_editor(self.hand_count_vars)
        self.status_var.set("Hand cleared. Enter the cards currently in your hand, then analyze.")

    def clear_table(self):
        self.clear_card_editor(self.table_count_vars)
        self.status_var.set("Table cleared. Enter your current table state and analyze when ready.")

    def start_next_hand(self):
        self.clear_card_editor(self.table_count_vars)
        self.clear_card_editor(self.hand_count_vars)
        self.status_var.set(
            "Started a new hand. The table and hand editors were both reset."
        )
