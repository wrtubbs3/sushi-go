"""
Quick diagnostic plots for DQN training CSV with robust style fallback.

Usage:
    python plot_dqn_diagnostics.py --file dqn_stats.csv --window 200

Requires:
    pandas, matplotlib, numpy
"""

import datetime
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def choose_style(preferred=None):
    avail = plt.style.available
    if preferred is None:
        preferred = ['seaborn-darkgrid', 'seaborn', 'ggplot', 'bmh', 'classic']
    for s in preferred:
        if s in avail:
            return s
    return None

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file at {path}")
    df = pd.read_csv(path)
    return df

def ensure_columns(df):
    expected = [
        "timestamp", "steps_done", "games_trained", "loss", "td_error", "avg_q",
        "epsilon", "replay_size", "batch_size", "grad_norm",
        "batch_reward_mean", "batch_reward_max", "n_terminals",
        "max_next_q", "q_targets_mean", "is_target_update", "batch_reward_std",
        "batch_reward_95"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    int_cols = ["steps_done", "games_trained", "replay_size", "batch_size", "n_terminals", "is_target_update"]
    float_cols = [
        "loss", "td_error", "avg_q", "epsilon", "grad_norm", "batch_reward_mean",
        "batch_reward_max", "max_next_q", "q_targets_mean", "batch_reward_std",
        "batch_reward_95"]
    for c in int_cols + float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rolling_or_raw(series, window):
    if window is not None and window > 1:
        return series.rolling(window=window, min_periods=1).mean()
    return series

def plot_diagnostics(df, window=1, show=True):
    print("[INFO] Starting plot generation...")
    style = choose_style()
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass

    if 'steps_done' in df.columns and not df['steps_done'].isna().all():
        x = df['steps_done'].values
        x_label = "steps_done"
    else:
        x = df.index.values
        x_label = "index"

    print(f"[INFO] DataFrame has {len(df)} rows, computing target updates...")
    # Robustly coerce the column to numeric (handles strings, floats, bools, NaN)
    if 'is_target_update' in df.columns:
        try:
            mask = pd.to_numeric(df['is_target_update'], errors='coerce').fillna(0).astype(int)
        except Exception:
            # fallback: interpret truthy values
            mask = df['is_target_update'].astype(bool).astype(int)
        target_updates = np.where(mask == 1)[0]
        n_updates = len(target_updates)
        print(f"[INFO] Found {n_updates} target updates (raw)")
        # Avoid plotting an excessive number of vertical lines which will hang matplotlib
        max_markers = 2000
        if n_updates > max_markers:
            step = max(1, n_updates // max_markers)
            target_updates = target_updates[::step]
            print(f"[INFO] Downsampled target updates to {len(target_updates)} markers (every {step}th)")
        target_updates = target_updates.tolist()
    else:
        target_updates = []

    w = int(window) if window is not None else 1

    print(f"[INFO] Creating Figure 1 (loss, td_error, replay_size, grad_norm)...")
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("DQN Training Diagnostics", fontsize=14)

    ax0 = axs[0]
    # ax0.plot(x, rolling_or_raw(df['loss'], w), label='loss', color='C0', alpha=0.9)
    ax0.semilogy(x, rolling_or_raw(df['loss'], w), label='loss', color='C0', alpha=0.9)
    ax0.set_ylabel("loss")
    ax0.legend(loc='upper left')
    ax0b = ax0.twinx()
    # ax0b.plot(x, rolling_or_raw(df['avg_q'], w), label='avg_q', color='C2', alpha=0.6)
    ax0b.semilogy(x, rolling_or_raw(df['avg_q'], w), label='avg_q', color='C2', alpha=0.6)
    ax0b.set_ylabel("avg_q")
    ax0b.legend(loc='upper right')

    # axs[1].plot(x, rolling_or_raw(df['td_error'], w), label='td_error', color='C1')
    axs[1].semilogy(x, rolling_or_raw(df['td_error'], w), label='td_error', color='C1')
    axs[1].set_ylabel("td_error")
    axs[1].legend(loc='upper left')

    ax2 = axs[2]
    ax2.plot(x, df['replay_size'], label='replay_size', color='C3', alpha=0.6)
    ax2.set_ylabel("replay_size")
    ax2.legend(loc='upper left')
    ax2b = ax2.twinx()
    ax2b.plot(x, rolling_or_raw(df['epsilon'], w), label='epsilon', color='C4', alpha=0.9)
    ax2b.set_ylabel("epsilon")
    ax2b.legend(loc='upper right')

    ax3 = axs[3]
    # ax3.plot(x, rolling_or_raw(df['grad_norm'], w), label='grad_norm', color='C5')
    ax3.semilogy(x, rolling_or_raw(df['grad_norm'], w), label='grad_norm', color='C5')
    ax3.set_ylabel("grad_norm")
    ax3.legend(loc='upper left')
    ax3b = ax3.twinx()
    # ax3b.plot(x, rolling_or_raw(df['batch_reward_max'], w), label='batch_reward_max', color='C6', alpha=0.7)
    ax3b.semilogy(x, rolling_or_raw(df['batch_reward_max'], w), label='batch_reward_max', color='C6', alpha=0.7)
    ax3b.set_ylabel("batch_reward_max")
    ax3b.legend(loc='upper right')

    print(f"[INFO] Creating Figure 2 (terminals, max_next_q, q_targets)...")
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig2.suptitle("Terminal & Target Q Diagnostics", fontsize=14)
    axs2[0].plot(x, df['n_terminals'], label='n_terminals', color='C7')
    axs2[0].set_ylabel("n_terminals")
    axs2[0].legend()
    # axs2[1].plot(x, rolling_or_raw(df['max_next_q'], w), label='max_next_q', color='C8', alpha=0.9)
    axs2[1].semilogy(x, rolling_or_raw(df['max_next_q'], w), label='max_next_q', color='C8', alpha=0.9)
    # axs2[1].plot(x, rolling_or_raw(df['q_targets_mean'], w), label='q_targets_mean', color='C9', alpha=0.7)
    axs2[1].semilogy(x, rolling_or_raw(df['q_targets_mean'], w), label='q_targets_mean', color='C9', alpha=0.7)
    axs2[1].set_ylabel("Q stats")
    axs2[1].legend()

    # Figure 3 for batch reward distribution stats
    print(f"[INFO] Creating Figure 3 (batch_reward_std, batch_reward_95)...")
    fig3, axs3 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig3.suptitle("Batch Reward Distribution Diagnostics", fontsize=14)
    axs3[0].plot(x, rolling_or_raw(df['batch_reward_std'], w), label='batch_reward_std', color='C10')
    axs3[0].set_ylabel("batch_reward_std")
    axs3[0].legend(loc='upper left')
    axs3[1].plot(x, rolling_or_raw(df['batch_reward_95'], w), label='batch_reward_95', color='C11')
    axs3[1].set_ylabel("batch_reward_95")
    axs3[1].legend(loc='upper left')

    # Build master axis list and add target-update marker lines (downsampled)
    all_axes = list(axs) + list(axs2) + list(axs3)
    print(f"[INFO] Adding {len(target_updates)} target update markers...")
    for axis in all_axes:
        for idx in target_updates:
            if idx < len(x):
                axis.axvline(x=x[idx], color='red', alpha=0.12, linewidth=0.8)

    axs[-1].set_xlabel(x_label)
    axs2[-1].set_xlabel(x_label)
    axs3[-1].set_xlabel(x_label)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Always save to script directory with timestamp
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = os.path.join(script_dir, f"dqn_stats_1_{timestamp}.png")
    filename2 = os.path.join(script_dir, f"dqn_stats_2_{timestamp}.png")
    filename3 = os.path.join(script_dir, f"dqn_stats_3_{timestamp}.png")
    print(f"[INFO] Saving figures to {filename1}, {filename2} and {filename3}...")
    fig.savefig(filename1, dpi=150)
    fig2.savefig(filename2, dpi=150)
    fig3.savefig(filename3, dpi=150)
    print(f"[INFO] Saved {filename1}, {filename2} and {filename3}")

    if show:
        print("[INFO] Displaying plots...")
        plt.show()
    else:
        plt.close('all')

    print("[INFO] Plot generation complete.")

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot DQN diagnostics CSV")
    parser.add_argument("--file", "-f", default="dqn_stats.csv", help="CSV file path")
    parser.add_argument("--window", "-w", type=int, default=200, help="rolling window for smoothing (use 1 for raw)")
    parser.add_argument("--no-show", action="store_true", help="don't call plt.show()")
    return parser


def main(file="dqn_stats.csv", window=200, show=True):
    """Load a diagnostics CSV and generate plots.

    This function is safe to import and call directly from notebooks or other
    Python modules without tripping over command-line argument parsing.
    """

    try:
        df = load_csv(file)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    df = ensure_columns(df)

    plot_diagnostics(df, window=window, show=show)


def cli(argv=None):
    """Command-line entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    main(file=args.file, window=args.window, show=(not args.no_show))

if __name__ == "__main__":
    cli()
