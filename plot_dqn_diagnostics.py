"""
Quick diagnostic plots for DQN training CSV with robust style fallback.

Usage:
    python plot_dqn_diagnostics.py --file dqn_stats.csv --window 200 --save diag.png

Requires:
    pandas, matplotlib, numpy
"""

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
        "max_next_q", "q_targets_mean", "is_target_update"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    int_cols = ["steps_done", "games_trained", "replay_size", "batch_size", "n_terminals", "is_target_update"]
    float_cols = ["loss", "td_error", "avg_q", "epsilon", "grad_norm", "batch_reward_mean", "batch_reward_max", "max_next_q", "q_targets_mean"]
    for c in int_cols + float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rolling_or_raw(series, window):
    if window is not None and window > 1:
        return series.rolling(window=window, min_periods=1).mean()
    return series

def plot_diagnostics(df, window=1, savepath=None, show=True):
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

    target_updates = df.index[df['is_target_update'] == 1].tolist() if 'is_target_update' in df.columns else []

    w = int(window) if window is not None else 1

    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("DQN Training Diagnostics", fontsize=14)

    ax0 = axs[0]
    ax0.plot(x, rolling_or_raw(df['loss'], w), label='loss', color='C0', alpha=0.9)
    ax0.set_ylabel("loss")
    ax0.legend(loc='upper left')
    ax0b = ax0.twinx()
    ax0b.plot(x, rolling_or_raw(df['avg_q'], w), label='avg_q', color='C2', alpha=0.6)
    ax0b.set_ylabel("avg_q")
    ax0b.legend(loc='upper right')

    axs[1].plot(x, rolling_or_raw(df['td_error'], w), label='td_error', color='C1')
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
    ax3.plot(x, rolling_or_raw(df['grad_norm'], w), label='grad_norm', color='C5')
    ax3.set_ylabel("grad_norm")
    ax3.legend(loc='upper left')
    ax3b = ax3.twinx()
    ax3b.plot(x, rolling_or_raw(df['batch_reward_max'], w), label='batch_reward_max', color='C6', alpha=0.7)
    ax3b.set_ylabel("batch_reward_max")
    ax3b.legend(loc='upper right')

    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig2.suptitle("Terminal & Target Q Diagnostics", fontsize=14)
    axs2[0].plot(x, df['n_terminals'], label='n_terminals', color='C7')
    axs2[0].set_ylabel("n_terminals")
    axs2[0].legend()
    axs2[1].plot(x, rolling_or_raw(df['max_next_q'], w), label='max_next_q', color='C8', alpha=0.9)
    axs2[1].plot(x, rolling_or_raw(df['q_targets_mean'], w), label='q_targets_mean', color='C9', alpha=0.7)
    axs2[1].set_ylabel("Q stats")
    axs2[1].legend()

    # CORRECTED concatenation: convert both to lists first
    all_axes = list(axs) + list(axs2)
    for axis in all_axes:
        for idx in target_updates:
            if idx < len(x):
                axis.axvline(x=x[idx], color='red', alpha=0.12, linewidth=0.8)

    axs[-1].set_xlabel(x_label)
    axs2[-1].set_xlabel(x_label)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savepath:
        base, ext = os.path.splitext(savepath)
        fig.savefig(base + "_part1.png", dpi=150)
        fig2.savefig(base + "_part2.png", dpi=150)
        print(f"Saved {base + '_part1.png'} and {base + '_part2.png'}")

    if show:
        plt.show()
    else:
        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Plot DQN diagnostics CSV")
    parser.add_argument("--file", "-f", default="dqn_stats.csv", help="CSV file path")
    parser.add_argument("--window", "-w", type=int, default=200, help="rolling window for smoothing (use 1 for raw)")
    parser.add_argument("--save", "-s", default=None, help="save figure to this path (png)")
    parser.add_argument("--no-show", action="store_true", help="don't call plt.show()")
    args = parser.parse_args()

    try:
        df = load_csv(args.file)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    df = ensure_columns(df)

    plot_diagnostics(df, window=args.window, savepath=args.save, show=(not args.no_show))

if __name__ == "__main__":
    main()
