from pathlib import Path

import matplotlib.pyplot as plt


def save_metrics_plot(episode_rows, window_rows, save_path, title=None):
    figure = create_metrics_figure(
        episode_rows=episode_rows,
        window_rows=window_rows,
        title=title,
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_metrics_figure(episode_rows, window_rows, title=None):
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

    plot_episode_reward(axes[0], episode_rows, window_rows)
    plot_window_win_rate(axes[1], window_rows)
    plot_window_episode_steps(axes[2], window_rows)

    if title:
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        figure.tight_layout()

    return figure


def plot_episode_reward(axis, episode_rows, window_rows):
    episode_indices = [int(row["episode"]) for row in episode_rows]
    episode_rewards = [float(row["episode_reward_sum"]) for row in episode_rows]
    window_centers = [
        (int(row["window_start_episode"]) + int(row["window_end_episode"])) / 2.0
        for row in window_rows
    ]
    avg_rewards = [float(row["avg_reward"]) for row in window_rows]

    axis.plot(
        episode_indices,
        episode_rewards,
        color="tab:blue",
        alpha=0.25,
        linewidth=1.0,
        label="Episode Reward",
    )
    axis.plot(
        window_centers,
        avg_rewards,
        color="tab:blue",
        linewidth=2.0,
        label="Window Avg Reward",
    )
    axis.set_title("Episode Reward")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.grid(alpha=0.3)
    axis.legend()


def plot_window_win_rate(axis, window_rows):
    window_ends = [int(row["window_end_episode"]) for row in window_rows]
    win_rates = [float(row["win_rate"]) for row in window_rows]

    axis.plot(
        window_ends,
        win_rates,
        color="tab:green",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    axis.set_title("Window Win Rate")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Win Rate")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)


def plot_window_episode_steps(axis, window_rows):
    window_ends = [int(row["window_end_episode"]) for row in window_rows]
    avg_episode_steps = [float(row["avg_episode_steps"]) for row in window_rows]

    axis.plot(
        window_ends,
        avg_episode_steps,
        color="tab:orange",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    axis.set_title("Window Average Episode Steps")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Avg Steps")
    axis.grid(alpha=0.3)
