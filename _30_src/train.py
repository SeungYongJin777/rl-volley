# Import Required External Libraries
import os
from tqdm import tqdm

# Import Required Internal Modules
import _00_environment
import _20_model
from _30_src.play import load_model


def run(conf):
    """====================================================================================================
    ## Create Required Instances
    ===================================================================================================="""
    # - Create Envionment Instance
    env = create_environment_instance(conf)

    """====================================================================================================
    ## Run a number of Episodes for Training
    ===================================================================================================="""
    # - Load Models for Training and Opponent Players
    model_train = load_model(conf, player=conf.train_side)
    model_opponent = load_model(
        conf, player='1p' if conf.train_side == '2p' else '2p')
    window_size = resolve_window_size(model_train)
    episode_rows = []
    window_rows = []

    # - Run a number of Episodes for Training
    for epi_idx in tqdm(range(conf.num_episode), desc="Training Progress"):
        episode_reward_sum = 0.0
        episode_steps = 0

        # - Set the Environment
        if conf.train_side == '1p':
            env.set(player1=model_train, player2=model_opponent,
                    random_serve=conf.random_serve, return_state=False)
        else:
            env.set(player1=model_opponent, player2=model_train,
                    random_serve=conf.random_serve, return_state=False)

        # - Get Initial State
        state_mat = env.get_state(player=conf.train_side)

        # - Run an Episode
        while True:
            # - Get Transition by Action Selection and Environment Run
            transition, state_next_mat = model_train.get_transition(
                env, state_mat)

            # - Update Policy by Transition
            model_train.update(transition)
            env = model_train.env

            # - Track Episode Metrics
            episode_reward_sum += extract_reward_from_transition(transition)
            episode_steps += 1

            # - Update State
            state_mat = state_next_mat

            # - Check Terminate Condition
            done = transition[-2]
            if done:
                break

        episode_no = epi_idx + 1
        score = extract_score_from_transition(transition)
        score_train = resolve_side_score(score, conf.train_side)
        score_opponent = resolve_side_score(
            score, '1p' if conf.train_side == '2p' else '2p')
        is_win = score_train > score_opponent

        episode_rows.append(
            {
                "episode": episode_no,
                "episode_reward_sum": float(episode_reward_sum),
                "episode_steps": int(episode_steps),
                "is_win": bool(is_win),
                "score_train": float(score_train),
                "score_opponent": float(score_opponent),
            }
        )

        if episode_no % window_size == 0:
            window_rows.append(build_window_row(episode_rows, window_size))

    remaining_count = len(episode_rows) % window_size
    if remaining_count > 0:
        window_rows.append(build_window_row(episode_rows, remaining_count))

    """====================================================================================================
    ## Save Trained Policy at the End of Episode
    ===================================================================================================="""
    model_train.save()
    save_training_plot(conf, model_train, episode_rows, window_rows)


def resolve_window_size(model_train):
    window_size = 50
    train_conf = getattr(model_train, "train_conf", None)
    if isinstance(train_conf, dict):
        configured_size = train_conf.get("progress_interval", window_size)
        try:
            window_size = int(configured_size)
        except (TypeError, ValueError):
            window_size = 50
    if window_size <= 0:
        return 50
    return window_size


def extract_reward_from_transition(transition):
    reward_value = transition[-3]
    if hasattr(reward_value, "item"):
        return float(reward_value.item())
    return float(reward_value)


def extract_score_from_transition(transition):
    score = transition[-1]
    if isinstance(score, dict):
        return score
    return {}


def resolve_side_score(score, side):
    score_keys = ("player1", "p1")
    if side == "2p":
        score_keys = ("player2", "p2")

    for score_key in score_keys:
        if score_key in score:
            return float(score[score_key])
    return 0.0


def build_window_row(episode_rows, row_count):
    selected_rows = episode_rows[-int(row_count):]
    window_length = len(selected_rows)

    avg_reward = sum(row["episode_reward_sum"] for row in selected_rows) / window_length
    win_rate = sum(1.0 for row in selected_rows if row["is_win"]) / window_length
    avg_episode_steps = sum(row["episode_steps"] for row in selected_rows) / window_length

    return {
        "window_start_episode": int(selected_rows[0]["episode"]),
        "window_end_episode": int(selected_rows[-1]["episode"]),
        "avg_reward": float(avg_reward),
        "win_rate": float(win_rate),
        "avg_episode_steps": float(avg_episode_steps),
    }


def resolve_policy_name(model_train, conf):
    policy_name = getattr(model_train, "policy_name", None)
    if policy_name is not None and str(policy_name).strip() != "":
        return str(policy_name).strip()

    train_policy = getattr(conf, "train_policy", None)
    if train_policy is not None and str(train_policy).strip() != "":
        return str(train_policy).strip()
    return "train"


def save_training_plot(conf, model_train, episode_rows, window_rows):
    if len(episode_rows) == 0:
        return

    try:
        from _30_src.plot_train_metrics import save_metrics_plot
    except Exception as error:
        print(f"[train] metrics plot skipped (import failed): {error}")
        return

    policy_name = resolve_policy_name(model_train, conf)
    model_output_dir = _20_model.get_model_output_dir(conf, model_train)
    plot_path = os.path.join(
        model_output_dir,
        "metrics",
        f"{policy_name}_train_metrics.png",
    )

    try:
        save_metrics_plot(
            episode_rows=episode_rows,
            window_rows=window_rows,
            save_path=plot_path,
            title=f"{str(conf.train_algorithm).upper()} | policy={policy_name}",
        )
        print(f"[train] metrics plot saved: {plot_path}")
    except Exception as error:
        print(f"[train] metrics plot skipped (save failed): {error}")


def create_environment_instance(conf):
    """====================================================================================================
    ## Creation of Environment Instance
    ===================================================================================================="""
    # - Load Configuration
    RENDER_MODE = "log"
    TARGET_SCORE = conf.target_score_train
    SEED = conf.seed

    # - Create Envionment Instance
    env = _00_environment.Env(
        render_mode=RENDER_MODE,
        target_score=TARGET_SCORE,
        seed=SEED,
    )

    # - Return Environment Instance
    return env


def load_model(conf, player):
    """====================================================================================================
    ## Loading Policy for Each Player
    ===================================================================================================="""
    # - Check Algorithm and Policy Name for Each Player
    ALGORITHM = conf.algorithm_1p if player == '1p' else conf.algorithm_2p
    POLICY_NAME = conf.policy_1p if player == '1p' else conf.policy_2p

    # - Load Selected Policy for Each Player
    if ALGORITHM == 'human':
        model = 'HUMAN'

    elif ALGORITHM == 'rule':
        model = 'RULE'

    else:
        model = _20_model.create_model(
            conf,
            algorithm_name=ALGORITHM,
            policy_name_for_play=POLICY_NAME,
        )

    # - Return Loaded Model for Each Player
    return model


if __name__ == "__main__":
    pass
