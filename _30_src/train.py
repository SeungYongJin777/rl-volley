# Import Required External Libraries
import os
from tqdm import tqdm

# Import Required Internal Modules
import _00_environment
import _20_model


def run(conf):
    """====================================================================================================
    ## Create Required Instances
    ===================================================================================================="""
    env = create_environment_instance(conf)

    """====================================================================================================
    ## Run a number of Episodes for Training
    ===================================================================================================="""
    model_train = load_train_model(conf)
    opponent_pool = create_opponent_pool_if_available(conf, model_train)
    if opponent_pool is None:
        model_opponent = load_fixed_opponent_model(conf)
    else:
        model_opponent = None
    window_size = resolve_window_size(model_train)
    episode_rows = []
    window_rows = []
    best_eval_result = None

    for epi_idx in tqdm(range(conf.num_episode), desc="Training Progress"):
        episode_reward_sum = 0.0
        episode_steps = 0
        episode_no = epi_idx + 1
        opponent_info = resolve_episode_opponent_info(
            conf,
            model_train,
            opponent_pool,
            model_opponent,
            epi_idx,
        )
        episode_opponent = opponent_info["runtime"]

        if conf.train_side == '1p':
            env.set(
                player1=model_train,
                player2=episode_opponent,
                random_serve=conf.random_serve,
                return_state=False,
            )
        else:
            env.set(
                player1=episode_opponent,
                player2=model_train,
                random_serve=conf.random_serve,
                return_state=False,
            )

        state_mat = env.get_state(player=conf.train_side)

        while True:
            transition, state_next_mat = model_train.get_transition(env, state_mat)

            model_train.update(transition)
            env = model_train.env

            episode_reward_sum += extract_reward_from_transition(transition)
            episode_steps += 1
            state_mat = state_next_mat

            done = transition[-2]
            if done:
                break

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
                "opponent_name": str(opponent_info["name"]),
                "opponent_type": str(opponent_info["type"]),
                "opponent_source": str(opponent_info["source"]),
            }
        )

        if episode_no % window_size == 0:
            window_rows.append(build_window_row(episode_rows, window_size))

        if should_save_snapshot(model_train, opponent_pool, episode_no):
            snapshot_info = model_train.save_snapshot(episode_no)
            opponent_pool.register_snapshot(snapshot_info)

        if should_run_pool_evaluation(model_train, opponent_pool, episode_no):
            evaluation_result = evaluate_across_sides(
                conf=conf,
                model_train=model_train,
                opponent_pool=opponent_pool,
            )
            print_evaluation_summary(episode_no, evaluation_result)
            if is_better_evaluation(evaluation_result, best_eval_result):
                best_eval_result = evaluation_result
                save_best_model(model_train)

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


def create_opponent_pool_if_available(conf, model_train):
    train_algorithm = str(getattr(conf, "train_algorithm", "")).strip().lower()
    if train_algorithm != "ppo":
        return None

    train_conf = getattr(model_train, "train_conf", None)
    if isinstance(train_conf, dict) is not True:
        return None
    if bool(train_conf.get("use_opponent_pool", False)) is not True:
        return None

    opponent_pool = _20_model.ppo._06_algorithm.create_opponent_pool(
        conf=conf,
        model_train=model_train,
    )
    print_opponent_pool_bootstrap(opponent_pool)
    return opponent_pool


def print_opponent_pool_bootstrap(opponent_pool):
    if opponent_pool is None:
        return

    group_counts = opponent_pool.get_group_counts()
    print(
        "[train] opponent pool: rule={rule} random={random} snapshot={snapshot} external={external}".format(
            **group_counts
        )
    )

    for event in opponent_pool.consume_initialization_events():
        level = str(event.get("level", "info")).strip().lower()
        message = str(event.get("message", ""))
        print(f"[train][opponent_pool][{level}] {message}")


def resolve_episode_opponent_info(conf, model_train, opponent_pool, fixed_opponent, epi_idx):
    del model_train

    if opponent_pool is not None:
        opponent_entry = opponent_pool.sample_opponent(epi_idx)
        return {
            "runtime": _20_model.ppo._06_algorithm.resolve_opponent_runtime(opponent_entry),
            "name": opponent_entry["name"],
            "type": opponent_entry["type"],
            "source": opponent_entry["source"],
        }

    runtime = fixed_opponent
    opponent_name = getattr(conf, "train_opponent", None)
    if runtime == "RULE":
        opponent_name = "rule"
    elif runtime == "HUMAN":
        opponent_name = "human"
    elif hasattr(runtime, "policy_name"):
        opponent_name = getattr(runtime, "policy_name")
    if opponent_name is None:
        opponent_name = "fixed_opponent"

    opponent_type = "model"
    if isinstance(runtime, str):
        opponent_type = str(runtime).strip().lower()

    return {
        "runtime": runtime,
        "name": str(opponent_name),
        "type": str(opponent_type),
        "source": "fixed",
    }


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


def should_save_snapshot(model_train, opponent_pool, episode_no):
    if opponent_pool is None:
        return False
    if hasattr(model_train, "save_snapshot") is not True:
        return False

    train_conf = getattr(model_train, "train_conf", {}) or {}
    save_interval = int(train_conf.get("save_snapshot_interval", 0) or 0)
    if save_interval <= 0:
        return False
    if episode_no <= 0:
        return False
    return episode_no % save_interval == 0


def should_run_pool_evaluation(model_train, opponent_pool, episode_no):
    if opponent_pool is None:
        return False

    train_conf = getattr(model_train, "train_conf", {}) or {}
    eval_interval = int(train_conf.get("eval_interval", 0) or 0)
    if eval_interval <= 0:
        return False
    if episode_no <= 0:
        return False
    return episode_no % eval_interval == 0


def print_evaluation_summary(episode_no, evaluation_result):
    combined_rule_win_rate = float(evaluation_result.get("combined_rule_win_rate", 0.0))
    combined_overall_win_rate = float(evaluation_result.get("combined_overall_win_rate", 0.0))
    min_side_overall_win_rate = float(evaluation_result.get("min_side_overall_win_rate", 0.0))
    print(
        "[train] eval episode={} combined_rule_win_rate={:.3f} combined_overall_win_rate={:.3f} min_side_overall_win_rate={:.3f}".format(
            int(episode_no),
            combined_rule_win_rate,
            combined_overall_win_rate,
            min_side_overall_win_rate,
        )
    )

    side_results = evaluation_result.get("side_results", {})
    for side_name in ("1p", "2p"):
        side_result = side_results.get(side_name, {})
        if len(side_result) == 0:
            continue
        print(
            "[train] eval side={} overall_win_rate={:.3f} worst_group_win_rate={:.3f}".format(
                side_name,
                float(side_result.get("overall_win_rate", 0.0)),
                float(side_result.get("worst_group_win_rate", 0.0)),
            )
        )
        for group_name, win_rate in side_result.get("group_win_rates", {}).items():
            print(
                "[train] eval side={} group={} win_rate={:.3f}".format(
                    side_name,
                    str(group_name),
                    float(win_rate),
                )
            )

    for group_name, win_rate in evaluation_result.get("combined_group_win_rates", {}).items():
        print(
            "[train] eval combined group={} win_rate={:.3f}".format(
                str(group_name),
                float(win_rate),
            )
        )


def is_better_evaluation(candidate_result, best_result):
    if candidate_result is None:
        return False
    if best_result is None:
        return True

    candidate_key = (
        float(candidate_result.get("combined_rule_win_rate", 0.0)),
        float(candidate_result.get("combined_overall_win_rate", 0.0)),
        float(candidate_result.get("min_side_overall_win_rate", 0.0)),
    )
    best_key = (
        float(best_result.get("combined_rule_win_rate", 0.0)),
        float(best_result.get("combined_overall_win_rate", 0.0)),
        float(best_result.get("min_side_overall_win_rate", 0.0)),
    )
    return candidate_key > best_key


def save_best_model(model_train):
    save_function = getattr(model_train, "save_best", None)
    if callable(save_function):
        save_function()


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
    render_mode = "log"
    target_score = conf.target_score_train
    seed = conf.seed

    env = _00_environment.Env(
        render_mode=render_mode,
        target_score=target_score,
        seed=seed,
    )
    return env


def create_model_runtime(conf, algorithm_name, policy_name=None):
    normalized_algorithm = str(algorithm_name).strip()

    if normalized_algorithm == 'human':
        return 'HUMAN'
    if normalized_algorithm == 'rule':
        return 'RULE'

    model_kwargs = {}
    if policy_name is not None and str(policy_name).strip() != "":
        model_kwargs["policy_name_for_play"] = str(policy_name).strip()

    return _20_model.create_model(
        conf,
        algorithm_name=normalized_algorithm,
        **model_kwargs,
    )


def load_train_model(conf):
    """====================================================================================================
    ## Loading Train Policy
    ===================================================================================================="""
    algorithm_name = getattr(conf, "train_algorithm", None)
    if algorithm_name is None or str(algorithm_name).strip() == "":
        raise ValueError("train_algorithm is required for training")

    policy_name = getattr(conf, "train_policy", None)
    return create_model_runtime(conf, algorithm_name, policy_name)


def load_fixed_opponent_model(conf):
    """====================================================================================================
    ## Loading Fixed Opponent Policy for Training
    ===================================================================================================="""
    opponent_spec = getattr(conf, "train_opponent", None)
    if opponent_spec is None:
        opponent_side = '1p' if str(getattr(conf, "train_side", "1p")).strip().lower() == '2p' else '2p'
        algorithm_name = conf.algorithm_1p if opponent_side == '1p' else conf.algorithm_2p
        policy_name = conf.policy_1p if opponent_side == '1p' else conf.policy_2p
        return create_model_runtime(conf, algorithm_name, policy_name)

    normalized_spec = str(opponent_spec).strip()
    if normalized_spec.lower() == 'human':
        return 'HUMAN'
    if normalized_spec.lower() == 'rule':
        return 'RULE'

    algorithm_name, separator, policy_name = normalized_spec.partition(':')
    if separator == "":
        policy_name = None
    if policy_name == "None":
        policy_name = None

    return create_model_runtime(conf, algorithm_name, policy_name)


def evaluate_across_sides(conf, model_train, opponent_pool):
    side_results = {}
    for evaluation_side in ("1p", "2p"):
        side_results[evaluation_side] = _20_model.ppo._06_algorithm.evaluate_against_pool(
            conf=conf,
            model_train=model_train,
            opponent_pool=opponent_pool,
            env_factory=create_environment_instance,
            evaluation_side=evaluation_side,
            update_pool_results=False,
        )

    combined_group_win_rates = {}
    for group_name in ("rule", "random", "snapshot", "external"):
        group_values = []
        for side_result in side_results.values():
            group_value = side_result.get("group_win_rates", {}).get(group_name)
            if group_value is not None:
                group_values.append(float(group_value))
        if len(group_values) > 0:
            combined_group_win_rates[group_name] = float(
                sum(group_values) / len(group_values)
            )

    combined_overall_win_rate = float(
        sum(
            float(side_result.get("overall_win_rate", 0.0))
            for side_result in side_results.values()
        ) / max(1, len(side_results))
    )
    combined_rule_win_rate = float(combined_group_win_rates.get("rule", 0.0))
    min_side_overall_win_rate = min(
        float(side_results["1p"].get("overall_win_rate", 0.0)),
        float(side_results["2p"].get("overall_win_rate", 0.0)),
    )
    worst_group_win_rate = combined_overall_win_rate
    if len(combined_group_win_rates) > 0:
        worst_group_win_rate = min(combined_group_win_rates.values())

    entry_results = []
    for evaluation_side, side_result in side_results.items():
        for row in side_result.get("entry_results", []):
            entry_row = dict(row)
            entry_row["evaluation_side"] = evaluation_side
            entry_results.append(entry_row)

    combined_result = {
        "combined_rule_win_rate": combined_rule_win_rate,
        "combined_overall_win_rate": combined_overall_win_rate,
        "min_side_overall_win_rate": float(min_side_overall_win_rate),
        "overall_win_rate": combined_overall_win_rate,
        "worst_group_win_rate": float(worst_group_win_rate),
        "combined_group_win_rates": combined_group_win_rates,
        "group_win_rates": combined_group_win_rates,
        "side_results": side_results,
        "entry_results": entry_results,
    }
    opponent_pool.latest_eval_results = combined_result
    return combined_result


if __name__ == "__main__":
    pass
