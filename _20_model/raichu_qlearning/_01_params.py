def get_train_params():
    return {
        "alpha": 0.20,
        "alpha_min": 0.02,
        "alpha_decay": 0.10,
        "gamma": 0.995,
        "epsilon_start": 0.90,
        "epsilon_end": 0.03,
        "epsilon_decay": 0.999999,
        "evaluation_num_episodes": 20,
        "evaluation_checkpoints": [5000, 10000, 15000, 20000, 25000, 30000],
        "gauntlet_num_seeds": 5,
        "gauntlet_seed_stride": 1000,
        "gauntlet_quick_target_score": 3,
        "gauntlet_quick_episodes_per_seed": 2,
        "gauntlet_final_target_score": 15,
        "gauntlet_final_episodes_per_seed": 4,
        "gauntlet_promote_win_rate": 0.55,
        "flagship_policy": "million_volts",
        "latest_policy": "million_volts_latest",
        "previous_policy": "million_volts_prev",
        "gauntlet_opponents": [
            {
                "algorithm": "rule",
                "policy": None,
                "label": "rule",
            },
            {
                "algorithm": "qlearning",
                "policy": "champ",
                "label": "qlearning:champ",
            },
            {
                "algorithm": "raichu_qlearning",
                "policy": "million_volts",
                "label": "raichu_qlearning:million_volts",
            },
            {
                "algorithm": "raichu_qlearning",
                "policy": "million_volts_latest",
                "label": "raichu_qlearning:million_volts_latest",
            },
        ],
        "max_steps_per_episode": 30 * 30,
        "show_progress": True,
        "progress_interval": 50,
    }


def get_play_params():
    return {
        "max_steps": 30 * 60 * 60,
    }
