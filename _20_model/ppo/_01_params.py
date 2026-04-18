def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate_actor": 5e-5,
        "learning_rate_critic": 5e-5,

        # Discount Factor
        "gamma": 0.99,
        "gae_lambda": 0.95,

        # PPO Update Parameters
        "clip_epsilon": 0.15,
        "update_epochs": 15,
        "entropy_coef": 0.005,
        "value_loss_coef": 0.5,
        "gradient_clip_norm": 0.5,
        "normalize_advantages": True,
        "advantage_norm_eps": 1e-8,

        # Neural Network Architecture Parameters
        "hidden_dim": 64,
        "hidden_layer_count": 2,

        # Maximum Steps per Episode
        "max_steps_per_episode": 30*30,

        # Progress Display Options
        "show_progress": True,
        "progress_interval": 50,

        # Opponent Pool
        "use_opponent_pool": True,
        "opponent_sampling_mode": "fixed_interval",
        "opponent_swap_interval": 5,

        # Opponent Types
        "use_rule_opponent": True,
        "use_random_opponent": True,
        "use_external_opponents": True,
        "use_snapshot_opponents": True,

        # Sampling Ratios
        "rule_ratio": 0.30,
        "random_ratio": 0.05,
        "snapshot_ratio": 0.20,
        "external_ratio": 0.45,

        # Snapshot / Evaluation
        "save_snapshot_interval": 250,
        "max_snapshot_size": 12,
        "eval_interval": 250,
        "eval_num_episode_per_opponent": 15,

        # External Opponents
        "external_opponent_paths": [
            "dqn_wjc:_20_model/DQN_wjc/outputs/policy_trained/dqn_exp_01.pth",
            "ppo_ksh:_20_model/ppo_ksh/outputs/policy_trained/ppo_killer_ksh.pth",
            "raichu_qlearning:_20_model/raichu_qlearning/outputs/policy_trained/million_volts_ep5000.pth",
            "ppo_ksh:_20_model/ppo_ksh/outputs/policy_trained/ppo_ksh.pth"
        ],

        # Hard Negative Sampling
        "use_hard_negative_sampling": True,
        "hard_negative_ratio": 0.30,
    }
    return TRAIN_PARAMS


def get_play_params():
    """====================================================================================================
    ## Hyperparameter Setting for Playing
    ===================================================================================================="""
    PLAY_PARAMS = {
        "max_steps": 30*60*60,
    }
    return PLAY_PARAMS
