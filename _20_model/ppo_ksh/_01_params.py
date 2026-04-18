def get_train_params():
    TRAIN_PARAMS = {
        # Optimizer
        "learning_rate_actor": 3e-4,
        "learning_rate_critic": 3e-4,

        # PPO core
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "update_epochs": 10,
        "rollout_size": 1024,
        "minibatch_size": 128,

        # Loss weights
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.03,

        # Network
        "hidden_dim": 128,
        "hidden_layer_count": 2,

        # Episode
        "max_steps_per_episode": 30 * 30,

        # Logging
        "show_progress": True,
        "progress_interval": 50,

        # 🔥 추가
        "evaluation_checkpoints": [
            100,
            500,
            1000,
            2000,
            5000,
            10000,
        ],
    }
    return TRAIN_PARAMS


def get_play_params():
    PLAY_PARAMS = {
        "max_steps": 30 * 60 * 60,
    }
    return PLAY_PARAMS






