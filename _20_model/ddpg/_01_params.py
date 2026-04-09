def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate_actor": 1e-3,
        "learning_rate_critic": 1e-3,

        # Discount Factor
        "gamma": 0.9999,

        # Exploration Noise Parameters
        "noise_scale_start": 1.00,
        "noise_scale_end": 0.05,
        "noise_scale_decay": 0.9999,

        # Replay Buffer Parameters
        "replay_buffer_size": 100000,
        "replay_start_size": 1000,
        "batch_size": 64,

        # Neural Network Architecture Parameters
        "hidden_dim": 64,
        "hidden_layer_count": 2,

        # Update Parameters
        "update_every": 4,
        "tau": 0.005,

        # Maximum Steps per Episode
        "max_steps_per_episode": 30*30,

        # Progress Display Options
        "show_progress": True,
        "progress_interval": 50,
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
