def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate_actor": 1e-5,
        "learning_rate_critic": 1e-5,

        # Discount Factor
        "gamma": 0.99,

        # PPO Update Parameters
        "clip_epsilon": 0.2,
        "update_epochs": 100,

        # Neural Network Architecture Parameters
        "hidden_dim": 32,
        "hidden_layer_count": 2,

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
