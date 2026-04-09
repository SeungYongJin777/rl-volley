def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    # Define the training parameters
    TRAIN_PARAMS = {
        # Learning Rate
        "learning_rate_actor": 1e-4,
        "learning_rate_critic": 1e-4,

        # Discount Factor
        "gamma": 0.9999,

        # Epsilon-Greedy Exploration Parameters
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.9995,

        # Neural Network Architecture Parameters
        "hidden_dim": 64,
        "hidden_layer_count": 10,
        "rollout_length": 512,

        # Maximum Steps per Episode
        "max_steps_per_episode": 30*30,

        # Progress Display Options
        "show_progress": True,
        "progress_interval": 50,
    }

    # Return the training parameters
    return TRAIN_PARAMS


def get_play_params():
    """====================================================================================================
    ## Hyperparameter Setting for Playing
    ===================================================================================================="""
    # Define the play parameters
    PLAY_PARAMS = {
        # Maximum Steps per Episode
        "max_steps": 30*60*60,
    }

    # Return the play parameters
    return PLAY_PARAMS
