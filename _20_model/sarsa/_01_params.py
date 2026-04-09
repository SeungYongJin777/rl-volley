def get_train_params():
    """====================================================================================================
    ## Hyperparameter Setting for training
    ===================================================================================================="""
    # Define the training parameters
    TRAIN_PARAMS = {
        # Learning Rate
        "alpha": 0.15,

        # Discount Factor
        "gamma": 0.995,

        # Epsilon-Greedy Exploration Parameters
        "epsilon_start": 0.40,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.99998,

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
