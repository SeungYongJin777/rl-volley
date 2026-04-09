# Import External Packages
import argparse
import copy
import importlib

# Import Internal Packages
import _10_config


def build_parser(conf):
    """====================================================================================================
    ## Design of Parser for CLI
    ===================================================================================================="""
    # - Create an Parser Instance
    parser = argparse.ArgumentParser()

    # - Mode Selection
    parser.add_argument('--mode', type=str,
                        choices=('play', 'train'), required=True)

    # - Algorithm and Policy Selection for Player 1
    parser.add_argument('--1p', dest='player1', type=str, required=False)

    # - Algorithm and Policy Selection for Player 2
    parser.add_argument('--2p', dest='player2', type=str, required=False)

    # - Train Algorithm
    parser.add_argument('--train_algorithm', type=str, required=False)

    # - Train Policy
    parser.add_argument('--train_policy', type=str, required=False)

    # - Train Side
    parser.add_argument('--train_side', type=str, required=False)

    # - Train Opponent
    parser.add_argument('--train_opponent', type=str, required=False)

    # - Rewrite Existing Policy
    parser.add_argument('--train_rewrite', type=bool, required=False)

    # - Target Score Selection
    parser.add_argument('--target_score', type=int, required=False)

    # - Train Episode
    parser.add_argument('--num_episode', type=int, required=False)

    # - Random Serve
    parser.add_argument('--random_serve', type=bool, required=False)

    # - Random Seed for Reproducibility
    parser.add_argument('--seed', type=int, required=False)

    # - Return the Parser Instance
    return parser


def parse_args(conf_default, args):
    """====================================================================================================
    ## Parsing the Command-Line Arguments
    ===================================================================================================="""
    # - Copy the Default Configuration
    conf_parsed = copy.deepcopy(conf_default)

    # - Parse the Mode Selection
    conf_parsed.mode = args.mode

    # - Parse the Target Score
    if args.target_score is not None:
        if args.mode == 'train':
            conf_parsed.target_score_train = args.target_score
        elif args.mode == 'play':
            conf_parsed.target_score_play = args.target_score

    # - Parse the Algorithm and Policy Selection for Player 1
    if args.player1 is not None:
        algorithm, sep, policy = args.player1.strip().partition(':')
        conf_parsed.algorithm_1p = algorithm
        conf_parsed.policy_1p = None if (
            not sep or policy == 'None') else policy

    # - Parse the Algorithm and Policy Selection for Player 2
    if args.player2 is not None:
        algorithm, sep, policy = args.player2.strip().partition(':')
        conf_parsed.algorithm_2p = algorithm
        conf_parsed.policy_2p = None if (
            not sep or policy == 'None') else policy

    # - Parse Train Algorithm
    if args.train_algorithm is not None:
        conf_parsed.train_algorithm = args.train_algorithm

    # - Parse Train Policy
    if args.train_policy is not None:
        conf_parsed.train_policy = args.train_policy

    # - Parse Train Side
    if args.train_side is not None:
        conf_parsed.train_side = args.train_side

    # - Train Opponent
    if args.train_opponent is not None:
        conf_parsed.train_opponent = args.train_opponent

    # - Parse Train Rewrite
    if args.train_rewrite is not None:
        conf_parsed.train_rewrite = args.train_rewrite

    # - Train Episode
    if args.num_episode is not None:
        conf_parsed.num_episode = args.num_episode

    # - Parse the Random Serve
    if args.random_serve is not None:
        conf_parsed.random_serve = args.random_serve

    # - Parse the Random Seed
    if args.seed is not None:
        conf_parsed.seed = args.seed

    # - Return the Parsed Configuration
    return conf_parsed


"""Modified"""


def main(DEBUG=False, DEBUG_ARGS=None):
    """====================================================================================================
    ## Main Part for CLI
    ===================================================================================================="""
    # - Load the Default Configuration
    conf_default = _10_config.conf.Config()

    # - Build the Parser Instance
    parser = build_parser(conf_default)

    # - Parse the Command-Line Arguments
    if DEBUG:
        args = parser.parse_args(DEBUG_ARGS)
    else:
        args = parser.parse_args()
    conf_parsed = parse_args(conf_default, args)

    # - Run the Program
    if conf_parsed.mode == 'play':
        play_module = importlib.import_module("_30_src.play")
        play_module.run(conf_parsed)
    elif conf_parsed.mode == 'train':
        train_module = importlib.import_module("_30_src.train")
        train_module.run(conf_parsed)


if __name__ == "__main__":
    # Debug Mode Selection: If You Wnat to Run in Debug Mode, Set to True; Otherwise, Set to False
    DEBUG = [False, True][0]
    DEBUG_ARGS = ['--mode', 'play', '--1p', 'rule', '--2p', 'qlearning:test']

    # Execute the Main Function
    main(DEBUG=DEBUG, DEBUG_ARGS=DEBUG_ARGS)
