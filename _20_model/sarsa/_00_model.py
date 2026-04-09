# Import Required External Libraries
import os
import numpy as np
import _20_model

# Import Required Internal Libraries
from _20_model import sarsa


class Sarsa:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Initialization
        ================================================================================================"""
        # - Initilization
        self.conf = conf

        # - Load Training Parameters
        self.train_conf = self.get_train_configuration()

        # - Variables
        self.epsilon = self.train_conf["epsilon_start"]
        self.action_next_mat = None

        # - Target Policy Name
        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        # - Target Policy Path
        self.policy_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + ".pt",
        )

        if self.conf.train_rewrite is not True:
            try:
                self.policy = sarsa._02_qtable.load_qtable(
                    self.policy_path)
            except:
                self.policy = sarsa._02_qtable.create_qtable()
        else:
            self.policy = sarsa._02_qtable.create_qtable()

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        # - Map State Material to Designed State
        state = self.map_to_designed_state(state_mat)

        # - Action Selection by Epsilon-Greedy Policy
        if self.action_next_mat is None:
            action_mat = sarsa._06_algorithm.epsilon_greedy_action_selection(
                policy=self.policy,
                state=state,
                epsilon=self.epsilon,
            )
        else:
            action_mat = self.action_next_mat
        action = self.map_to_designed_action(action_mat)

        # - Run Environment and Get Transition
        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        # - Map to Designed State and Reward
        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        # - Select Next Action for SARSA Update
        if done is not True:
            action_next_mat = sarsa._06_algorithm.epsilon_greedy_action_selection(
                policy=self.policy,
                state=state_next,
                epsilon=self.epsilon,
            )
            action_next = self.map_to_designed_action(action_next_mat)
        else:
            action_next_mat = None
            action_next = None
                
        # - Aggregate Transition
        transition = (state, action_mat, state_next,
                      action_next, reward_next, done, score)

        # - Update Epsilon and Cache the Next Action
        self.update_epsilon()
        self.action_next_mat = action_next_mat

        # - Return Transition
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Update Q-Table by Transition
        ================================================================================================"""
        # - Unpack Transition
        state, action, state_next, action_next, reward_next, done, _ = transition

        # - Convert State to Q-Table Key
        state = tuple(state)
        state_next = tuple(state_next)

        # - Resolve Action Index
        action_idx = int(np.argmax(np.asarray(action, dtype=float)))

        # - Calculater Target Q-Value
        td_target = sarsa._06_algorithm.calculate_qtarget(
            policy=self.policy,
            reward=reward_next,
            state_next=state_next,
            action_next=action_next,
            gamma=self.train_conf["gamma"],
            done=done,
        )

        # - Update Q-Table by Current Transition
        alpha = float(self.train_conf["alpha"])
        self.policy[state][action_idx] = self.policy[state][action_idx] +\
            alpha * (td_target - self.policy[state][action_idx])

    def get_train_configuration(self):
        train_conf = sarsa._01_params.get_train_params()
        return train_conf

    def update_epsilon(self):
        """====================================================================================================
        ## Get next epsilon by Algorithm
        ===================================================================================================="""
        # Calculate next epsilon by decay
        self.epsilon = sarsa._06_algorithm.\
            decay_epsilon(epsilon_start=self.epsilon, epsilon_decay=self.train_conf["epsilon_decay"],
                          epsilon_end=self.train_conf["epsilon_end"])

    def map_to_designed_state(self, state_mat):
        """====================================================================================================
        ## Mapping from Environment State to Designed State
        ===================================================================================================="""
        state_custom = sarsa._03_state_design.calculate_state_key(
            state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        """====================================================================================================
        ## Mapping from Policy Action to Designed Action
        ===================================================================================================="""
        action_custom = action_mat *\
            sarsa._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        """====================================================================================================
        ## Mapping from Environment Reward to Designed Reward
        ===================================================================================================="""
        reward_custom = sarsa._05_reward_design.calculate_reward(
            reward_mat)
        return reward_custom

    def save(self):
        """================================================================================================
        ## Save Trained Policy Q-Table
        ================================================================================================"""
        sarsa._02_qtable.save_qtable(self.policy, self.policy_path)
