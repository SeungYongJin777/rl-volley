# Import Required External Libraries
import os
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim

# Import Required Internal Libraries
from _20_model import a2c


class A2C:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Parameters for A2C
        ================================================================================================"""
        # - Load Parameter Sets
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        # - Parameters for Epsilon-Greedy Policy
        self.epsilon = float(self.train_conf["epsilon_start"])
        self.epsilon_end = float(self.train_conf["epsilon_end"])
        self.epsilon_decay = float(self.train_conf["epsilon_decay"])

        # - Parameters for Calculating Return
        self.gamma = float(self.train_conf["gamma"])

        # - Device for Training
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")

        # - Learning Rate for Training Networks
        self.learning_rate_actor = float(
            self.train_conf["learning_rate_actor"])
        self.learning_rate_critic = float(
            self.train_conf["learning_rate_critic"])

        # - Dimensions for Neural Network
        self.state_dim = int(a2c._03_state_design.get_state_dim())
        self.dim_action = len(a2c._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        # - Initial Values for Training
        self.loss_function = nn.MSELoss()

        self.rollout_length = int(self.train_conf["rollout_length"])
        self.clear_rollout()

        """================================================================================================
        ## Load Target Policy
        ================================================================================================"""
        # - Target Policy Name
        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        # - Path for Target Policy
        self.actor_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '.pth',
        )

        self.critic_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '_critic' + '.pth',
        )

        # - Create Policy Network
        self.actor = a2c._02_network.create_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)

        self.critic = a2c._02_network.create_nn(
            self.state_dim, 1,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)

        # - Load Existing Weights if Present
        if self.conf.train_rewrite is not True:
            if os.path.exists(self.actor_path):
                self.actor.load_state_dict(torch.load(
                    self.actor_path,
                    map_location=self.device,
                    weights_only=True,
                ))
            if os.path.exists(self.critic_path):
                self.critic.load_state_dict(torch.load(
                    self.critic_path,
                    map_location=self.device,
                    weights_only=True,
                ))

        # - Create Optimizer for Actor and Critic Networks
        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_critic)

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        # Map State Material to Designed State
        state = self.map_to_designed_state(state_mat)

        # Action Selection by Epsilon-Greedy Policy
        action_mat = a2c._06_algorithm.stochastic_action_selection(
            policy=self.actor, state=state, epsilon=self.epsilon)
        action_idx = int(action_mat.argmax())
        action = self.map_to_designed_action(action_mat)

        # Run Environment and Get Transition
        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        # Map to Designed State and Reward
        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        # Aggregate Transition
        transition = (state, action_idx, state_next, reward_next, done, score)

        # Update Epsilon for Next Action Selection
        self.update_epsilon()

        # Return Transition
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Accumulate a Rollout and Update Actor/Critic in Batch
        ================================================================================================"""
        state, action_idx, state_next, reward, done, _ = transition
        self.rollout_states.append(state)
        self.rollout_next_states.append(state_next)
        self.rollout_action_indices.append(action_idx)
        self.rollout_rewards.append(float(reward))
        self.rollout_dones.append(float(done))

        # - Delay Optimization Until the Rollout is Ready
        if len(self.rollout_states) < self.rollout_length and not done:
            return

        # - Update the Networks with the Current Rollout
        self.update_rollout()

    def update_rollout(self):
        """================================================================================================
        ## Update Networks from the Stored Rollout
        ================================================================================================"""
        if len(self.rollout_states) == 0:
            return

        states = torch.as_tensor(
            self.rollout_states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            self.rollout_next_states, dtype=torch.float32, device=self.device)
        action_indices = torch.as_tensor(
            self.rollout_action_indices, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(
            self.rollout_rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            self.rollout_dones, dtype=torch.float32, device=self.device)

        values = self.critic(states).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)
            targets = rewards + self.gamma * next_values * (1.0 - dones)
        advantages = targets - values

        # - Update Critic Network
        loss_critic = self.loss_function(values, targets)
        self.optimizer_for_critic.zero_grad(set_to_none=True)
        loss_critic.backward()
        self.optimizer_for_critic.step()

        # - Update Actor Network
        logits = self.actor(states)
        action_probs = torch.softmax(logits, dim=-1)
        selected_action_probs = action_probs.gather(
            1, action_indices.unsqueeze(1)).squeeze(1)
        selected_action_probs = torch.clamp(selected_action_probs, min=1e-8)
        actor_signal = torch.log(selected_action_probs) * advantages.detach()
        loss_actor = -actor_signal.mean()
        self.optimizer_for_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        self.optimizer_for_actor.step()

        self.clear_rollout()

    def clear_rollout(self):
        self.rollout_states = []
        self.rollout_next_states = []
        self.rollout_action_indices = []
        self.rollout_rewards = []
        self.rollout_dones = []

    def get_train_configuration(self):
        train_conf = a2c._01_params.get_train_params()
        return train_conf

    def update_epsilon(self):
        """====================================================================================================
        ## Get next epsilon by Algorithm
        ===================================================================================================="""
        # Calculate next epsilon by decay
        self.epsilon = a2c._06_algorithm.\
            decay_epsilon(epsilon_start=self.epsilon, epsilon_decay=self.epsilon_decay,
                          epsilon_end=self.epsilon_end)

    def map_to_designed_state(self, state_mat):
        """====================================================================================================
        ## Mapping from Environment State to Designed State
        ===================================================================================================="""
        state_custom = a2c._03_state_design.calculate_state_key(
            state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        """====================================================================================================
        ## Mapping from Policy Action to Designed Action
        ===================================================================================================="""
        action_custom = action_mat *\
            a2c._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        """====================================================================================================
        ## Mapping from Environment Reward to Designed Reward
        ===================================================================================================="""
        reward_custom = a2c._05_reward_design.calculate_reward(
            reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        """====================================================================================================
        ## Select Action for Playing
        ===================================================================================================="""
        epsilon = 0
        state = self.map_to_designed_state(state_mat)
        action_mat = a2c._06_algorithm.stochastic_action_selection(
            policy=self.actor, state=state, epsilon=epsilon)
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        a2c._02_network.save_nn(self.actor, self.actor_path)
        a2c._02_network.save_nn(self.critic, self.critic_path)
