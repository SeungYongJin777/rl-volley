# Import Required External Libraries
import os
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim

# Import Required Internal Libraries
from _20_model import ppo


class PPO:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Parameters for PPO
        ================================================================================================"""
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        self.gamma = float(self.train_conf["gamma"])
        self.clip_epsilon = float(self.train_conf["clip_epsilon"])
        self.update_epochs = int(self.train_conf["update_epochs"])

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate_actor = float(
            self.train_conf["learning_rate_actor"])
        self.learning_rate_critic = float(
            self.train_conf["learning_rate_critic"])

        self.state_dim = int(ppo._03_state_design.get_state_dim())
        self.dim_action = len(ppo._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        self.loss_function = nn.MSELoss()
        self.rollout_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []

        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        self.actor_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '.pth',
        )
        self.critic_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + '_critic' + '.pth',
        )

        self.actor = ppo._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)
        self.critic = ppo._02_network.create_critic_nn(
            self.state_dim,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)
        self.actor_old = ppo._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)

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

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_critic)

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        state = self.map_to_designed_state(state_mat)

        # - Collect rollouts with old policy
        action_mat, action_idx, log_prob_old = \
            ppo._06_algorithm.stochastic_action_selection(
                policy=self.actor_old,
                state=state,
            )
        action = self.map_to_designed_action(action_mat)

        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        transition = (
            state,
            action_idx,
            log_prob_old,
            state_next,
            reward_next,
            done,
            score,
        )
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Accumulate Rollout and Update PPO in Batch
        ================================================================================================"""
        state, action_idx, log_prob_old, state_next, reward, done, _ = transition
        del state_next

        self.rollout_states.append(state)
        self.rollout_action_indices.append(action_idx)
        self.rollout_log_probs_old.append(float(log_prob_old))
        self.rollout_rewards.append(float(reward))

        if done:
            self.update_rollout()

    def update_rollout(self):
        """================================================================================================
        ## PPO Update from Collected Rollout
        ================================================================================================"""
        if len(self.rollout_states) == 0:
            return

        states = torch.as_tensor(
            self.rollout_states, dtype=torch.float32, device=self.device)
        action_indices = torch.as_tensor(
            self.rollout_action_indices, dtype=torch.long, device=self.device)
        log_probs_old = torch.as_tensor(
            self.rollout_log_probs_old, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            self.rollout_rewards, dtype=torch.float32, device=self.device)

        # - Build reward-to-go returns:
        #   return[t] = reward[t] + gamma * return[t+1]
        with torch.no_grad():
            returns = torch.empty_like(rewards)
            running_return = torch.zeros(
                (), dtype=torch.float32, device=self.device)

            for step_idx in range(rewards.shape[0] - 1, -1, -1):
                running_return = rewards[step_idx] + \
                    self.gamma * running_return
                returns[step_idx] = running_return

            values_old = self.critic(states).squeeze(-1)
            advantages = returns - values_old

        for _ in range(self.update_epochs):
            # - Use the whole rollout as one batch each epoch.
            logits = self.actor(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                1, action_indices.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(selected_log_probs - log_probs_old)

            # - PPO actor loss:
            #   min(ratio * advantage, clip(ratio) * advantage)
            clipped_ratios = torch.clamp(
                ratios,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )
            surrogate_unclipped = ratios * advantages
            surrogate_clipped = clipped_ratios * advantages
            loss_actor = -torch.minimum(
                surrogate_unclipped,
                surrogate_clipped,
            ).mean()
            self.optimizer_for_actor.zero_grad(set_to_none=True)
            loss_actor.backward()
            self.optimizer_for_actor.step()

            # - Critic fits the return target.
            values = self.critic(states).squeeze(-1)
            loss_critic = self.loss_function(values, returns)
            self.optimizer_for_critic.zero_grad(set_to_none=True)
            loss_critic.backward()
            self.optimizer_for_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.rollout_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []

    def get_train_configuration(self):
        return ppo._01_params.get_train_params()

    def map_to_designed_state(self, state_mat):
        state_custom = ppo._03_state_design.calculate_state_key(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        action_custom = action_mat * ppo._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        reward_custom = ppo._05_reward_design.calculate_reward(reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        del epsilon
        state = self.map_to_designed_state(state_mat)
        action_mat, _, _ = ppo._06_algorithm.stochastic_action_selection(
            policy=self.actor,
            state=state,
        )
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        ppo._02_network.save_nn(self.actor, self.actor_path)
        ppo._02_network.save_nn(self.critic, self.critic_path)
