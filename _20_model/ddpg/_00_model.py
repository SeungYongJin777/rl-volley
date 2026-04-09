# Import Required External Libraries
import os
import random
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# Import Required Internal Libraries
from _20_model import ddpg


class DDPG:
    def __init__(self, conf, policy_name_for_play=None):
        """================================================================================================
        ## Parameters for DDPG
        ================================================================================================"""
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        self.gamma = float(self.train_conf["gamma"])

        self.noise_scale = float(self.train_conf["noise_scale_start"])
        self.noise_scale_end = float(self.train_conf["noise_scale_end"])
        self.noise_scale_decay = float(self.train_conf["noise_scale_decay"])

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda"

        self.learning_rate_actor = float(
            self.train_conf["learning_rate_actor"])
        self.learning_rate_critic = float(
            self.train_conf["learning_rate_critic"])

        self.state_dim = int(ddpg._03_state_design.get_state_dim())
        self.dim_action = len(ddpg._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        self.replay_buffer_size = int(self.train_conf["replay_buffer_size"])
        self.replay_buffer = ReplayDataset(self.replay_buffer_size)
        self.replay_start_size = int(self.train_conf["replay_start_size"])
        self.batch_size = int(self.train_conf["batch_size"])

        self.update_every = int(self.train_conf["update_every"])
        self.tau = float(self.train_conf["tau"])
        self.env_steps = 0
        self.training_steps = 0

        self.loss_function = nn.MSELoss()

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

        self.actor = ddpg._02_network.create_actor_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)
        self.critic = ddpg._02_network.create_critic_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)
        self.actor_target = ddpg._02_network.create_actor_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
        ).to(self.device)
        self.critic_target = ddpg._02_network.create_critic_nn(
            self.state_dim, self.dim_action,
            self.hidden_dim, self.hidden_layer_count,
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

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_critic)

    def get_transition(self, env, state_mat):
        """====================================================================================================
        ## Get Transition by Algorithm
        ===================================================================================================="""
        state = self.map_to_designed_state(state_mat)

        # - a_t = mu_theta(s_t) + noise
        action_mat = ddpg._06_algorithm.deterministic_action_selection(
            policy=self.actor,
            state=state,
            noise_scale=self.noise_scale,
        )
        action = self.map_to_designed_action(action_mat)

        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side, run_type='ai', action=action)

        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(reward_next_mat)

        transition = (state, action_mat, state_next, reward_next, done, score)
        self.update_noise_scale()
        return transition, state_next_mat

    def update(self, transition):
        """================================================================================================
        ## Update Actor/Critic by Transition
        ================================================================================================"""
        state, action, state_next, reward_next, done, _ = transition

        # - D.add(s_t, a_t, r_t+1, s_t+1, done)
        self.replay_buffer.append(
            state, action, reward_next, state_next, done)

        if len(self.replay_buffer) < self.replay_start_size:
            return

        if self.env_steps % self.update_every != 0:
            self.env_steps += 1
            return
        self.env_steps += 1

        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch = random.sample(self.replay_buffer.transitions, batch_size)

        states, actions, rewards, states_next, dones = zip(*batch)
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        states_next = torch.stack(states_next).to(self.device)
        dones = torch.stack(dones).to(self.device)

        # - y_hat_i = r_i + gamma * Q_target(s_i+1, mu_target(s_i+1)) * (not done)
        with torch.no_grad():
            actions_next = self.get_deterministic_action_vector(
                self.actor_target, states_next)
            q_targets_next = self.critic_target(
                states_next, actions_next).squeeze(-1)
            q_targets = rewards + self.gamma * q_targets_next * (1.0 - dones)

        # - phi <- phi - beta * grad_phi (Q_phi(s_i, a_i) - y_hat_i)^2
        qvalues = self.critic(states, actions).squeeze(-1)
        loss_critic = self.loss_function(qvalues, q_targets)
        self.optimizer_for_critic.zero_grad(set_to_none=True)
        loss_critic.backward()
        self.optimizer_for_critic.step()

        # - theta <- theta + alpha * grad_theta Q_phi(s, mu_theta(s))
        actor_actions = self.get_deterministic_action_vector(
            self.actor, states)
        loss_actor = -self.critic(states, actor_actions).mean()
        self.optimizer_for_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        self.optimizer_for_actor.step()

        # - soft target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        self.training_steps += 1

    def get_train_configuration(self):
        return ddpg._01_params.get_train_params()

    def update_noise_scale(self):
        self.noise_scale = ddpg._06_algorithm.decay_noise_scale(
            noise_scale_start=self.noise_scale,
            noise_scale_decay=self.noise_scale_decay,
            noise_scale_end=self.noise_scale_end,
        )

    def get_deterministic_action_vector(self, actor_model, states):
        # - Discrete environment adaptation:
        #   Use a probability vector as mu(s) so the critic still receives
        #   a differentiable action representation.
        logits = actor_model(states)
        return torch.softmax(logits, dim=-1)

    def soft_update(self, target_model, source_model):
        tau = float(self.tau)
        with torch.no_grad():
            for target_param, source_param in zip(
                target_model.parameters(),
                source_model.parameters(),
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * source_param.data)

    def map_to_designed_state(self, state_mat):
        state_custom = ddpg._03_state_design.calculate_state_key(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        action_custom = action_mat * ddpg._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        reward_custom = ddpg._05_reward_design.calculate_reward(reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        # - Keep the public method name for environment compatibility.
        del epsilon
        state = self.map_to_designed_state(state_mat)
        action_mat = ddpg._06_algorithm.deterministic_action_selection(
            policy=self.actor,
            state=state,
            noise_scale=0.0,
        )
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        ddpg._02_network.save_nn(self.actor, self.actor_path)
        ddpg._02_network.save_nn(self.critic, self.critic_path)


class ReplayDataset(Dataset):
    def __init__(self, max_size):
        self.max_size = int(max_size)
        self.transitions = []

    def append(self, state, action, reward, state_next, done):
        self.transitions.append(
            (
                torch.as_tensor(state, dtype=torch.float32),
                torch.as_tensor(action, dtype=torch.float32),
                torch.tensor(float(reward), dtype=torch.float32),
                torch.as_tensor(state_next, dtype=torch.float32),
                torch.tensor(float(done), dtype=torch.float32),
            )
        )
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        return self.transitions[index]
