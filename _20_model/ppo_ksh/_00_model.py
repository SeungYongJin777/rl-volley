# Import Required External Libraries
import os
import random
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim

# Import Required Internal Libraries
from _20_model import ppo_ksh


class PPO:
    def __init__(self, conf, policy_name_for_play=None):
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        self.gamma = float(self.train_conf["gamma"])
        self.gae_lambda = float(self.train_conf["gae_lambda"])
        self.clip_epsilon = float(self.train_conf["clip_epsilon"])
        self.update_epochs = int(self.train_conf["update_epochs"])
        self.rollout_size = int(self.train_conf["rollout_size"])
        self.minibatch_size = int(self.train_conf["minibatch_size"])

        self.entropy_coef = float(self.train_conf.get("ent_coef", 0.0))
        self.value_coef = float(self.train_conf.get("vf_coef", 0.5))
        self.max_grad_norm = float(self.train_conf.get("max_grad_norm", 0.5))
        self.target_kl = float(self.train_conf.get("target_kl", 0.03))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate_actor = float(self.train_conf["learning_rate_actor"])
        self.learning_rate_critic = float(self.train_conf["learning_rate_critic"])

        self.state_dim = int(ppo_ksh._03_state_design.get_state_dim())
        self.dim_action = len(ppo_ksh._04_action_space_design.action_mask())
        self.hidden_dim = int(self.train_conf["hidden_dim"])
        self.hidden_layer_count = int(self.train_conf["hidden_layer_count"])

        self.loss_function = nn.MSELoss()

        self.rollout_states = []
        self.rollout_next_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []

        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        self.actor_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + ".pth",
        )
        self.critic_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + "_critic" + ".pth",
        )

        self.actor = ppo_ksh._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)

        self.critic = ppo_ksh._02_network.create_critic_nn(
            self.state_dim,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)

        self.actor_old = ppo_ksh._02_network.create_actor_nn(
            self.state_dim,
            self.dim_action,
            self.hidden_dim,
            self.hidden_layer_count,
        ).to(self.device)

        if self.conf.train_rewrite is not True:
            if os.path.exists(self.actor_path):
                self.actor.load_state_dict(
                    torch.load(
                        self.actor_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            if os.path.exists(self.critic_path):
                self.critic.load_state_dict(
                    torch.load(
                        self.critic_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate_actor,
        )
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic,
        )

    def get_transition(self, env, state_mat):
        state = self.map_to_designed_state(state_mat)

        action_mat, action_idx, log_prob_old = ppo_ksh._06_algorithm.stochastic_action_selection(
            policy=self.actor_old,
            state=state,
        )
        action = self.map_to_designed_action(action_mat)

        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side,
            run_type="ai",
            action=action,
        )

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
        state, action_idx, log_prob_old, state_next, reward, done, _ = transition

        self.rollout_states.append(state)
        self.rollout_next_states.append(state_next)
        self.rollout_action_indices.append(action_idx)
        self.rollout_log_probs_old.append(float(log_prob_old))
        self.rollout_rewards.append(float(reward))
        self.rollout_dones.append(float(done))

        if len(self.rollout_states) >= self.rollout_size or done:
            self.update_rollout()

    def update_rollout(self):
        if len(self.rollout_states) == 0:
            return

        states = torch.as_tensor(
            self.rollout_states, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            self.rollout_next_states, dtype=torch.float32, device=self.device
        )
        action_indices = torch.as_tensor(
            self.rollout_action_indices, dtype=torch.long, device=self.device
        )
        log_probs_old = torch.as_tensor(
            self.rollout_log_probs_old, dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            self.rollout_rewards, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            self.rollout_dones, dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)

            advantages = torch.zeros_like(rewards)
            gae = 0.0

            for t in reversed(range(len(rewards))):
                non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_values[t] * non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
                advantages[t] = gae

            returns = advantages + values
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)

            if torch.isnan(adv_std) or adv_std.item() < 1e-8:
                advantages = advantages - adv_mean
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        batch_size = states.shape[0]
        indices = list(range(batch_size))

        for _ in range(self.update_epochs):
            random.shuffle(indices)

            approx_kl_values = []

            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = action_indices[mb_idx]
                mb_old_log_probs = log_probs_old[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = values[mb_idx]

                
                logits = self.actor(mb_states)

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("warning: invalid logits detected, skipping minibatch")
                    continue
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    print("warning: invalid log_probs detected, skipping minibatch")
                    continue

                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print("warning: invalid probs detected, skipping minibatch")
                    continue

                selected_log_probs = log_probs.gather(
                    1, mb_actions.unsqueeze(1)
                ).squeeze(1)

                ratios = torch.exp(selected_log_probs - mb_old_log_probs)

                unclipped = ratios * mb_advantages
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()

                new_values = self.critic(mb_states).squeeze(-1)

                # value clipping
                value_pred_clipped = mb_old_values + (new_values - mb_old_values).clamp(
                    -self.clip_epsilon, self.clip_epsilon
                )
                value_loss_unclipped = (new_values - mb_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy = -(probs * log_probs).sum(dim=1).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer_for_actor.zero_grad(set_to_none=True)
                self.optimizer_for_critic.zero_grad(set_to_none=True)
                if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy) or torch.isnan(loss):
                    print("warning: NaN detected in PPO losses, skipping update")
                    continue
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer_for_actor.step()
                self.optimizer_for_critic.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - selected_log_probs).mean().abs().item()
                    approx_kl_values.append(approx_kl)

            if len(approx_kl_values) > 0 and sum(approx_kl_values) / len(approx_kl_values) > self.target_kl:
                break

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.rollout_states = []
        self.rollout_next_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []

    def get_train_configuration(self):
        return ppo_ksh._01_params.get_train_params()

    def map_to_designed_state(self, state_mat):
        state_custom = ppo_ksh._03_state_design.calculate_state_key(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        action_custom = action_mat * ppo_ksh._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat):
        reward_custom = ppo_ksh._05_reward_design.calculate_reward(reward_mat)
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        del epsilon
        state = self.map_to_designed_state(state_mat)
        action_mat, _, _ = ppo_ksh._06_algorithm.stochastic_action_selection(
            policy=self.actor,
            state=state,
        )
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        ppo_ksh._02_network.save_nn(self.actor, self.actor_path)
        ppo_ksh._02_network.save_nn(self.critic, self.critic_path)