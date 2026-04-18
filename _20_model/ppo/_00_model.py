# Import Required External Libraries
import os
from pathlib import Path
import _20_model

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import Required Internal Libraries
from _20_model import ppo


class PPO:
    def __init__(
        self,
        conf,
        policy_name_for_play=None,
        checkpoint_path=None,
        fixed_for_opponent=False,
        eval_mode=False,
    ):
        """================================================================================================
        ## Parameters for PPO
        ================================================================================================"""
        self.conf = conf
        self.train_conf = self.get_train_configuration()
        self.fixed_for_opponent = bool(fixed_for_opponent)
        self.eval_mode = bool(eval_mode)

        self.gamma = float(self.train_conf["gamma"])
        self.gae_lambda = float(self.train_conf["gae_lambda"])
        self.clip_epsilon = float(self.train_conf["clip_epsilon"])
        self.update_epochs = int(self.train_conf["update_epochs"])
        self.entropy_coef = float(self.train_conf["entropy_coef"])
        self.value_loss_coef = float(self.train_conf["value_loss_coef"])
        self.gradient_clip_norm = float(self.train_conf["gradient_clip_norm"])
        self.normalize_advantages = bool(self.train_conf["normalize_advantages"])
        self.advantage_norm_eps = float(self.train_conf["advantage_norm_eps"])

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
        self.rollout_next_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.reset_state_history()

        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            self.policy_name = str(self.conf.train_policy).strip()

        if checkpoint_path is not None:
            self.actor_path = str(checkpoint_path)
            self.critic_path = self.resolve_critic_path(self.actor_path)
        else:
            self.actor_path = os.path.join(
                _20_model.get_model_policy_dir(self.conf, self),
                self.policy_name + '.pth',
            )
            self.critic_path = os.path.join(
                _20_model.get_model_policy_dir(self.conf, self),
                self.policy_name + '_critic' + '.pth',
            )
        self.init_policy_name = self._resolve_init_policy_name(checkpoint_path)
        self.init_actor_path = self._resolve_init_actor_path()
        self.init_critic_path = self._resolve_init_critic_path()

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

        if checkpoint_path is not None:
            actor_loaded = self._load_actor_checkpoint_with_fallback()
            critic_loaded = self._load_critic_checkpoint_with_fallback()
            if actor_loaded is not True or critic_loaded is not True:
                raise FileNotFoundError(
                    f"failed to load checkpoint for PPO policy: {checkpoint_path}"
                )
        elif self.conf.train_rewrite is not True:
            actor_loaded = self._load_actor_checkpoint_with_fallback()
            critic_loaded = self._load_critic_checkpoint_with_fallback()

            if actor_loaded is not True and self.init_actor_path is not None:
                self._load_state_dict_from_path(self.actor, self.init_actor_path)
                print(
                    f"[ppo] actor bootstrap init applied: {self.init_actor_path} -> {self.actor_path}"
                )

            if critic_loaded is not True and self.init_critic_path is not None:
                self._load_state_dict_from_path(self.critic, self.init_critic_path)
                print(
                    f"[ppo] critic bootstrap init applied: {self.init_critic_path} -> {self.critic_path}"
                )

            if (
                actor_loaded is not True
                and critic_loaded is not True
                and self.init_policy_name is not None
                and (self.init_actor_path is None or self.init_critic_path is None)
            ):
                raise FileNotFoundError(
                    "failed to bootstrap PPO init policy: "
                    + f"{self.init_policy_name}"
                )

        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer_for_actor = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_actor)
        self.optimizer_for_critic = optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_critic)

        if self.eval_mode or self.fixed_for_opponent:
            self.actor.eval()
            self.actor_old.eval()
            self.critic.eval()

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
        reward_next = self.map_to_designed_reward(
            reward_next_mat,
            current_state_mat=state_mat,
            next_state_mat=state_next_mat,
        )

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
        if self.fixed_for_opponent or self.eval_mode:
            return

        state, action_idx, log_prob_old, state_next, reward, done, _ = transition

        self.rollout_states.append(state)
        self.rollout_next_states.append(state_next)
        self.rollout_action_indices.append(action_idx)
        self.rollout_log_probs_old.append(float(log_prob_old))
        self.rollout_rewards.append(float(reward))
        self.rollout_dones.append(float(done))

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
        next_states = torch.as_tensor(
            self.rollout_next_states,
            dtype=torch.float32,
            device=self.device,
        )
        action_indices = torch.as_tensor(
            self.rollout_action_indices, dtype=torch.long, device=self.device)
        log_probs_old = torch.as_tensor(
            self.rollout_log_probs_old, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            self.rollout_rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(
            self.rollout_dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values_old = self.critic(states).squeeze(-1)
            next_values_old = self.critic(next_states).squeeze(-1)

            deltas = rewards + self.gamma * next_values_old * (1.0 - dones) - values_old
            advantages = torch.empty_like(rewards)
            running_advantage = torch.zeros(
                (), dtype=torch.float32, device=self.device
            )

            for step_idx in range(rewards.shape[0] - 1, -1, -1):
                running_advantage = deltas[step_idx] + self.gamma * self.gae_lambda * (
                    1.0 - dones[step_idx]
                ) * running_advantage
                advantages[step_idx] = running_advantage

            returns = advantages + values_old

            if self.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + self.advantage_norm_eps
                )

        for _ in range(self.update_epochs):
            logits = self.actor(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_probs = torch.softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(
                1, action_indices.unsqueeze(1)).squeeze(1)
            ratios = torch.exp(selected_log_probs - log_probs_old)
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()

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
            ).mean() - self.entropy_coef * entropy
            self.optimizer_for_actor.zero_grad(set_to_none=True)
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.gradient_clip_norm,
            )
            self.optimizer_for_actor.step()

            values = self.critic(states).squeeze(-1)
            loss_critic = self.value_loss_coef * self.loss_function(values, returns)
            self.optimizer_for_critic.zero_grad(set_to_none=True)
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.gradient_clip_norm,
            )
            self.optimizer_for_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.rollout_states = []
        self.rollout_next_states = []
        self.rollout_action_indices = []
        self.rollout_log_probs_old = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.reset_state_history()

    def get_train_configuration(self):
        return ppo._01_params.get_train_params()

    def map_to_designed_state(self, state_mat):
        history_context = self.get_state_history_context()
        state_custom = ppo._03_state_design.calculate_state_key(
            state_mat,
            history_context=history_context,
        )
        self.update_state_history(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        action_custom = action_mat * ppo._04_action_space_design.action_mask()
        return action_custom

    def map_to_designed_reward(self, reward_mat, current_state_mat=None, next_state_mat=None):
        reward_custom = ppo._05_reward_design.calculate_reward(
            reward_mat,
            current_state_mat=current_state_mat,
            next_state_mat=next_state_mat,
        )
        return reward_custom

    def select_action(self, state_mat, epsilon=0.0):
        del epsilon
        state = self.map_to_designed_state(state_mat)
        action_mat, _ = ppo._06_algorithm.deterministic_action_selection(
            policy=self.actor,
            state=state,
        )
        action = self.map_to_designed_action(action_mat)
        return action

    def save(self):
        ppo._02_network.save_nn(self.actor, self.actor_path)
        ppo._02_network.save_nn(self.critic, self.critic_path)

    @staticmethod
    def resolve_critic_path(actor_path):
        actor_path_obj = Path(actor_path)
        actor_stem = actor_path_obj.stem
        critic_name = actor_stem + "_critic" + actor_path_obj.suffix
        if actor_stem.endswith("_critic"):
            critic_name = actor_path_obj.name
        return str(actor_path_obj.with_name(critic_name))

    def _resolve_init_policy_name(self, checkpoint_path):
        if checkpoint_path is not None:
            return None

        init_policy_name = getattr(self.conf, "train_init_policy", None)
        if init_policy_name is None:
            return None

        normalized_name = str(init_policy_name).strip()
        if normalized_name == "":
            return None
        return normalized_name

    def _resolve_init_actor_path(self):
        if self.init_policy_name is None:
            return None
        if Path(self.actor_path).exists():
            return None

        policy_dir = Path(_20_model.get_model_policy_dir(self.conf, self))
        candidate_path = policy_dir / f"{self.init_policy_name}.pth"
        if candidate_path.exists() is not True:
            return None
        return str(candidate_path)

    def _resolve_init_critic_path(self):
        if self.init_actor_path is None:
            return None

        candidate_path = Path(self.resolve_critic_path(self.init_actor_path))
        if candidate_path.exists() is not True:
            return None
        return str(candidate_path)

    def save_snapshot(self, episode_idx):
        snapshot_dir = Path(_20_model.get_model_policy_dir(self.conf, self)) / "snapshots"
        snapshot_name = f"{self.policy_name}_ep{int(episode_idx):06d}"
        actor_snapshot_path = snapshot_dir / f"{snapshot_name}.pth"
        critic_snapshot_path = snapshot_dir / f"{snapshot_name}_critic.pth"

        ppo._02_network.save_nn(self.actor, actor_snapshot_path)
        ppo._02_network.save_nn(self.critic, critic_snapshot_path)

        return {
            "episode": int(episode_idx),
            "name": snapshot_name,
            "actor_path": str(actor_snapshot_path),
            "critic_path": str(critic_snapshot_path),
        }

    def save_best(self, label="best"):
        best_label = str(label).strip() or "best"
        best_actor_path = Path(_20_model.get_model_policy_dir(self.conf, self)) / f"{self.policy_name}_{best_label}.pth"
        best_critic_path = Path(_20_model.get_model_policy_dir(self.conf, self)) / f"{self.policy_name}_{best_label}_critic.pth"

        ppo._02_network.save_nn(self.actor, best_actor_path)
        ppo._02_network.save_nn(self.critic, best_critic_path)

        return {
            "actor_path": str(best_actor_path),
            "critic_path": str(best_critic_path),
        }

    def _load_state_dict_from_path(self, model, path):
        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    def _actor_candidate_paths(self):
        candidates = []
        actor_path_obj = Path(self.actor_path)
        candidates.append(str(actor_path_obj))

        if actor_path_obj.stem.endswith("_critic"):
            actor_stem = actor_path_obj.stem[:-7]
            candidates.append(str(actor_path_obj.with_name(actor_stem + actor_path_obj.suffix)))

        deduplicated = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduplicated.append(candidate)
        return deduplicated

    def _critic_candidate_paths(self):
        candidates = []
        critic_path_obj = Path(self.critic_path)
        candidates.append(str(critic_path_obj))

        if critic_path_obj.stem.endswith("_critic") is not True:
            candidates.append(str(critic_path_obj.with_name(critic_path_obj.stem + "_critic" + critic_path_obj.suffix)))

        deduplicated = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduplicated.append(candidate)
        return deduplicated

    def _load_actor_checkpoint_with_fallback(self):
        last_error = None
        for candidate_path in self._actor_candidate_paths():
            if os.path.exists(candidate_path) is not True:
                continue
            try:
                self._load_state_dict_from_path(self.actor, candidate_path)
                if candidate_path != self.actor_path:
                    print(
                        f"[ppo] actor checkpoint fallback applied: {self.actor_path} -> {candidate_path}"
                    )
                    self.actor_path = candidate_path
                return True
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        return False

    def _load_critic_checkpoint_with_fallback(self):
        last_error = None
        for candidate_path in self._critic_candidate_paths():
            if os.path.exists(candidate_path) is not True:
                continue
            try:
                self._load_state_dict_from_path(self.critic, candidate_path)
                if candidate_path != self.critic_path:
                    print(
                        f"[ppo] critic checkpoint fallback applied: {self.critic_path} -> {candidate_path}"
                    )
                    self.critic_path = candidate_path
                return True
            except Exception as error:
                last_error = error

        if last_error is not None:
            raise last_error
        return False

    def reset_state_history(self):
        self.state_history = {
            "opponent_x": deque(maxlen=3),
            "opponent_action_name": deque(maxlen=3),
            "opponent_jump": deque(maxlen=3),
        }

    def get_state_history_context(self):
        return {
            "opponent_x_history": list(self.state_history["opponent_x"]),
            "opponent_action_history": list(self.state_history["opponent_action_name"]),
            "opponent_jump_history": list(self.state_history["opponent_jump"]),
        }

    def update_state_history(self, state_mat):
        raw = state_mat.get("raw", {}) if isinstance(state_mat, dict) else {}
        opponent_raw = raw.get("opponent", {}) if isinstance(raw, dict) else {}
        rally_step = int(raw.get("rally_step", 0)) if isinstance(raw, dict) else 0

        if rally_step <= 1:
            self.reset_state_history()

        opponent_x = float(opponent_raw.get("x", 0.0))
        opponent_action_name = str(opponent_raw.get("action_name", "normal"))
        opponent_state = str(opponent_raw.get("state", "normal"))

        is_jump = int(
            opponent_state == "jump"
            or opponent_action_name in ("jump", "jump_forward", "jump_backward")
        )

        self.state_history["opponent_x"].append(opponent_x)
        self.state_history["opponent_action_name"].append(opponent_action_name)
        self.state_history["opponent_jump"].append(is_jump)
