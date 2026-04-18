import os

import numpy as np
import _20_model

from _20_model import raichu_qlearning


class RaichuQlearning:
    def __init__(self, conf, policy_name_for_play=None):
        self.conf = conf
        self.train_conf = self.get_train_configuration()

        default_policy_name = str(
            self.train_conf.get("flagship_policy", "million_volts")
        ).strip()
        if policy_name_for_play is not None:
            self.policy_name = str(policy_name_for_play).strip()
        else:
            policy_name = getattr(self.conf, "train_policy", None)
            self.policy_name = (
                default_policy_name
                if policy_name in (None, "")
                else str(policy_name).strip()
            )

        self.policy_path = os.path.join(
            _20_model.get_model_policy_dir(self.conf, self),
            self.policy_name + ".pt",
        )

        self.visit_counts = {}
        self.total_updates = 0

        if getattr(self.conf, "train_rewrite", False) is not True:
            try:
                payload = raichu_qlearning._02_qtable.load_qtable(self.policy_path)
                self.policy = payload["table"]
                self.visit_counts = {
                    tuple(key): int(value)
                    for key, value in payload.get("visit_counts", {}).items()
                }
                self.total_updates = int(payload.get("total_updates", 0))
            except Exception:
                self.policy = raichu_qlearning._02_qtable.create_qtable()
        else:
            self.policy = raichu_qlearning._02_qtable.create_qtable()

        self.epsilon = self._epsilon_from_updates(self.total_updates)

    def _epsilon_from_updates(self, update_count):
        epsilon = (
            float(self.train_conf["epsilon_start"])
            * (float(self.train_conf["epsilon_decay"]) ** int(update_count))
        )
        return max(float(self.train_conf["epsilon_end"]), epsilon)

    def _alpha_from_visit_count(self, visit_count):
        alpha_base = float(self.train_conf["alpha"])
        alpha_min = float(self.train_conf.get("alpha_min", alpha_base))
        alpha_decay = float(self.train_conf.get("alpha_decay", 0.0))
        scaled_count = max(0.0, float(visit_count) * alpha_decay)
        alpha = alpha_base / np.sqrt(1.0 + scaled_count)
        return max(alpha_min, alpha)

    def get_transition(self, env, state_mat):
        state = self.map_to_designed_state(state_mat)
        previous_raw = state_mat["raw"]
        action_internal = raichu_qlearning._06_algorithm.epsilon_greedy_action_selection(
            policy=self.policy,
            state=state,
            epsilon=self.epsilon,
        )
        action_env = self.map_to_designed_action(action_internal)

        score, state_next_mat, reward_next_mat, done = env.run(
            player=self.conf.train_side,
            run_type="ai",
            action=action_env,
        )

        state_next = self.map_to_designed_state(state_next_mat)
        reward_next = self.map_to_designed_reward(
            reward_next_mat,
            previous_raw=previous_raw,
            state_next_mat=state_next_mat,
        )

        transition = (state, action_internal, state_next, reward_next, done, score)
        self.update_epsilon()
        return transition, state_next_mat

    def update(self, transition):
        state, action, state_next, reward_next, done, _ = transition
        state = tuple(state)
        state_next = tuple(state_next)
        action_idx = int(np.argmax(np.asarray(action, dtype=float)))

        td_target = raichu_qlearning._06_algorithm.calculate_qtarget(
            policy=self.policy,
            reward=reward_next,
            state_next=state_next,
            gamma=self.train_conf["gamma"],
            done=done,
        )

        qvector = raichu_qlearning._02_qtable.get_qvector(self.policy, state)
        count_key = (state, action_idx)
        visit_count = int(self.visit_counts.get(count_key, 0))
        alpha = self._alpha_from_visit_count(visit_count)
        qvector[action_idx] = qvector[action_idx] + alpha * (td_target - qvector[action_idx])

        self.visit_counts[count_key] = visit_count + 1
        self.total_updates += 1

    def get_train_configuration(self):
        return raichu_qlearning._01_params.get_train_params()

    def update_epsilon(self):
        self.epsilon = raichu_qlearning._06_algorithm.decay_epsilon(
            epsilon_start=self.epsilon,
            epsilon_decay=self.train_conf["epsilon_decay"],
            epsilon_end=self.train_conf["epsilon_end"],
        )

    def map_to_designed_state(self, state_mat):
        state_custom = raichu_qlearning._03_state_design.calculate_state_key(state_mat)
        return tuple(state_custom)

    def map_to_designed_action(self, action_mat):
        return raichu_qlearning._04_action_space_design.map_internal_to_environment_action(
            action_mat
        )

    def map_to_designed_reward(
        self,
        reward_mat,
        state_mat=None,
        previous_raw=None,
        state_next_mat=None,
    ):
        reward_materials = dict(reward_mat)
        if previous_raw is not None:
            reward_materials["previous_raw"] = previous_raw
        elif state_mat is not None:
            reward_materials["previous_raw"] = state_mat["raw"]
        if state_next_mat is not None:
            reward_materials["next_raw"] = state_next_mat["raw"]
        return raichu_qlearning._05_reward_design.calculate_reward(reward_materials)

    def select_action(self, state_mat, epsilon=0.0):
        state = self.map_to_designed_state(state_mat)
        action_internal = raichu_qlearning._06_algorithm.epsilon_greedy_action_selection(
            policy=self.policy,
            state=state,
            epsilon=epsilon,
        )
        return self.map_to_designed_action(action_internal)

    def save(self):
        raichu_qlearning._02_qtable.save_qtable(
            self.policy,
            self.policy_path,
            metadata={
                "visit_counts": self.visit_counts,
                "total_updates": self.total_updates,
            },
        )
