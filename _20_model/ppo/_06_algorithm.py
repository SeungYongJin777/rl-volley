import copy
import random
from pathlib import Path

import numpy as np
import torch

import _20_model
from _20_model import ppo


def stochastic_action_selection(policy, state):
    """====================================================================================================
    ## Select Action by Stochastic Policy
    ===================================================================================================="""
    device = next(policy.parameters()).device

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(state).squeeze(0)
        action_probs = torch.softmax(logits, dim=0)
        action_idx = int(torch.multinomial(action_probs, num_samples=1).item())
        selected_action_prob = action_probs[action_idx]
        selected_action_prob = torch.clamp(selected_action_prob, min=1e-8)
        selected_log_prob = float(torch.log(selected_action_prob).item())

    dim_action = int(logits.shape[0])
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0
    return action, action_idx, selected_log_prob


def deterministic_action_selection(policy, state):
    """====================================================================================================
    ## Select Action by Deterministic Policy
    ===================================================================================================="""
    device = next(policy.parameters()).device

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = policy(state).squeeze(0)
        action_idx = int(torch.argmax(logits).item())

    dim_action = int(logits.shape[0])
    action = np.zeros(dim_action, dtype=np.float32)
    action[action_idx] = 1.0
    return action, action_idx


class RandomPolicyOpponent:
    def __init__(self, action_dim, name="random"):
        self.action_dim = int(action_dim)
        self.policy_name = str(name)

    def select_action(self, state_mat, epsilon=0.0):
        del state_mat, epsilon
        action_idx = random.randrange(self.action_dim)
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[action_idx] = 1.0
        return action

    def reset_state_history(self):
        return None


class OpponentPool:
    def __init__(self, conf, model_train):
        self.conf = conf
        self.model_train = model_train
        self.train_conf = getattr(model_train, "train_conf", {}) or {}
        self.train_side = str(getattr(conf, "train_side", "1p")).strip().lower()
        self.dim_action = len(ppo._04_action_space_design.action_mask())

        self.groups = {
            "rule": [],
            "random": [],
            "snapshot": [],
            "external": [],
        }
        self._registered_paths = set()
        self._cached_opponent = None
        self._cached_bucket = None
        self.latest_eval_results = {}
        self.available_model_names = set(_20_model.get_available_model_names())
        self.initialization_events = []

        self._build_initial_pool()

    def _build_initial_pool(self):
        if self._flag("use_rule_opponent", True):
            self.add_rule_opponent()

        if self._flag("use_random_opponent", True):
            self.add_random_opponent()

        if self._flag("use_external_opponents", False):
            for external_path in self.train_conf.get("external_opponent_paths", []):
                self.add_external_opponent(external_path)

        if self._flag("use_snapshot_opponents", True):
            actor_path = getattr(self.model_train, "actor_path", None)
            if actor_path and Path(actor_path).exists() and getattr(self.conf, "train_rewrite", False) is not True:
                self.register_snapshot(
                    {
                        "episode": 0,
                        "name": f"{getattr(self.model_train, 'policy_name', 'ppo')}_bootstrap",
                        "actor_path": str(actor_path),
                        "critic_path": getattr(self.model_train, "critic_path", None),
                    }
                )

        self._record_event(
            "info",
            "opponent pool initialized: "
            + "rule={rule}, random={random}, snapshot={snapshot}, external={external}".format(
                **self.get_group_counts()
            ),
        )

    def _record_event(self, level, message):
        self.initialization_events.append(
            {
                "level": str(level).strip().lower(),
                "message": str(message),
            }
        )

    def _flag(self, key, default=False):
        return bool(self.train_conf.get(key, default))

    def _float_value(self, key, default=0.0):
        try:
            return float(self.train_conf.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _int_value(self, key, default=0):
        try:
            return int(self.train_conf.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _build_frozen_ppo(self, actor_path, name):
        return _20_model.create_model(
            self.conf,
            algorithm_name="ppo",
            policy_name_for_play=str(name),
            checkpoint_path=str(actor_path),
            fixed_for_opponent=True,
            eval_mode=True,
        )

    def _build_frozen_model(self, algorithm_name, model_path, policy_name):
        normalized_algorithm = str(algorithm_name).strip().lower()
        if normalized_algorithm == "ppo":
            return self._build_frozen_ppo(model_path, policy_name)

        # External opponents should always load an existing policy file.
        conf_for_opponent = copy.copy(self.conf)
        conf_for_opponent.train_rewrite = False
        model_path_obj = Path(str(model_path))
        setattr(
            conf_for_opponent,
            f"path_{normalized_algorithm}_policy",
            str(model_path_obj.parent),
        )

        return _20_model.create_model(
            conf_for_opponent,
            algorithm_name=normalized_algorithm,
            policy_name_for_play=str(policy_name),
        )

    def _resolve_path(self, path_like):
        candidate_path = Path(str(path_like)).expanduser()
        if candidate_path.exists():
            return candidate_path
        candidate_path_from_cwd = (Path.cwd() / candidate_path).resolve()
        if candidate_path_from_cwd.exists():
            return candidate_path_from_cwd
        return candidate_path

    def _infer_algorithm_from_path(self, model_path):
        for path_part in model_path.parts:
            normalized_part = str(path_part).strip().lower()
            if normalized_part in self.available_model_names:
                return normalized_part
        return None

    def _parse_external_spec(self, external_spec):
        raw_spec = str(external_spec).strip()
        explicit_algorithm = None
        path_part = raw_spec

        if ":" in raw_spec:
            prefix, _, suffix = raw_spec.partition(":")
            normalized_prefix = str(prefix).strip().lower()
            if normalized_prefix in self.available_model_names:
                explicit_algorithm = normalized_prefix
                path_part = str(suffix).strip()

        resolved_path = self._resolve_path(path_part)
        algorithm_name = explicit_algorithm
        if algorithm_name is None:
            algorithm_name = self._infer_algorithm_from_path(resolved_path)

        return algorithm_name, resolved_path

    def _register_entry(self, group_name, entry):
        self.groups[group_name].append(entry)

    def add_rule_opponent(self, name="rule"):
        self._register_entry(
            "rule",
            {
                "type": "rule",
                "name": str(name),
                "controller": "RULE",
                "model": "RULE",
                "source": "built_in_rule",
                "weight": 1.0,
            },
        )

    def add_random_opponent(self, name="random"):
        random_model = RandomPolicyOpponent(
            action_dim=self.dim_action,
            name=name,
        )
        self._register_entry(
            "random",
            {
                "type": "random",
                "name": str(name),
                "controller": "model",
                "model": random_model,
                "source": "uniform_random_policy",
                "weight": 1.0,
            },
        )

    def add_external_opponent(self, actor_path, name=None):
        raw_spec = str(actor_path)
        algorithm_name, actor_path = self._parse_external_spec(actor_path)
        if actor_path.exists() is not True:
            self._record_event(
                "warn",
                f"external skipped (missing file): {raw_spec}",
            )
            return None
        if algorithm_name is None:
            self._record_event(
                "warn",
                f"external skipped (cannot infer algorithm): {raw_spec}",
            )
            return None

        actor_path_key = str(actor_path.resolve())
        if actor_path_key in self._registered_paths:
            self._record_event(
                "info",
                f"external skipped (duplicate): {actor_path_key}",
            )
            return None

        if name is None:
            name = actor_path.stem

        try:
            frozen_model = self._build_frozen_model(
                algorithm_name=algorithm_name,
                model_path=actor_path_key,
                policy_name=name,
            )
        except Exception:
            self._record_event(
                "warn",
                "external skipped (load failed): "
                + f"{algorithm_name}:{actor_path_key}",
            )
            return None
        entry = {
            "type": "external",
            "name": str(name),
            "controller": "model",
            "model": frozen_model,
            "source": f"{algorithm_name}:{actor_path_key}",
            "weight": 1.0,
            "algorithm": algorithm_name,
        }
        self._registered_paths.add(actor_path_key)
        self._register_entry("external", entry)
        self._record_event(
            "info",
            f"external added: {algorithm_name}:{actor_path_key}",
        )
        return entry

    def register_snapshot(self, snapshot_info):
        actor_path = Path(str(snapshot_info["actor_path"]))
        if actor_path.exists() is not True:
            return None

        actor_path_key = str(actor_path.resolve())
        if actor_path_key in self._registered_paths:
            return None

        snapshot_name = str(snapshot_info.get("name") or actor_path.stem)
        frozen_model = self._build_frozen_ppo(actor_path_key, snapshot_name)
        entry = {
            "type": "snapshot",
            "name": snapshot_name,
            "controller": "model",
            "model": frozen_model,
            "source": actor_path_key,
            "weight": 1.0,
            "episode": int(snapshot_info.get("episode", 0)),
        }
        self._registered_paths.add(actor_path_key)
        self._register_entry("snapshot", entry)

        max_snapshot_size = max(1, self._int_value("max_snapshot_size", 5))
        while len(self.groups["snapshot"]) > max_snapshot_size:
            removed_entry = self.groups["snapshot"].pop(0)
            removed_source = str(removed_entry.get("source", ""))
            if removed_source in self._registered_paths:
                self._registered_paths.remove(removed_source)

        return entry

    def _active_group_specs(self):
        specs = []
        for group_name, ratio_key in (
            ("rule", "rule_ratio"),
            ("random", "random_ratio"),
            ("snapshot", "snapshot_ratio"),
            ("external", "external_ratio"),
        ):
            entries = self.groups[group_name]
            if len(entries) == 0:
                continue

            ratio = self._float_value(ratio_key, 0.0)
            if ratio <= 0.0:
                continue

            if self._flag("use_hard_negative_sampling", False):
                group_eval = self.latest_eval_results.get("group_win_rates", {})
                group_win_rate = float(group_eval.get(group_name, 0.5))
                ratio = ratio * (1.0 + max(0.0, 1.0 - group_win_rate))

            specs.append((group_name, ratio))

        if len(specs) == 0:
            fallback_entries = []
            for group_name, entries in self.groups.items():
                if len(entries) > 0:
                    fallback_entries.append((group_name, 1.0))
            return fallback_entries

        return specs

    def _sample_group_name(self):
        active_specs = self._active_group_specs()
        total_weight = sum(weight for _, weight in active_specs)
        if total_weight <= 0.0:
            return active_specs[0][0]

        pivot = random.random() * total_weight
        cumulative_weight = 0.0
        for group_name, weight in active_specs:
            cumulative_weight += weight
            if pivot <= cumulative_weight:
                return group_name
        return active_specs[-1][0]

    def sample_opponent(self, epi_idx):
        sampling_mode = str(self.train_conf.get("opponent_sampling_mode", "per_episode")).strip().lower()
        swap_interval = max(1, self._int_value("opponent_swap_interval", 1))
        episode_bucket = int(epi_idx) // swap_interval

        if sampling_mode == "fixed_interval":
            if self._cached_opponent is not None and self._cached_bucket == episode_bucket:
                return self._cached_opponent

        sampled_group = self._sample_group_name()
        sampled_entry = random.choice(self.groups[sampled_group])
        self._cached_opponent = sampled_entry
        self._cached_bucket = episode_bucket
        return sampled_entry

    def get_evaluation_entries(self):
        entries = []
        for group_name in ("rule", "random", "snapshot", "external"):
            entries.extend(self.groups[group_name])
        return entries

    def get_group_counts(self):
        return {
            "rule": len(self.groups["rule"]),
            "random": len(self.groups["random"]),
            "snapshot": len(self.groups["snapshot"]),
            "external": len(self.groups["external"]),
        }

    def consume_initialization_events(self):
        events = list(self.initialization_events)
        self.initialization_events = []
        return events


def create_opponent_pool(conf, model_train):
    return OpponentPool(conf, model_train)


def resolve_opponent_runtime(opponent_entry):
    return opponent_entry["model"]


def reset_model_runtime(model):
    if isinstance(model, str):
        return
    if hasattr(model, "action_next_mat"):
        model.action_next_mat = None
    if hasattr(model, "last_action_idx"):
        model.last_action_idx = None
    reset_function = getattr(model, "reset_state_history", None)
    if callable(reset_function):
        reset_function()


def evaluate_against_pool(conf, model_train, opponent_pool, env_factory, evaluation_side=None, update_pool_results=True):
    evaluation_entries = opponent_pool.get_evaluation_entries()
    if len(evaluation_entries) == 0:
        return {
            "evaluation_side": str(
                evaluation_side if evaluation_side is not None else getattr(conf, "train_side", "1p")
            ).strip().lower(),
            "overall_win_rate": 0.0,
            "worst_group_win_rate": 0.0,
            "group_win_rates": {},
            "entry_results": [],
        }

    evaluation_side_name = str(
        evaluation_side if evaluation_side is not None else getattr(conf, "train_side", "1p")
    ).strip().lower()
    evaluation_env = env_factory(conf)
    evaluation_results = []
    eval_episode_count = max(
        1,
        int(getattr(model_train, "train_conf", {}).get("eval_num_episode_per_opponent", 10)),
    )

    try:
        for opponent_entry in evaluation_entries:
            entry_wins = 0.0
            entry_score_diffs = []
            opponent_runtime = resolve_opponent_runtime(opponent_entry)

            for _ in range(eval_episode_count):
                reset_model_runtime(model_train)
                reset_model_runtime(opponent_runtime)

                if evaluation_side_name == "1p":
                    evaluation_env.set(
                        player1=model_train,
                        player2=opponent_runtime,
                        random_serve=conf.random_serve,
                        return_state=False,
                    )
                else:
                    evaluation_env.set(
                        player1=opponent_runtime,
                        player2=model_train,
                        random_serve=conf.random_serve,
                        return_state=False,
                    )

                while True:
                    play_result = evaluation_env.run_play_step()
                    if bool(play_result["done"]):
                        score = play_result["score"]
                        train_score = resolve_side_score(score, evaluation_side_name)
                        opponent_side = "1p" if evaluation_side_name == "2p" else "2p"
                        opponent_score = resolve_side_score(score, opponent_side)
                        entry_wins += float(train_score > opponent_score)
                        entry_score_diffs.append(float(train_score - opponent_score))
                        break

            total_eval = max(1, len(entry_score_diffs))
            evaluation_results.append(
                {
                    "name": opponent_entry["name"],
                    "type": opponent_entry["type"],
                    "source": opponent_entry["source"],
                    "win_rate": float(entry_wins / total_eval),
                    "avg_score_diff": float(sum(entry_score_diffs) / total_eval),
                }
            )
    finally:
        evaluation_env.close()

    group_win_rates = {}
    for group_name in ("rule", "random", "snapshot", "external"):
        group_rows = [
            row for row in evaluation_results if row["type"] == group_name
        ]
        if len(group_rows) == 0:
            continue
        group_win_rates[group_name] = float(
            sum(row["win_rate"] for row in group_rows) / len(group_rows)
        )

    overall_win_rate = float(
        sum(row["win_rate"] for row in evaluation_results) / len(evaluation_results)
    )
    worst_group_win_rate = overall_win_rate
    if len(group_win_rates) > 0:
        worst_group_win_rate = min(group_win_rates.values())

    output = {
        "evaluation_side": evaluation_side_name,
        "overall_win_rate": overall_win_rate,
        "worst_group_win_rate": float(worst_group_win_rate),
        "group_win_rates": group_win_rates,
        "entry_results": evaluation_results,
    }
    if update_pool_results:
        opponent_pool.latest_eval_results = output
    return output


def resolve_side_score(score, side):
    normalized_side = str(side).strip().lower()
    score_keys = ("player1", "p1")
    if normalized_side == "2p":
        score_keys = ("player2", "p2")

    for score_key in score_keys:
        if score_key in score:
            return float(score[score_key])
    return 0.0
