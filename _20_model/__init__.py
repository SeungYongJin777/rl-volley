from importlib import import_module
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
import inspect
from pathlib import Path
import re
import sys


MODEL_ROOT_DIR = Path(__file__).resolve().parent


def _is_model_package_dir(path):
    return (
        path.is_dir()
        and not path.name.startswith("__")
        and (path / "__init__.py").exists()
        and (path / "_00_model.py").exists()
    )


def _normalize_model_name(raw_name):
    normalized_name = re.sub(
        r"[^0-9a-zA-Z_]+",
        "_",
        str(raw_name).strip().lower(),
    )
    normalized_name = re.sub(r"_+", "_", normalized_name).strip("_")

    if normalized_name == "":
        raise ValueError(f"failed to normalize model name: {raw_name}")

    if normalized_name[0].isdigit():
        normalized_name = "model_" + normalized_name

    return normalized_name


def _discover_model_registry():
    model_registry = {}

    for path in sorted(MODEL_ROOT_DIR.iterdir()):
        if _is_model_package_dir(path) is not True:
            continue

        base_model_name = _normalize_model_name(path.name)
        model_name = base_model_name
        suffix_idx = 2

        while model_name in model_registry:
            model_name = f"{base_model_name}_{suffix_idx}"
            suffix_idx += 1

        model_registry[model_name] = path

    return model_registry


def get_available_model_names():
    return list(_discover_model_registry().keys())


def get_model_package_dir(model_or_algorithm):
    model_name = resolve_model_name(model_or_algorithm)
    model_registry = _discover_model_registry()

    if model_name not in model_registry:
        raise ValueError(f"unknown algorithm: {model_or_algorithm}")

    return model_registry[model_name]


def _replace_cloned_model_references(model_name, model_dir):
    model_registry = _discover_model_registry()
    referenced_model_names = {}
    source_files = sorted(model_dir.glob("*.py"))

    for source_path in source_files:
        source_text = source_path.read_text(encoding="utf-8")

        imported_model_names = re.findall(
            r"from _20_model import ([A-Za-z0-9_]+)",
            source_text,
        )
        for imported_model_name in imported_model_names:
            normalized_name = _normalize_model_name(imported_model_name)
            if normalized_name == model_name:
                continue
            if normalized_name in model_registry:
                referenced_model_names[normalized_name] = (
                    referenced_model_names.get(normalized_name, 0) + 1
                )

        for candidate_model_name in model_registry:
            if candidate_model_name == model_name:
                continue

            reference_count = len(
                re.findall(
                    rf"\b{re.escape(candidate_model_name)}\._",
                    source_text,
                )
            )
            reference_count += source_text.count(
                f"path_{candidate_model_name}_"
            )
            reference_count += source_text.count(
                f"_20_model.{candidate_model_name}."
            )

            if reference_count > 0:
                referenced_model_names[candidate_model_name] = (
                    referenced_model_names.get(candidate_model_name, 0)
                    + reference_count
                )

    referenced_model_names = {
        name: count
        for name, count in referenced_model_names.items()
        if count > 0
    }

    if len(referenced_model_names) != 1:
        return

    source_model_name = next(iter(referenced_model_names))
    if source_model_name == model_name:
        return

    for source_path in source_files:
        source_text = source_path.read_text(encoding="utf-8")
        updated_text = source_text

        updated_text = re.sub(
            rf"from _20_model import {re.escape(source_model_name)}\b",
            f"from _20_model import {model_name}",
            updated_text,
        )
        updated_text = re.sub(
            rf"\b{re.escape(source_model_name)}(?=\._)",
            model_name,
            updated_text,
        )
        updated_text = updated_text.replace(
            f"path_{source_model_name}_",
            f"path_{model_name}_",
        )
        updated_text = updated_text.replace(
            f"_20_model.{source_model_name}.",
            f"_20_model.{model_name}.",
        )

        if updated_text != source_text:
            source_path.write_text(updated_text, encoding="utf-8")


def _resolve_model_class(model_module, algorithm_name):
    normalized_algorithm_name = str(algorithm_name).replace("_", "").lower()
    candidates = []

    for object_name, object_value in inspect.getmembers(
        model_module, inspect.isclass
    ):
        if object_value.__module__ != model_module.__name__:
            continue
        if not all(
            hasattr(object_value, method_name)
            for method_name in ("get_transition", "update", "save")
        ):
            continue

        candidates.append(object_value)
        normalized_object_name = object_name.replace("_", "").lower()
        if normalized_object_name == normalized_algorithm_name:
            return object_value

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 0:
        return candidates[0]

    raise AttributeError(
        f"failed to resolve model class for algorithm: {algorithm_name}"
    )


def import_model_package(algorithm_name):
    model_name = resolve_model_name(algorithm_name)
    model_dir = get_model_package_dir(model_name)
    module_name = f"{__name__}.{model_name}"

    if module_name in sys.modules:
        model_package = sys.modules[module_name]
        globals()[model_name] = model_package
        return model_package

    _replace_cloned_model_references(model_name, model_dir)

    spec = spec_from_file_location(
        module_name,
        model_dir / "__init__.py",
        submodule_search_locations=[str(model_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load model package: {algorithm_name}")

    model_package = module_from_spec(spec)
    sys.modules[module_name] = model_package
    globals()[model_name] = model_package
    spec.loader.exec_module(model_package)
    return model_package


def create_model(conf, algorithm_name, policy_name_for_play=None):
    model_name = resolve_model_name(algorithm_name)
    import_model_package(model_name)

    model_module = import_module(f"{__name__}.{model_name}._00_model")
    model_class = _resolve_model_class(model_module, model_name)
    model = model_class(
        conf,
        policy_name_for_play=policy_name_for_play,
    )
    return model


def resolve_model_name(model_or_algorithm):
    if isinstance(model_or_algorithm, str):
        normalized_name = _normalize_model_name(model_or_algorithm)
        if normalized_name in _discover_model_registry():
            return normalized_name
        raise ValueError(f"unknown algorithm: {model_or_algorithm}")

    if inspect.isclass(model_or_algorithm):
        module_name = model_or_algorithm.__module__
    else:
        module_name = model_or_algorithm.__class__.__module__

    module_parts = str(module_name).split(".")
    if len(module_parts) >= 2 and module_parts[0] == __name__:
        return str(module_parts[1]).strip().lower()

    raise ValueError(
        f"failed to resolve model name from: {model_or_algorithm}"
    )


def get_model_output_dir(conf, model_or_algorithm):
    model_name = resolve_model_name(model_or_algorithm)
    return getattr(conf, f"path_{model_name}_output")


def get_model_policy_dir(conf, model_or_algorithm):
    model_name = resolve_model_name(model_or_algorithm)
    return getattr(conf, f"path_{model_name}_policy")


def __getattr__(name):
    normalized_name = _normalize_model_name(name)
    if normalized_name in _discover_model_registry():
        return import_model_package(normalized_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + get_available_model_names())


__all__ = [
    "create_model",
    "get_available_model_names",
    "get_model_output_dir",
    "get_model_package_dir",
    "get_model_policy_dir",
    "import_model_package",
    "resolve_model_name",
] + get_available_model_names()
