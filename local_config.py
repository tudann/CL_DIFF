"""Load optional <script>.local.yaml overrides next to a project script."""
import os


def local_override_path(script_file):
    root, _ = os.path.splitext(os.path.abspath(script_file))
    return f"{root}.local.yaml"


def load_yaml_mapping(path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for *.local.yaml. Install with: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Local config root must be a mapping: {path}")
    return data


def apply_local_overrides(defaults, script_file):
    """Merge <script>.local.yaml into argparse defaults if the file exists.

    CLI arguments still override both the script defaults and this file.
    """
    path = local_override_path(script_file)
    if not os.path.isfile(path):
        return defaults

    override = load_yaml_mapping(path)
    unknown = [key for key in override if key not in defaults]
    if unknown:
        print(f"Warning: ignored unknown keys in {path}: {unknown}")
    applied = {key: value for key, value in override.items() if key in defaults}
    defaults.update(applied)
    print(f"Loaded local overrides: {path} ({len(applied)} keys)")
    return defaults
