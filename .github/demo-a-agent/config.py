"""Configuration loader for demo_a_foundation.

This module separates two configuration concerns:

1. Structural / non-secret config (paths, defaults) — loaded from
   ``config.json`` next to this file. Already used by the harness
   (``materials.resolve_path`` for ``{repo_snapshot}`` etc.).

2. Runtime / secret config (GitHub token, LLM base URL / model / API key /
   timeouts) — loaded from a dotenv file (``DEMO_A_ENV_FILE`` if set,
   else ``.env`` next to ``config.json``). Values can be overridden by
   real environment variables of the same name. Tokens and API keys are
   NEVER printed or persisted to disk by this module; their presence is
   reported only as ``filled: True/False``.

A configuration is reported as:

  - ``structure_valid``     — keys present, types valid, no missing required ones
  - ``secrets_filled``      — credentials (GitHub / LLM API key) non-empty
  - ``secrets_missing``     — list of credential keys still empty

A missing credential is never a hard error for the offline harness; the
LLM- and GitHub-touching tools simply stay disabled. Each tool checks
its own dependency at runtime and surfaces a structured error.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent

# Schema version, bumped on breaking config changes.
CONFIG_SCHEMA = "demo_a_foundation/config/2"

# Runtime configuration keys (dotenv / env-driven). Order is preserved
# for stable output.
RUNTIME_KEYS: tuple[str, ...] = (
    "DEMO_A_LLM_BASE_URL",
    "DEMO_A_LLM_MODEL",
    "DEMO_A_LLM_API_KEY",
    "DEMO_A_LLM_TIMEOUT_SECONDS",
    "DEMO_A_LLM_MAX_OUTPUT_TOKENS",
    "DEMO_A_AGENT_MAX_STEPS",
    "DEMO_A_MAX_REQUEST_CHARS",
    "DEMO_A_GITHUB_TOKEN",
)

# Keys whose values are credentials. Their filled/empty status is
# reported but the value itself is never returned by the loader.
SECRET_KEYS: frozenset[str] = frozenset(
    {"DEMO_A_LLM_API_KEY", "DEMO_A_GITHUB_TOKEN"}
)

# Keys expected to be parseable as integers.
INT_KEYS: frozenset[str] = frozenset(
    {"DEMO_A_LLM_TIMEOUT_SECONDS", "DEMO_A_LLM_MAX_OUTPUT_TOKENS", "DEMO_A_AGENT_MAX_STEPS", "DEMO_A_MAX_REQUEST_CHARS"}
)

# Defaults applied when a runtime key is absent (only for non-secret keys).
DEFAULTS: dict[str, str] = {
    "DEMO_A_LLM_BASE_URL": "https://api.siliconflow.cn/v1",
    "DEMO_A_LLM_MODEL": "deepseek-ai/DeepSeek-V4-Flash",
    "DEMO_A_LLM_TIMEOUT_SECONDS": "120",
    "DEMO_A_LLM_MAX_OUTPUT_TOKENS": "4096",
    "DEMO_A_AGENT_MAX_STEPS": "8",
    "DEMO_A_MAX_REQUEST_CHARS": "30000",
}


@dataclass
class RuntimeConfig:
    """A validated, frozen view of the runtime config."""

    schema: str = CONFIG_SCHEMA
    values: dict[str, str] = field(default_factory=dict)
    structure_valid: bool = True
    secrets_filled: dict[str, bool] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def get_int(self, key: str) -> int | None:
        raw = self.values.get(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def is_filled(self, key: str) -> bool:
        return bool(self.secrets_filled.get(key, False))

    def public_view(self) -> dict:
        """A redacted view safe to log/persist (no secret values)."""
        return {
            "schema": self.schema,
            "structure_valid": self.structure_valid,
            "values": {
                k: v
                for k, v in self.values.items()
                if k not in SECRET_KEYS
            },
            "secrets_filled": dict(self.secrets_filled),
            "diagnostics": list(self.diagnostics),
        }


def _parse_dotenv(text: str) -> dict[str, str]:
    """Parse a minimal KEY=VALUE dotenv file. No quotes, no expansion, no
    multi-line. Lines starting with '#' (after optional whitespace) and
    blank lines are ignored. Inline '#' comments are not supported on
    purpose, to keep the parser deterministic and dependency-free.
    """
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        out[key] = value
    return out


def _load_dotenv_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        return _parse_dotenv(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def load_structural_config(path: str | Path | None = None) -> dict:
    """Load the structural config (config.json). Returns {} when missing."""
    target = Path(path) if path else DEMO_ROOT / "config.json"
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Preserve config_dir so materials.resolve_path works the same way.
    data.setdefault("config_dir", str(DEMO_ROOT))
    return data


def load_runtime_config(
    env_file: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> RuntimeConfig:
    """Load and validate runtime configuration.

    Order of precedence (later overrides earlier):
      1. DEFAULTS for non-secret keys
      2. dotenv file (DEMO_A_ENV_FILE if set, else .env)
      3. process environment (os.environ when env is None)

    Secret values are never returned by this loader; only their
    filled/empty status is reported.
    """
    env_file_from_env = os.environ.get("DEMO_A_ENV_FILE")
    if env_file:
        file_path = Path(env_file)
    elif env_file_from_env:
        file_path = Path(env_file_from_env)
    else:
        file_path = DEMO_ROOT / ".env"
    file_values = _load_dotenv_file(file_path)
    real_env = os.environ if env is None else env

    diagnostics: list[str] = []
    values: dict[str, str] = {}
    secrets_filled: dict[str, bool] = {}

    for key in RUNTIME_KEYS:
        chosen: str | None = None
        if key in real_env and real_env[key] != "":
            chosen = real_env[key]
        elif file_values.get(key, "") != "":
            chosen = file_values[key]
        elif key in DEFAULTS:
            chosen = DEFAULTS[key]

        if key in SECRET_KEYS:
            filled = bool(chosen and chosen.strip())
            secrets_filled[key] = filled
            # Keep "filled" boolean but DO NOT store the secret value.
            values[key] = "" if filled is False else "__filled__"
        else:
            if chosen is None or chosen == "":
                diagnostics.append(f"runtime key {key!r} not set and has no default")
                values[key] = ""
            else:
                if key in INT_KEYS:
                    try:
                        int(chosen)
                    except ValueError:
                        diagnostics.append(
                            f"runtime key {key!r} must be an integer; got {chosen!r}"
                        )
                        chosen = DEFAULTS.get(key, "")
                values[key] = chosen

    structure_valid = not any(
        d.startswith("runtime key") for d in diagnostics
    )

    return RuntimeConfig(
        schema=CONFIG_SCHEMA,
        values=values,
        structure_valid=structure_valid,
        secrets_filled=secrets_filled,
        diagnostics=diagnostics,
    )


def load_all(
    env_file: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """Convenience: load both structural and runtime config together."""
    structural = load_structural_config()
    runtime = load_runtime_config(env_file=env_file, env=env)
    return {
        "schema": CONFIG_SCHEMA,
        "structural": structural,
        "runtime": runtime.public_view(),
    }


def load_secrets(
    env_file: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the ACTUAL secret values for runtime API use only.

    This is the one narrow place a credential's real value leaves config.py,
    and it is intended solely for the network client (LLM provider / GitHub
    fetcher) to build an Authorization header. It must NEVER be passed to
    public_view(), logged, written to a report, or printed. Same precedence
    as load_runtime_config: process env > dotenv file. Keys in SECRET_KEYS
    that are empty anywhere are omitted from the result.
    """
    env_file_from_env = os.environ.get("DEMO_A_ENV_FILE")
    if env_file:
        file_path = Path(env_file)
    elif env_file_from_env:
        file_path = Path(env_file_from_env)
    else:
        file_path = DEMO_ROOT / ".env"
    file_values = _load_dotenv_file(file_path)
    real_env = os.environ if env is None else env

    secrets: dict[str, str] = {}
    for key in SECRET_KEYS:
        value = ""
        if key in real_env and real_env[key] != "":
            value = real_env[key]
        elif file_values.get(key, "") != "":
            value = file_values[key]
        if value.strip():
            secrets[key] = value
    return secrets


__all__ = [
    "CONFIG_SCHEMA",
    "RUNTIME_KEYS",
    "SECRET_KEYS",
    "DEFAULTS",
    "RuntimeConfig",
    "load_structural_config",
    "load_runtime_config",
    "load_all",
    "load_secrets",
    "DEMO_ROOT",
]
