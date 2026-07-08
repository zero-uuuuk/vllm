# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuotaServe config schema and loader.

PR1 is intentionally data-only: loading this module or parsing a config must
not change the KV-cache allocation or eviction path. Later PRs wire this into
the scheduler/collector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, cast

QuotaServeMode = Literal["off", "static", "dynamic"]

_ALLOWED_MODES: Final[set[str]] = {"off", "static", "dynamic"}
_ENV_CONFIG: Final[str] = "QUOTA_SERVE_CONFIG"
_ENV_LOG: Final[str] = "QUOTA_SERVE_LOG"
_ENV_MODE: Final[str] = "QUOTA_SERVE_MODE"
_UNKNOWN_WORKLOAD_QUOTA_RATIO: Final[float] = 1.0


@dataclass(frozen=True)
class WorkloadQuota:
    """Static quota target for one workload.

    quota_ratio is relative to quota_base_blocks. PR4 initially defines
    quota_base_blocks as the total number of KV-cache blocks.
    """

    quota_ratio: float

    def __post_init__(self) -> None:
        quota_ratio = _coerce_float("quota_ratio", self.quota_ratio)
        if quota_ratio < 0.0 or quota_ratio > 1.0:
            raise ValueError(
                "quota_ratio must be between 0 and 1 inclusive, "
                f"got {quota_ratio!r}"
            )
        object.__setattr__(self, "quota_ratio", quota_ratio)


@dataclass(frozen=True)
class QuotaServeConfig:
    """QuotaServe runtime config.

    `is_active` is the PR1 parity gate. If it is false, callers must keep the
    baseline LRU path. `victim_selection_active` is stricter and becomes true
    only for modes that are allowed to change eviction order.
    """

    enabled: bool = False
    mode: QuotaServeMode = "off"
    tick_sec: int = 30
    shadow_ttl_sec: int = 120
    workloads: Mapping[str, WorkloadQuota] = field(default_factory=dict)
    log_path: str | None = None

    def __post_init__(self) -> None:
        enabled = _coerce_bool("enabled", self.enabled)
        mode = _coerce_mode(self.mode)
        tick_sec = _coerce_positive_int("tick_sec", self.tick_sec)
        shadow_ttl_sec = _coerce_positive_int(
            "shadow_ttl_sec", self.shadow_ttl_sec
        )
        workloads = _coerce_workloads(self.workloads)
        log_path = _coerce_optional_str("log_path", self.log_path)

        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "tick_sec", tick_sec)
        object.__setattr__(self, "shadow_ttl_sec", shadow_ttl_sec)
        object.__setattr__(self, "workloads", workloads)
        object.__setattr__(self, "log_path", log_path)

    @property
    def is_active(self) -> bool:
        """Whether QuotaServe observation/collector code may be enabled."""
        return self.enabled and self.mode != "off"

    @property
    def victim_selection_active(self) -> bool:
        """Whether QuotaServe may change victim selection order."""
        return self.enabled and self.mode in ("static", "dynamic")

    def quota_for(self, workload: str | None) -> WorkloadQuota:
        """Return the quota config for a workload.

        Unknown workloads use quota_ratio=1.0 so they are effectively excluded
        from over-quota victim selection in PR4.
        """
        if workload is None:
            return WorkloadQuota(_UNKNOWN_WORKLOAD_QUOTA_RATIO)
        return self.workloads.get(
            workload, WorkloadQuota(_UNKNOWN_WORKLOAD_QUOTA_RATIO)
        )

    def validate(self) -> "QuotaServeConfig":
        """Return self after dataclass post-init validation.

        Kept as an explicit API because the PR1 roadmap calls out a validate()
        entry point.
        """
        return self

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "QuotaServeConfig":
        """Build a config object from the `quota_serve` YAML section."""
        if not isinstance(data, Mapping):
            raise TypeError("QuotaServe config must be a mapping")

        allowed_keys = {
            "enabled",
            "log_path",
            "mode",
            "shadow_ttl_sec",
            "tick_sec",
            "workloads",
        }
        unknown_keys = sorted(set(data) - allowed_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown QuotaServe config key(s): "
                + ", ".join(str(key) for key in unknown_keys)
            )

        return cls(
            enabled=data.get("enabled", False),
            mode=data.get("mode", "off"),
            tick_sec=data.get("tick_sec", 30),
            shadow_ttl_sec=data.get("shadow_ttl_sec", 120),
            workloads=data.get("workloads", {}),
            log_path=data.get("log_path"),
        )


def load_quota_serve_config(
    path: str | os.PathLike[str] | None = None,
) -> QuotaServeConfig:
    """Load QuotaServe config from path/env and apply env overrides.

    Priority:
    1. explicit `path`
    2. QUOTA_SERVE_CONFIG
    3. disabled default config

    Then QUOTA_SERVE_MODE and QUOTA_SERVE_LOG override the loaded values.
    """
    config_path = path if path is not None else os.environ.get(_ENV_CONFIG)

    if config_path:
        config = QuotaServeConfig.from_mapping(_load_config_section(config_path))
    else:
        config = QuotaServeConfig()

    return _apply_env_overrides(config)


def _load_config_section(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    import yaml

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    if not isinstance(raw_config, Mapping):
        raise TypeError(
            "QuotaServe config file must contain a YAML mapping, "
            f"got {type(raw_config).__name__}"
        )

    section = raw_config.get("quota_serve", raw_config)
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise TypeError(
            "quota_serve section must be a mapping, "
            f"got {type(section).__name__}"
        )
    return section


def _apply_env_overrides(config: QuotaServeConfig) -> QuotaServeConfig:
    updates: dict[str, Any] = {}

    env_mode = os.environ.get(_ENV_MODE)
    if env_mode is not None and env_mode.strip():
        mode = env_mode.strip()
        updates["mode"] = mode
        if mode != "off":
            updates["enabled"] = True

    env_log = os.environ.get(_ENV_LOG)
    if env_log is not None:
        updates["log_path"] = env_log.strip() or None

    if not updates:
        return config
    return replace(config, **updates)


def _coerce_workloads(
    workloads: Mapping[str, WorkloadQuota | Mapping[str, Any]],
) -> Mapping[str, WorkloadQuota]:
    if not isinstance(workloads, Mapping):
        raise TypeError("workloads must be a mapping")

    converted: dict[str, WorkloadQuota] = {}
    for workload, quota in workloads.items():
        if not isinstance(workload, str) or not workload:
            raise ValueError(f"workload name must be a non-empty string: {workload!r}")

        if isinstance(quota, WorkloadQuota):
            converted[workload] = quota
            continue

        if not isinstance(quota, Mapping):
            raise TypeError(
                f"workloads.{workload} must be a mapping, "
                f"got {type(quota).__name__}"
            )
        if "quota_ratio" not in quota:
            raise ValueError(f"workloads.{workload}.quota_ratio is required")
        converted[workload] = WorkloadQuota(quota["quota_ratio"])

    return MappingProxyType(converted)


def _coerce_mode(mode: Any) -> QuotaServeMode:
    if not isinstance(mode, str):
        raise TypeError(f"mode must be a string, got {type(mode).__name__}")
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            "mode must be one of "
            + ", ".join(sorted(_ALLOWED_MODES))
            + f", got {mode!r}"
        )
    return cast(QuotaServeMode, mode)


def _coerce_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _coerce_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _coerce_optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None, got {type(value).__name__}")
    return value
