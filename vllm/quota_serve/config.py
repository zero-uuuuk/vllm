# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuotaServe config schema + loader (PR 1).

QUOTASERVE_IMPLEMENTATION_PLAN.md §5 기준.

이 모듈은 **순수 데이터 + 로딩/검증**만 담당한다. 실제 정책(victim 선택,
counter 등)은 이후 PR이 이 config를 읽어 동작한다. PR 1의 핵심 불변식:

    mode == "off" (또는 enabled == False)  ==>  baseline LRU와 동일

따라서 로더는 hot path를 건드리지 않고, off일 때는 어떤 워크로드 quota도
적용되지 않도록 `is_active`로 게이트한다(§5.3 parity test).

우선순위(override 순서):
    1. 함수 인자 path  > 2. env QUOTA_SERVE_CONFIG  로 YAML을 찾는다.
    YAML이 없으면 비활성(off, disabled) 기본값을 반환한다 → baseline.
    그 뒤 env QUOTA_SERVE_MODE / QUOTA_SERVE_LOG 가 있으면 덮어쓴다.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, get_args

import yaml

from vllm.logger import init_logger

logger = init_logger(__name__)

# 정책 모드. off는 baseline, dry_run~dynamic_full은 PR이 진행되며 켜진다.
#   off          : 정책 미적용 (baseline LRU). PR 1.
#   dry_run      : victim은 LRU 그대로, shadow 비교 로그만. PR 4에서 활성.
#   static       : floor/cap 고정 3-tier victim selection. PR 4.
#   dynamic_floor: floor만 feedback로 조정. PR 6.
#   dynamic_full : floor + cap feedback. PR 7.
QuotaServeMode = Literal["off", "dry_run", "static", "dynamic_floor", "dynamic_full"]
_VALID_MODES: frozenset[str] = frozenset(get_args(QuotaServeMode))

# 환경 변수 이름 (§5.2)
ENV_MODE = "QUOTA_SERVE_MODE"  # config의 mode를 override
ENV_CONFIG = "QUOTA_SERVE_CONFIG"  # yaml 경로
ENV_LOG = "QUOTA_SERVE_LOG"  # eviction/tick JSONL 로그 경로

# YAML 최상위 키
_TOP_KEY = "quota_serve"


@dataclass(frozen=True)
class WorkloadQuota:
    """워크로드 하나의 floor/cap 비율.

    비율은 evictable cached prefix block pool에 대한 상대값이다(절대 block 수가
    아니다). 절대값 변환(`ratio * total_block_pool`)은 runtime 양인
    `total_block_pool`이 필요하므로 PR 4(`_abs_floor`/`_abs_cap`, §8.2)에서
    한다. 여기서는 비율만 보관/검증한다.

    Attributes:
        floor_ratio: 보호 하한 비율. "이 workload의 cached prefix block을 최소
            이만큼은 보호한다." (DESIGN §4.1)
        cap_ratio: 누적 상한 비율. "이 이상 쌓지 않는다." (DESIGN §4.2)
    """

    floor_ratio: float
    cap_ratio: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.floor_ratio <= self.cap_ratio <= 1.0):
            raise ValueError(
                "WorkloadQuota는 0.0 <= floor_ratio <= cap_ratio <= 1.0 이어야 "
                f"한다. got floor_ratio={self.floor_ratio}, "
                f"cap_ratio={self.cap_ratio}"
            )


@dataclass
class QuotaServeConfig:
    """QuotaServe 전체 설정.

    기본값은 **비활성(baseline)**이다. yaml/env가 없으면 이 기본값이 그대로
    쓰여 mode=off parity가 보장된다.

    Attributes:
        enabled: 마스터 스위치. False면 mode와 무관하게 정책 미적용.
        mode: 정책 모드(§QuotaServeMode).
        scan_limit: victim 탐색 시 LRU 스캔 상한(PR 4 bounded scan, §8.2).
        tick_sec: dynamic feedback 루프 주기(PR 6/7, DESIGN §5).
        shadow_ttl_sec: shadow cache 항목 TTL(PR 5, §9.1).
        workloads: workload_id -> WorkloadQuota.
        log_path: eviction/tick JSONL 로그 경로(없으면 비활성).
    """

    enabled: bool = False
    mode: QuotaServeMode = "off"
    scan_limit: int = 256
    tick_sec: float = 30.0
    shadow_ttl_sec: float = 120.0
    workloads: dict[str, WorkloadQuota] = field(default_factory=dict)
    log_path: str | None = None

    @property
    def is_active(self) -> bool:
        """정책이 실제로 victim 선택을 바꾸는가.

        off거나 disabled면 False → caller는 baseline LRU 경로를 그대로 탄다.
        이 게이트가 PR 1 parity의 핵심이다.
        """
        return self.enabled and self.mode != "off"

    def quota_for(self, workload_id: str) -> WorkloadQuota:
        """workload의 quota를 반환. 미설정 workload는 무제약 기본값.

        `floor=0.0, cap=1.0`은 "보호 없음 + 상한 없음" = 사실상 quota 미적용이라,
        설정되지 않은 워크로드(예: ``unknown``)가 우연히 보호/제한받는 일을
        막는다.
        """
        return self.workloads.get(workload_id, _UNCONSTRAINED_QUOTA)

    def validate(self) -> None:
        """스키마 무결성 검사. 로딩 마지막 단계에서 호출한다."""
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"알 수 없는 mode={self.mode!r}. 허용: {sorted(_VALID_MODES)}"
            )
        if not (isinstance(self.scan_limit, int) and self.scan_limit > 0):
            raise ValueError(f"scan_limit는 양의 정수여야 한다. got {self.scan_limit}")
        if self.tick_sec <= 0:
            raise ValueError(f"tick_sec는 > 0 이어야 한다. got {self.tick_sec}")
        if self.shadow_ttl_sec <= 0:
            raise ValueError(
                f"shadow_ttl_sec는 > 0 이어야 한다. got {self.shadow_ttl_sec}"
            )
        # WorkloadQuota 자체는 __post_init__에서 이미 검증됨.


# "보호 없음 + 상한 없음" — 설정되지 않은 워크로드의 안전한 기본값.
_UNCONSTRAINED_QUOTA = WorkloadQuota(floor_ratio=0.0, cap_ratio=1.0)


def load_quota_serve_config(path: str | None = None) -> QuotaServeConfig:
    """QuotaServe config를 로드한다.

    Args:
        path: YAML 경로. None이면 env ``QUOTA_SERVE_CONFIG``를 본다.

    Returns:
        검증된 ``QuotaServeConfig``. YAML이 없으면 비활성(baseline) 기본값.
    """
    cfg_path = path or os.environ.get(ENV_CONFIG)

    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = _from_dict(raw)
        logger.info("QuotaServe config loaded from %s", cfg_path)
    else:
        if cfg_path:
            # 경로가 지정됐는데 파일이 없으면 조용히 넘기지 않는다(오타 방지).
            logger.warning(
                "QUOTA_SERVE_CONFIG=%s 를 찾을 수 없어 비활성(off) 기본값을 쓴다.",
                cfg_path,
            )
        config = QuotaServeConfig()

    # ----- env override (§5.2): YAML보다 우선 -----
    env_mode = os.environ.get(ENV_MODE)
    if env_mode:
        config.mode = env_mode  # type: ignore[assignment]  # validate에서 검사
        # mode를 명시적으로 켜면 enabled도 켠 것으로 본다(off는 그대로 off).
        if env_mode != "off":
            config.enabled = True

    env_log = os.environ.get(ENV_LOG)
    if env_log:
        config.log_path = env_log

    config.validate()
    logger.info(
        "QuotaServe: enabled=%s mode=%s active=%s workloads=%s",
        config.enabled,
        config.mode,
        config.is_active,
        sorted(config.workloads),
    )
    return config


def _from_dict(raw: dict) -> QuotaServeConfig:
    """파싱된 YAML dict를 QuotaServeConfig로 변환한다.

    최상위 ``quota_serve:`` 키가 있으면 그 안을 쓰고, 없으면 dict 자체를 쓴다.
    """
    body = raw.get(_TOP_KEY, raw) if isinstance(raw, dict) else {}

    workloads_raw = body.get("workloads") or {}
    workloads: dict[str, WorkloadQuota] = {}
    for name, wq in workloads_raw.items():
        if "floor_ratio" not in wq or "cap_ratio" not in wq:
            raise ValueError(
                f"workload {name!r}에는 floor_ratio와 cap_ratio가 모두 필요하다."
            )
        workloads[str(name)] = WorkloadQuota(
            floor_ratio=float(wq["floor_ratio"]),
            cap_ratio=float(wq["cap_ratio"]),
        )

    # 누락 필드는 dataclass 기본값을 그대로 사용한다.
    defaults = QuotaServeConfig()
    return QuotaServeConfig(
        enabled=bool(body.get("enabled", defaults.enabled)),
        mode=body.get("mode", defaults.mode),
        scan_limit=int(body.get("scan_limit", defaults.scan_limit)),
        tick_sec=float(body.get("tick_sec", defaults.tick_sec)),
        shadow_ttl_sec=float(body.get("shadow_ttl_sec", defaults.shadow_ttl_sec)),
        workloads=workloads,
        log_path=body.get("log_path", defaults.log_path),
    )
