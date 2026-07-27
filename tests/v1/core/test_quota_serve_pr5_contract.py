# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.quota_serve.collector import is_useful_eviction


def _event(**overrides: object) -> dict[str, object]:
    event = {
        "trigger_workload": "longctx",
        "evicted_workload": "chat",
        "is_cross_workload": True,
        "eviction_time": 100.0,
        "useful_counted": False,
    }
    event.update(overrides)
    return event


def test_pr5_useful_eviction_contract() -> None:
    kwargs = {"is_cache_miss": True, "now": 150.0, "shadow_ttl_sec": 120.0}

    assert is_useful_eviction(_event(), "chat", **kwargs)
    assert not is_useful_eviction(
        _event(trigger_workload="chat", is_cross_workload=False),
        "chat",
        **kwargs,
    )
    assert not is_useful_eviction(_event(), "agent", **kwargs)
    assert not is_useful_eviction(_event(useful_counted=True), "chat", **kwargs)
    assert not is_useful_eviction(
        _event(), "chat", is_cache_miss=False, now=150.0, shadow_ttl_sec=120.0
    )
    assert not is_useful_eviction(
        _event(), "chat", is_cache_miss=True, now=221.0, shadow_ttl_sec=120.0
    )
