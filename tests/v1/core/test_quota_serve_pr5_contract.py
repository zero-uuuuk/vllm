# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

from vllm.quota_serve.collector import (
    QuotaServeCollector,
    is_useful_eviction,
)
from vllm.v1.core.block_pool import BlockPool


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


def test_pr5_reuse_hook_receives_request(monkeypatch) -> None:
    class FakeCollector:
        def __init__(self) -> None:
            self.calls = []

        def on_block_reused(self, event, request) -> None:
            self.calls.append((event, request))

    pool = BlockPool.__new__(BlockPool)
    pool._pending_evictions = {b"hash": [_event()]}
    pool._pending_evictions_count = 1
    pool.metrics_collector = FakeCollector()
    request = object()

    monkeypatch.setattr(
        "vllm.v1.core.block_pool._log_eviction_event", lambda event: None
    )
    pool._complete_pending_reuse(b"hash", request)

    assert pool.metrics_collector.calls[0][1] is request


def test_pr5_window_counts_useful_once_and_expires() -> None:
    collector = QuotaServeCollector(
        sample_rate=1.0, shadow_ttl_sec=10, window_size=2
    )
    event = _event(eviction_time=100.0)

    with patch("vllm.quota_serve.collector.time.time", return_value=100.0):
        collector.on_eviction_recorded(event)
    with patch("vllm.quota_serve.collector.time.time", return_value=105.0):
        request = type("Request", (), {"request_id": "chat_smoke_0"})()
        collector.on_block_reused(event, request)
        collector.on_block_reused(event, request)

    with patch("vllm.quota_serve.collector.time.time", return_value=105.0):
        snapshot = collector.useful_eviction_snapshot()["chat"]
    assert snapshot["sample_count"] == 1
    assert snapshot["useful_evictions"] == 1
    assert snapshot["useful_eviction_ratio"] == 1.0

    with patch("vllm.quota_serve.collector.time.time", return_value=111.0):
        expired_snapshot = collector.useful_eviction_snapshot()["chat"]
    assert expired_snapshot["sample_count"] == 0
    assert expired_snapshot["useful_evictions"] == 0
    assert expired_snapshot["shadow_expired"] == 1


def test_pr5_reuse_counts_only_matching_cross_workload() -> None:
    collector = QuotaServeCollector(
        sample_rate=1.0, shadow_ttl_sec=120, window_size=10
    )
    self_event = _event(
        trigger_workload="chat",
        evicted_workload="chat",
        is_cross_workload=False,
    )
    wrong_workload_event = _event(eviction_time=101.0)

    with patch("vllm.quota_serve.collector.time.time", return_value=105.0):
        collector.on_eviction_recorded(self_event)
        collector.on_block_reused(
            self_event,
            type("Request", (), {"request_id": "chat_smoke_0"})(),
        )
        collector.on_eviction_recorded(wrong_workload_event)
        collector.on_block_reused(
            wrong_workload_event,
            type("Request", (), {"request_id": "agent_smoke_0"})(),
        )
        snapshot = collector.useful_eviction_snapshot()["chat"]
        assert snapshot["sample_count"] == 1
        assert snapshot["useful_evictions"] == 0
        assert snapshot["useful_eviction_ratio"] == 0.0


def test_pr5_window_drops_oldest_event() -> None:
    collector = QuotaServeCollector(
        sample_rate=1.0, shadow_ttl_sec=120, window_size=2
    )

    with patch("vllm.quota_serve.collector.time.time", return_value=102.0):
        for eviction_time in (100.0, 101.0, 102.0):
            collector.on_eviction_recorded(
                _event(eviction_time=eviction_time)
            )

        snapshot = collector.useful_eviction_snapshot()["chat"]

    assert snapshot["sample_count"] == 2
    assert snapshot["cross_workload_evictions"] == 2
    assert snapshot["useful_evictions"] == 0
    assert snapshot["shadow_dropped"] == 1


def test_pr5_signal_log_respects_tick_and_flushes(tmp_path) -> None:
    log_path = tmp_path / "quota_serve.jsonl"
    collector = QuotaServeCollector(
        sample_rate=1.0,
        shadow_ttl_sec=120,
        window_size=2,
        tick_sec=10,
        log_path=str(log_path),
    )

    with patch("vllm.quota_serve.collector.time.time", return_value=100.0):
        collector.on_eviction_recorded(_event(eviction_time=100.0))

    with patch("vllm.quota_serve.collector.time.time", return_value=105.0):
        assert collector.maybe_log_signal() == 0

    with patch("vllm.quota_serve.collector.time.time", return_value=111.0):
        assert collector.maybe_log_signal() == 1
        assert collector.flush_signal_log() == 1

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert records[0]["type"] == "useful_eviction_signal"
    assert records[0]["workload"] == "chat"
    assert records[0]["sample_count"] == 1
    assert records[0]["useful_eviction_ratio"] == 0.0
