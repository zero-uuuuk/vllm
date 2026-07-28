# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuotaServe PR3 occupancy accounting and PR5 signal contract."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from vllm.quota_serve.workload import infer_workload
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request


@dataclass(slots=True)
class WorkloadState:
    """Runtime counters for one workload."""

    evictable_cached: int = 0


def is_useful_eviction(
    event: Mapping[str, object],
    requester_workload: str,
    *,
    is_cache_miss: bool,
    now: float,
    shadow_ttl_sec: float,
) -> bool:
    """Return whether a pending eviction satisfies the PR5-1 contract.

    The caller invokes this only after a request actually misses the cache
    and recomputes the evicted prefix.  The event is counted only when it is
    cross-workload, belongs to the requesting workload, is within the shadow
    TTL, and has not already been counted.
    """
    evictor = event.get("trigger_workload")
    victim = event.get("evicted_workload")
    eviction_time = event.get("eviction_time")

    if not is_cache_miss:
        return False
    if not isinstance(evictor, str) or not isinstance(victim, str):
        return False
    if (
        not evictor
        or evictor == victim
        or event.get("is_cross_workload") is not True
    ):
        return False
    if requester_workload != victim:
        return False
    if event.get("useful_counted", False) is True:
        return False
    if type(eviction_time) not in (int, float):
        return False

    age = now - eviction_time
    return 0 <= age <= shadow_ttl_sec


class QuotaServeCollector(KVCacheMetricsCollector):
    """Track owner-aware occupancy without changing victim selection.

    PR3 defines occupancy as the number of physical blocks owned by a
    workload that are both cached and currently evictable:

        block.ref_cnt == 0 and block.block_hash is not None

    Occupancy update flow:

        block state changes
        -> BlockPool calls a lifecycle hook
        -> _is_evictable_cached() evaluates the current state
        -> _sync_membership() updates the occupancy counter and block flag
        -> WorkloadState.evictable_cached stores the resulting occupancy

    The flag on ``KVCacheBlock`` makes every lifecycle hook idempotent. The
    lock protects the flag and its corresponding workload counter as one
    transition, so a concurrent hook cannot apply the same update twice.
    """

    def __init__(
        self,
        sample_rate: float = 0.01,
        *,
        collect_residency_metrics: bool = False,
        shadow_ttl_sec: float = 120,
        window_size: int = 1000,
    ) -> None:
        super().__init__(sample_rate)
        if shadow_ttl_sec <= 0:
            raise ValueError("shadow_ttl_sec must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._collect_residency_metrics = collect_residency_metrics
        self._shadow_ttl_sec = float(shadow_ttl_sec)
        self._window_size = window_size
        self.state: defaultdict[str, WorkloadState] = defaultdict(WorkloadState)
        self._quota_lock = Lock()
        self._signal_lock = Lock()
        self._signal_windows: defaultdict[
            str, deque[dict[str, object]]
        ] = defaultdict(deque)
        self._signal_event_ids: set[int] = set()
        self._useful_event_ids: set[int] = set()
        self._useful_evictions: defaultdict[str, int] = defaultdict(int)
        self._shadow_expired: defaultdict[str, int] = defaultdict(int)
        self._shadow_dropped: defaultdict[str, int] = defaultdict(int)

    @staticmethod
    def _owner(block: "KVCacheBlock") -> str:
        # Empty/None owner metadata can occur on paths without a request.
        return block.workload_tag or "unknown"

    @staticmethod
    def _is_evictable_cached(block: "KVCacheBlock") -> bool:
        return (
            not block.is_null
            and block.ref_cnt == 0
            and block.block_hash is not None
        )

    def _sync_membership(self, block: "KVCacheBlock") -> None:
        """Apply the current block state as one idempotent transition."""
        with self._quota_lock:
            should_be_counted = self._is_evictable_cached(block)

            if should_be_counted and not block.is_counted_as_evictable_cached:
                self.state[self._owner(block)].evictable_cached += 1
                block.is_counted_as_evictable_cached = True
            elif (
                not should_be_counted
                and block.is_counted_as_evictable_cached
            ):
                owner = self._owner(block)
                current = self.state[owner].evictable_cached
                if current <= 0:
                    raise RuntimeError(
                        "occupancy counter underflow for "
                        f"workload={owner!r}, block_id={block.block_id}"
                    )
                self.state[owner].evictable_cached = current - 1
                block.is_counted_as_evictable_cached = False

    def on_block_allocated(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Assign the new owner when a physical block is reused."""
        if self._collect_residency_metrics:
            super().on_block_allocated(block, request)
        block.workload_tag = infer_workload(
            request.request_id if request is not None else None
        )

    def on_block_accessed(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Remove a block from occupancy after a cache hit pins it."""
        if self._collect_residency_metrics:
            super().on_block_accessed(block, request)
        self._sync_membership(block)

    def on_block_cached(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Count a newly cached block if it is already evictable."""
        self._sync_membership(block)

    def on_eviction_recorded(self, event: Mapping[str, object]) -> None:
        """Add a cross-workload eviction to the bounded signal window."""
        with self._signal_lock:
            now = time.time()
            self._expire_signal_events_locked(now)

            victim = event.get("evicted_workload")
            if (
                event.get("is_cross_workload") is not True
                or not isinstance(victim, str)
                or not victim
            ):
                return

            window = self._signal_windows[victim]
            window.append(event)  # Keep the same object for reuse matching.
            self._signal_event_ids.add(id(event))
            while len(window) > self._window_size:
                self._drop_signal_event_locked(
                    victim, window.popleft(), expired=False
                )

    def on_block_reused(
        self,
        event: Mapping[str, object],
        request: "Request | None" = None,
    ) -> None:
        """Mark one pending cross-workload event useful at most once."""
        requester_workload = infer_workload(
            request.request_id if request is not None else None
        )
        with self._signal_lock:
            now = time.time()
            self._expire_signal_events_locked(now)

            victim = event.get("evicted_workload")
            if (
                not isinstance(victim, str)
                or id(event) not in self._signal_event_ids
            ):
                return
            candidate_event = {
                **event,
                "useful_counted": id(event) in self._useful_event_ids,
            }
            if is_useful_eviction(
                candidate_event,
                requester_workload,
                is_cache_miss=True,
                now=now,
                shadow_ttl_sec=self._shadow_ttl_sec,
            ):
                self._useful_event_ids.add(id(event))
                self._useful_evictions[victim] += 1

    def on_block_freed(
        self,
        block: "KVCacheBlock",
        prev_ref_cnt: int,
        new_ref_cnt: int,
    ) -> None:
        """Count a cached block when its reference count reaches zero."""
        del prev_ref_cnt, new_ref_cnt
        self._sync_membership(block)

    def on_block_evicted(
        self,
        block: "KVCacheBlock",
        trigger_request: "Request | None" = None,
    ) -> None:
        """Remove the block from occupancy before its hash is reset."""
        if self._collect_residency_metrics:
            super().on_block_evicted(block, trigger_request)
        with self._quota_lock:
            if not block.is_counted_as_evictable_cached:
                return

            owner = self._owner(block)
            current = self.state[owner].evictable_cached
            if current <= 0:
                raise RuntimeError(
                    "occupancy counter underflow for "
                    f"workload={owner!r}, block_id={block.block_id}"
                )
            self.state[owner].evictable_cached = current - 1
            block.is_counted_as_evictable_cached = False

    def _expire_signal_events_locked(self, now: float) -> None:
        for workload, window in self._signal_windows.items():
            while window:
                event = window[0]
                eviction_time = event.get("eviction_time")
                if (
                    type(eviction_time) in (int, float)
                    and now - eviction_time <= self._shadow_ttl_sec
                ):
                    break
                self._drop_signal_event_locked(
                    workload, window.popleft(), expired=True
                )

    def _drop_signal_event_locked(
        self,
        workload: str,
        event: Mapping[str, object],
        *,
        expired: bool,
    ) -> None:
        self._signal_event_ids.discard(id(event))
        was_useful = id(event) in self._useful_event_ids
        self._useful_event_ids.discard(id(event))
        if was_useful:
            current = self._useful_evictions[workload]
            self._useful_evictions[workload] = max(0, current - 1)
        counter = self._shadow_expired if expired else self._shadow_dropped
        counter[workload] += 1

    def useful_eviction_snapshot(self) -> dict[str, dict[str, int | float]]:
        """Return current event-count windows and useful ratios."""
        with self._signal_lock:
            self._expire_signal_events_locked(time.time())
            workloads = set(self._signal_windows)
            workloads.update(self._useful_evictions)
            return {
                workload: {
                    "window_size": self._window_size,
                    "sample_count": len(self._signal_windows[workload]),
                    "cross_workload_evictions": len(
                        self._signal_windows[workload]
                    ),
                    "useful_evictions": self._useful_evictions[workload],
                    "useful_eviction_ratio": (
                        self._useful_evictions[workload]
                        / len(self._signal_windows[workload])
                        if self._signal_windows[workload]
                        else 0.0
                    ),
                    "shadow_expired": self._shadow_expired[workload],
                    "shadow_dropped": self._shadow_dropped[workload],
                }
                for workload in sorted(workloads)
            }

    def reset(self) -> None:
        """Clear occupancy after BlockPool resets every block membership flag."""
        super().reset()
        with self._quota_lock:
            self.state.clear()
        with self._signal_lock:
            self._signal_windows.clear()
            self._signal_event_ids.clear()
            self._useful_event_ids.clear()
            self._useful_evictions.clear()
            self._shadow_expired.clear()
            self._shadow_dropped.clear()

    def occupancy_snapshot(self) -> dict[str, int]:
        """Return a stable workload-to-occupancy snapshot."""
        with self._quota_lock:
            return {
                workload: workload_state.evictable_cached
                for workload, workload_state in self.state.items()
            }

    def verify_occupancy(self, blocks: Iterable["KVCacheBlock"]) -> None:
        """Raise if counters differ from the physical block state."""
        actual: defaultdict[str, int] = defaultdict(int)
        for block in blocks:
            if self._is_evictable_cached(block):
                actual[self._owner(block)] += 1

        with self._quota_lock:
            expected = {
                workload: workload_state.evictable_cached
                for workload, workload_state in self.state.items()
                if workload_state.evictable_cached != 0
            }
            actual_nonzero = {
                workload: count
                for workload, count in actual.items()
                if count != 0
            }
            if actual_nonzero != expected:
                raise RuntimeError(
                    "occupancy counter drift: "
                    f"actual={actual_nonzero}, expected={expected}"
                )
