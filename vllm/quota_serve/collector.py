# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuotaServe PR3 block ownership and occupancy accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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
    ) -> None:
        super().__init__(sample_rate)
        self._collect_residency_metrics = collect_residency_metrics
        self.state: defaultdict[str, WorkloadState] = defaultdict(WorkloadState)
        self._quota_lock = Lock()

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

    def reset(self) -> None:
        """Clear occupancy after BlockPool resets every block membership flag."""
        super().reset()
        with self._quota_lock:
            self.state.clear()

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
