# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mean-footprint KV guarantees with a shared LRU borrowing pool."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


class QuotaServeAdapter:
    """Reclaim borrowed capacity before evicting another app's entitlement.

    Count both pinned and cached blocks. Only unreferenced cached blocks can
    be victims. Mean demand persists across think time; cache reset clears it.
    """

    def __init__(
        self, blocks: list[KVCacheBlock], free_block_queue: FreeKVCacheBlockQueue
    ) -> None:
        try:
            apps = json.loads(os.getenv("QUOTASERVE_APPS", ""))
        except (ValueError, TypeError) as exc:
            raise ValueError("QUOTASERVE_APPS must be a JSON list of app IDs") from exc
        if (
            not isinstance(apps, list)
            or not apps
            or any(not isinstance(app, str) or not app for app in apps)
            or len(set(apps)) != len(apps)
        ):
            raise ValueError("QUOTASERVE_APPS requires distinct nonempty app IDs")
        self.apps = tuple(apps)
        self.capacity = sum(not block.is_null for block in blocks)
        self.mean_demand: dict[str, float] = {}
        self.quotas: dict[str, int] = {}
        self._refresh_quotas()
        self.free_block_queue = free_block_queue
        self.resident: Counter[str | None] = Counter()
        self._owners: dict[int, str | None] = {}
        self._free_uncached: OrderedDict[int, KVCacheBlock] = OrderedDict(
            (block.block_id, block) for block in blocks if not block.is_null
        )
        self._free_cached: dict[str | None, OrderedDict[int, KVCacheBlock]] = {}
        self._free_order: dict[int, int] = {}
        self._order = 0

    def observe_demand(self, app: str | None, num_blocks: int) -> None:
        """Observe a normally completed request's physical KV footprint.

        Cached hits count toward footprint. Arrival frequency and recomputed
        token rate do not enter the score. A quiet app retains its last mean.
        """
        if app not in self.apps or num_blocks <= 0:
            return
        assert app is not None
        previous = self.mean_demand.get(app, float(num_blocks))
        self.mean_demand[app] = 0.8 * previous + 0.2 * num_blocks
        self._refresh_quotas()

    def _refresh_quotas(self) -> None:
        # An unseen app inherits the observed apps' mean, avoiding a zero-quota
        # cold start. Before any observations, equal scores give equal quotas.
        prior = (
            sum(self.mean_demand.values()) / len(self.mean_demand)
            if self.mean_demand
            else self.capacity / len(self.apps)
        )
        scores = {
            app: max(1, math.ceil(self.mean_demand.get(app, prior)))
            for app in self.apps
        }
        total = sum(scores.values())
        budget = min(self.capacity, total)
        exact = {app: budget * score / total for app, score in scores.items()}
        quotas = {app: math.floor(value) for app, value in exact.items()}
        order = sorted(exact, key=lambda app: (-(exact[app] - quotas[app]), app))
        for app in order[: budget - sum(quotas.values())]:
            quotas[app] += 1
        if quotas != self.quotas:
            self.quotas = quotas
            logger.info(
                "QuotaServe mean-demand quotas: demand=%s blocks=%s",
                self.mean_demand,
                self.quotas,
            )

    def _release(self, block: KVCacheBlock) -> None:
        if block.block_id in self._owners:
            owner = self._owners.pop(block.block_id)
            self.resident[owner] -= 1
            if not self.resident[owner]:
                del self.resident[owner]

    def on_allocate(self, block: KVCacheBlock, app: str | None) -> None:
        self._release(block)
        self._owners[block.block_id] = app
        self.resident[app] += 1

    def on_cached(self, block: KVCacheBlock, request: Request) -> None:
        block.owner = request.application_id
        self.on_allocate(block, request.application_id)

    def on_touch(self, block: KVCacheBlock) -> None:
        self._forget_free(block)
        # Ownership/occupancy does not disappear when a cache hit pins a block.

    def on_free(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            self._release(block)
            self._free_uncached[block.block_id] = block
        else:
            self._free_cached.setdefault(block.owner, OrderedDict())[block.block_id] = (
                block
            )
            self._order += 1
            self._free_order[block.block_id] = self._order

    def on_evict(self, block: KVCacheBlock) -> None:
        if block.ref_cnt == 0:
            self._forget_free(block)
            self._release(block)
        # An externally invalidated pinned block still consumes physical memory.

    def on_reset(self) -> None:
        self.mean_demand.clear()
        self._refresh_quotas()
        self.resident.clear()
        self._owners.clear()
        self._free_cached.clear()
        self._free_order.clear()
        self._order = 0
        self._free_uncached = OrderedDict(
            (block.block_id, block)
            for block in self.free_block_queue.get_all_free_blocks()
        )

    def _forget_free(self, block: KVCacheBlock) -> None:
        self._free_uncached.pop(block.block_id, None)
        owner_blocks = self._free_cached.get(block.owner)
        if owner_blocks is not None:
            owner_blocks.pop(block.block_id, None)
            if not owner_blocks:
                del self._free_cached[block.owner]
        self._free_order.pop(block.block_id, None)

    def take_free_block(
        self, application_id: str | None = None
    ) -> tuple[KVCacheBlock, str | None]:
        if self._free_uncached:
            block = next(iter(self._free_uncached.values()))
            reason = None
        else:
            # Reclaim the oldest borrowed block. Include the incoming allocation
            # so an app at quota can replace its own blocks without stealing.
            # ponytail: O(apps) scan; use a heap if app count warrants it.
            candidates = []
            for app, blocks in self._free_cached.items():
                quota = self.quotas.get(app, 0) if app is not None else 0
                excess = self.resident[app] + (app == application_id) - quota
                if excess > 0:
                    head_id = next(iter(blocks))
                    candidates.append((self._free_order[head_id], head_id))
            if candidates:
                _, head_id = min(candidates)
                # The queue map avoids a scan over blocks within the chosen app.
                owner = self._owners[head_id]
                block = self._free_cached[owner][head_id]
                reason = "quotaserve"
            else:
                # Excess owners can have all their blocks pinned. Preserve progress
                # and expose this exception rather than claiming strict isolation.
                block = self.free_block_queue.fake_free_list_head.next_free_block
                reason = "quotaserve_pressure"
                assert block is not None
        self._forget_free(block)
        self.free_block_queue.remove(block)
        return block, reason
