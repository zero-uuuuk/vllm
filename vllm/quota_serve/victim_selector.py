# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR4 static quota victim selection.

This module contains the policy decision only.  ``occupancy_snapshot`` is
provided by the PR3 collector, while ``BlockPool`` remains responsible for
removing the selected block from the free queue.  Keeping those responsibilities
separate is important when one allocation needs more than one block: BlockPool
calls this selector once per block, so the next call sees the queue after the
previous block has been removed.

이 모듈은 어떤 block을 victim으로 선택할지만 결정한다. ``occupancy_snapshot``
은 PR3 collector가 제공하고, 선택된 block을 free queue에서 실제로 제거하는
일은 ``BlockPool``이 담당한다. 한 번의 allocation에서 여러 block이 필요할 수
있으므로 이 역할을 분리해야 한다. BlockPool은 block을 하나 선택하고 제거한
뒤 이 selector를 다시 호출하므로, 다음 호출은 갱신된 queue를 본다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.quota_serve.config import QuotaServeConfig

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class VictimSelection:
    """A victim decision and the metadata needed for PR4 logging.

    The selector returns this record internally and stores it as
    ``last_selection``.  The selected block itself is returned to BlockPool;
    the remaining fields make the decision explainable in an eviction log.

    selector는 이 record를 내부적으로 만들고 ``last_selection``에 저장한다.
    실제로 BlockPool에 반환되는 것은 선택된 block이고, 나머지 필드는 eviction
    log에서 선택 이유를 설명하기 위해 사용한다.
    """

    # The physical KV block that BlockPool should remove or reuse.
    # BlockPool이 제거하거나 재사용할 실제 physical KV block이다.
    block: "KVCacheBlock"
    # Why this block was selected, for example over_quota_selected or fallback.
    # over_quota_selected 또는 fallback처럼 block이 선택된 이유다.
    reason: str
    # Number of free-queue nodes inspected, starting at the LRU head.
    # LRU head부터 몇 개의 free-queue node를 확인했는지 나타낸다.
    scan_steps: int
    # Owner recorded on the block at the time of the decision.
    # 선택 시점에 block에 기록되어 있던 owner workload다.
    victim_workload: str
    # Owner's occupancy from the single snapshot used for this decision.
    # 이 선택에서 사용한 하나의 snapshot에 기록된 owner의 occupancy다.
    victim_occupancy: int
    # Owner's ratio converted to an absolute block quota, when applicable.
    # owner의 quota ratio를 absolute block quota로 변환한 값이다.
    victim_quota: int | None
    # Occupancy for all workloads, retained so the decision can be audited.
    # 선택 과정을 확인할 수 있도록 보관하는 전체 workload occupancy다.
    occupancy_snapshot: dict[str, int]


class QuotaAwareVictimSelector:
    """Select a victim using static workload quotas and queue LRU order.

    The free queue is ordered from oldest to newest, so its head is the global
    LRU candidate.  If one or more workloads are over quota, the selector scans
    from that head and returns the first eligible block owned by one of them.
    Therefore the result is the LRU block within the over-quota candidates,
    while preserving the existing global LRU behavior for fallback cases.

    따라서 결과는 over-quota 후보들 중 LRU block이다. over-quota 후보가 없는
    fallback 상황에서는 기존 global LRU 동작을 그대로 유지한다.

    This class only decides which block should be selected.  It does not mutate
    ``free_queue``; BlockPool removes the returned block.  That ownership lets
    BlockPool repeat the decision safely when ``num_blocks > 1``.

    이 class는 어떤 block을 선택할지만 결정하고 ``free_queue``를 직접 수정하지
    않는다. 반환된 block을 queue에서 제거하는 일은 BlockPool이 담당한다. 이
    구조 덕분에 ``num_blocks > 1``일 때도 BlockPool이 선택을 안전하게 반복할
    수 있다.
    """

    def __init__(
        self,
        quota_config: QuotaServeConfig,
        occupancy_snapshot: Callable[[], Mapping[str, int]],
        quota_base_blocks: int,
    ) -> None:
        if quota_base_blocks <= 0:
            raise ValueError(
                "quota_base_blocks must be positive, "
                f"got {quota_base_blocks!r}"
            )
        self.quota_config = quota_config
        # The collector is the source of truth for the current evictable
        # occupancy.  It is read once per selection, rather than maintained
        # independently here, to avoid two counters drifting apart.
        # 현재 evictable occupancy의 기준값은 collector가 가진다. 여기서 별도의
        # counter를 관리하지 않고 선택마다 한 번 읽어 두 counter가 어긋나는
        # 문제를 피한다.
        self._occupancy_snapshot = occupancy_snapshot
        # Ratios need a common absolute denominator before they can be compared
        # with an integer occupancy count.  This is fixed for the static mode.
        # ratio를 정수 occupancy와 비교하려면 공통 absolute 기준이 필요하다.
        # static mode에서는 이 기준값을 고정해서 사용한다.
        self.quota_base_blocks = quota_base_blocks
        # Kept for the caller/logger; setting this does not change queue state.
        # 호출자와 logger가 마지막 선택 결과를 확인할 수 있도록 저장한다.
        # 이 값을 저장하는 것만으로 queue 상태가 바뀌지는 않는다.
        self.last_selection: VictimSelection | None = None

    def __call__(
        self,
        free_queue: "FreeKVCacheBlockQueue",
        trigger_request: "Request | None" = None,
    ) -> "KVCacheBlock":
        """Return a selected block without mutating ``free_queue``.

        BlockPool uses the selector as a callable.  ``trigger_request`` is
        accepted as part of that interface so a later PR can attribute an
        eviction to the request that triggered it.  PR4's choice itself is
        based on the queue, the occupancy snapshot, and the static quotas.

        BlockPool이 selector를 callable처럼 사용하기 때문에 이 인터페이스를
        따른다. ``trigger_request``는 나중에 eviction을 발생시킨 request를
        attribution하기 위해 미리 받지만, PR4의 선택 자체는 queue,
        occupancy snapshot, static quota만을 기준으로 한다.
        """
        del trigger_request
        selection = self.select_victim(free_queue)
        self.last_selection = selection
        return selection.block

    def select_victim(
        self,
        free_queue: "FreeKVCacheBlockQueue",
    ) -> VictimSelection:
        """Select the oldest valid candidate under the static quota policy.

        The decision order is:

        1. ``off`` or inactive mode: preserve the global LRU head.
        2. An uncached head: reuse it without counting the event as eviction.
        3. No over-quota workload: preserve the global LRU head as fallback.
        4. Otherwise, scan from the LRU head until the first eligible
           over-quota block is found.

        The scan is intentionally unbounded.  If the occupancy counter says a
        workload owns an evictable cached block, failing to find one in the
        queue indicates a state/accounting bug and raises an error instead of
        silently degrading to an unexplained LRU choice.

        선택 순서는 다음과 같다.

        1. ``off`` 또는 비활성 mode이면 global LRU head를 유지한다.
        2. cached block이 아닌 head이면 eviction 없이 바로 재사용한다.
        3. over-quota workload가 없으면 global LRU head를 fallback으로 사용한다.
        4. over-quota workload가 있으면 LRU head부터 순회해 조건에 맞는
           첫 번째 block을 선택한다.

        이 순회는 중간에 임의로 멈추지 않는다. occupancy counter가 어떤
        workload에 evictable cached block이 있다고 판단했는데 queue에서 찾지
        못했다면 상태 또는 accounting이 어긋난 것이므로, 설명되지 않은 LRU
        fallback으로 숨기지 않고 오류를 발생시킨다.
        """
        # The queue uses sentinel nodes.  The real first node is the oldest
        # (LRU) candidate; the tail sentinel marks the end of the queue.
        # queue는 sentinel node를 사용한다. 실제 첫 node가 가장 오래된
        # (LRU) 후보이고, tail sentinel이 queue의 끝을 나타낸다.
        head = free_queue.fake_free_list_head.next_free_block
        tail = free_queue.fake_free_list_tail
        if head is None or head is tail:
            raise RuntimeError("cannot select a victim from an empty free queue")

        # Copy the collector result so all fields in this decision describe
        # one point-in-time view, even if the caller later mutates its mapping.
        # collector 결과를 복사해 선택 시점의 하나의 상태로 고정한다. 이후
        # 호출자가 원본 mapping을 바꿔도 이 decision의 모든 필드는 같은 시점의
        # 값을 설명한다.
        occupancy = dict(self._occupancy_snapshot())

        if not self.quota_config.victim_selection_active:
            # ``off`` must be behaviorally equivalent to the pre-QuotaServe
            # path: select the queue head and do not apply quota filtering.
            # ``off``에서는 QuotaServe 적용 전과 동작이 같아야 한다. queue
            # head를 선택하고 quota filtering을 적용하지 않는다.
            return self._decision(
                head,
                reason="baseline_lru_off_mode",
                scan_steps=1,
                occupancy=occupancy,
            )

        # An uncached block can be reused immediately.  No cached prefix is
        # removed, so this is reported separately from eviction fallbacks.
        # cached block이 아닌 block은 즉시 재사용할 수 있다. cached prefix를
        # 제거한 것이 아니므로 eviction fallback과 별도의 이유로 기록한다.
        if head.block_hash is None:
            return self._decision(
                head,
                reason="uncached_head",
                scan_steps=1,
                occupancy=occupancy,
                victim_quota=None,
            )

        # Only configured workloads participate in the quota comparison.  A
        # workload is over quota strictly when occupancy is greater than its
        # absolute quota; equality is still within quota.
        # quota 설정에 등록된 workload만 비교 대상이다. occupancy가 absolute
        # quota보다 클 때만 over-quota이며, 둘이 같은 경우는 quota 이내다.
        over_quota_workloads = {
            workload
            for workload in self.quota_config.workloads
            if occupancy.get(workload, 0) > self._absolute_quota(workload)
        }
        if not over_quota_workloads:
            # QuotaServe has no workload to protect by eviction, so retain the
            # global LRU decision rather than forcing an arbitrary scan.
            # 우선적으로 evict할 workload가 없으므로 임의로 queue를 더 훑지
            # 않고 global LRU 결정을 그대로 사용한다.
            return self._decision(
                head,
                reason="fallback_no_over_quota",
                scan_steps=1,
                occupancy=occupancy,
            )

        # Because we scan from the head, the first matching block is the oldest
        # block among all eligible over-quota candidates.
        # head부터 순회하므로 처음 발견한 matching block이 모든 eligible
        # over-quota 후보 중 가장 오래된 block이다.
        scan_steps = 0
        block = head
        while block is not None and block is not tail:
            scan_steps += 1
            if (
                not block.is_null
                and block.ref_cnt == 0
                and block.block_hash is not None
                and block.workload_tag in over_quota_workloads
            ):
                return self._decision(
                    block,
                    reason="over_quota_selected",
                    scan_steps=scan_steps,
                    occupancy=occupancy,
                )
            block = block.next_free_block

        # Reaching here means the occupancy state and the free queue disagree:
        # an over-quota workload was reported, but no evictable cached block
        # owned by it was present.  Strict failure exposes that drift early.
        # 여기까지 도달했다는 것은 occupancy 상태와 free queue가 서로 다르다는
        # 뜻이다. over-quota workload가 있다고 기록됐지만 해당 workload의
        # evictable cached block이 queue에 없다. strict failure로 이 drift를
        # 조기에 드러낸다.
        raise RuntimeError(
            "over-quota workload exists but no evictable cached block was found"
        )

    def _absolute_quota(self, workload: str) -> int:
        # Convert the configured ratio into the integer block count used by
        # the occupancy comparison.  Static mode keeps the base fixed.
        # 설정된 ratio를 occupancy 비교에 사용할 정수 block 수로 변환한다.
        # static mode에서는 base를 고정해서 사용한다.
        quota_ratio = self.quota_config.quota_for(workload).quota_ratio
        return int(quota_ratio * self.quota_base_blocks)

    def _decision(
        self,
        block: "KVCacheBlock",
        *,
        reason: str,
        scan_steps: int,
        occupancy: dict[str, int],
        victim_quota: int | None = None,
    ) -> VictimSelection:
        # Blocks without an owner can occur in baseline/uncached paths.  Keep
        # the log value explicit rather than treating an empty tag as a real
        # workload name.
        # baseline 또는 uncached 경로에서는 owner가 없을 수 있다. 빈 tag를
        # 실제 workload 이름으로 취급하지 않고 log에 명시적으로 기록한다.
        workload = block.workload_tag or "unknown"
        if victim_quota is None and block.block_hash is not None:
            victim_quota = self._absolute_quota(workload)
        # Package the exact state used to explain this choice.  This method
        # does not select or remove anything; it only builds log metadata.
        # 이 선택을 설명하는 데 사용한 상태를 그대로 묶는다. 이 함수는
        # block을 선택하거나 제거하지 않고 log metadata만 만든다.
        return VictimSelection(
            block=block,
            reason=reason,
            scan_steps=scan_steps,
            victim_workload=workload,
            victim_occupancy=occupancy.get(workload, 0),
            victim_quota=victim_quota,
            occupancy_snapshot=occupancy,
        )
