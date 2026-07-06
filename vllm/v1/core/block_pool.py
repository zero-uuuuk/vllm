# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import json
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.quota_serve.workload import infer_workload
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)

_MAX_PENDING_EVICTIONS = 200_000
_eviction_log_path: str | None = os.environ.get("VLLM_EVICTION_LOG")


def _open_eviction_log(path: str | None):
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115


_eviction_log_file = _open_eviction_log(_eviction_log_path)


def _log_eviction_event(event: dict[str, Any]) -> None:
    if _eviction_log_file is not None:
        _eviction_log_file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _request_id(request: Request | None) -> str | None:
    return request.request_id if request is not None else None


def _request_workload(request: Request | None) -> str | None:
    request_id = _request_id(request)
    return infer_workload(request_id) if request_id is not None else None


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        # QuotaServe hook map (PR 0, QUOTASERVE_IMPLEMENTATION_PLAN.md §4.1)
        # ------------------------------------------------------------------
        # 모든 block lifecycle hook은 이 collector를 통해 fire되며, BlockPool이
        # 유일한 호출 지점이다. collector가 None이거나 base 구현이면 부작용이
        # 없어 baseline LRU와 동일하게 동작한다(mode=off parity, §5.3).
        #
        #   Hook #1 on_block_allocated -> get_new_blocks()        (ref_cnt 0→1)
        #   Hook #2 on_block_evicted   -> _maybe_evict_cached_block()
        #   Hook #3 on_block_cached    -> cache_full_blocks()      (insert 직후)
        #   Hook #4 on_block_accessed  -> touch()                  (cache hit)
        #   Hook #5 on_block_freed     -> free_blocks()            (ref_cnt 감소)
        #
        # 이 5개는 block lifecycle 전이를 관찰/attribution하는 진입점이다.
        # Future PR에서 owner metadata와 occupancy counter를 붙일 때 이 hook들을
        # 사용한다. victim selection을 실제로 가로채는 policy hook도 future PR에서
        # popleft_n 경로에 추가한다. PR0에서는 LRU를 그대로 두므로 불필요하다.
        self.metrics_collector = metrics_collector
        # VLLM_EVICTION_LOG keeps evicted cached-prefix blocks pending until the
        # same prefix hash is cached again. At that point the event is written
        # with reused_later=True. Any remaining pending events are flushed as
        # reused_later=False through /flush_eviction_log or at process exit.
        self._pending_evictions: dict[bytes, list[dict[str, Any]]] = {}
        self._pending_evictions_count = 0
        atexit.register(self._flush_pending_evictions)

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            blk.workload_tag = _request_workload(request)
            blk.cached_request_id = request.request_id
            blk.block_index = num_cached_blocks + i
            blk.last_access_time = time.time()
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            self._complete_pending_reuse(bytes(block_hash))
            # ===== QuotaServe Hook #3: on_block_cached (cache 등록 순간) =====
            # 이 insert로 block이 prefix cache에 등록된다(is_cached: F → T).
            # Future PR에서 occupancy counter("ref==0 and is_cached")를 붙이면
            # 이 is_cached 전이에서 재평가해야 하므로 hook을 둔다. request는 이
            # 경로에서 항상 가용하다(cache_full_blocks가 request를 받음).
            if self.metrics_collector:
                self.metrics_collector.on_block_cached(blk, request)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                )
            )

    def get_new_blocks(
        self,
        num_blocks: int,
        request: Request | None = None,
    ) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            request: (QuotaServe PR 0) trigger 요청. 이 요청이 새 block을
                필요로 해서 free queue에서 block을 끌어오고, 필요 시 cached
                block을 evict한다. owner 부여(Hook #1 on_block_allocated)와
                eviction trigger attribution(Hook #2 on_block_evicted)에
                사용된다. 호출자(single_type_kv_cache_manager)가 아직 request_id
                만 가진 경로가 있어 default는 None이다. 실제 request threading은
                future PR에서 완성한다. None이면 base collector가 인자를 무시하므로
                동작은 기존과 동일하다.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # NOTE(QuotaServe): victim 선택은 popleft_n()이 free queue의 head부터
        # (LRU 순) block을 pop하면서 결정된다. PR 0에서는 이 LRU 동작을 바꾸지
        # 않는다(=baseline parity). 실제 victim selection override는 future PR에서
        # static quota policy를 구현할 때 이 자리에 추가한다. 그
        # 전까지 "실제로 뭐가 evict됐나"는 아래 루프의 on_block_evicted(Hook #2)
        # 가 block 단위로 이미 기록하므로 별도 진입점이 필요 없다.
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                # popleft된 block이 cached 상태면 여기서 evict된다. trigger
                # request를 함께 넘겨 어떤 workload가 이 eviction을 유발했는지
                # attribution한다(Hook #2).
                self._maybe_evict_cached_block(block, trigger_request=request)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                # ===== QuotaServe Hook #1: on_block_allocated =====
                # Future PR에서 새로 할당된 block에 owner workload를 부여할 때
                # 사용할 지점이다. PR0에서는 base collector가 이 인자를 무시한다.
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block, request)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block, request)
        return ret

    def _maybe_evict_cached_block(
        self,
        block: KVCacheBlock,
        trigger_request: Request | None = None,
    ) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.
            trigger_request: (QuotaServe PR 0) 이 eviction을 유발한 요청. 새
                block이 필요해 cached block을 밀어낸 trigger workload를
                attribution하는 데 쓰인다. 외부 evict 경로(connector의
                evict_blocks)에서는 명시적 trigger가 없어 None이다.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # ===== QuotaServe Hook #2: on_block_evicted (trigger attribution) =====
        # Clean up metrics tracking first to prevent leaks.
        # block(victim) + trigger_request(가해 workload)를 함께 넘긴다. Future PR의
        # collector는 여기서 trigger attribution과 owner cleanup을 처리할 수 있다.
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block, trigger_request)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        self._remember_eviction_event(block, block_hash, trigger_request)
        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def _complete_pending_reuse(self, raw_hash_bytes: bytes) -> None:
        if self._pending_evictions_count == 0:
            return
        pending_list = self._pending_evictions.get(raw_hash_bytes)
        if not pending_list:
            return
        event = pending_list.pop(0)
        self._pending_evictions_count -= 1
        if not pending_list:
            del self._pending_evictions[raw_hash_bytes]
        event["reused_later"] = True
        event["time_until_next_reuse"] = round(
            time.time() - event["eviction_time"], 6
        )
        _log_eviction_event(event)

    def _remember_eviction_event(
        self,
        block: KVCacheBlock,
        block_hash: BlockHashWithGroupId,
        trigger_request: Request | None,
    ) -> None:
        if _eviction_log_file is None:
            return
        raw_hash_bytes = bytes(get_block_hash(block_hash))
        event: dict[str, Any] = {
            "evicted_workload": block.workload_tag,
            "trigger_workload": _request_workload(trigger_request),
            "evicted_request_id": block.cached_request_id,
            "trigger_request_id": _request_id(trigger_request),
            "evicted_prefix_hash": raw_hash_bytes.hex(),
            "evicted_block_index": block.block_index,
            "evicted_block_size": self.hash_block_size,
            "eviction_time": time.time(),
            "last_access_time": block.last_access_time,
            "reused_later": False,
            "time_until_next_reuse": None,
        }
        if self._pending_evictions_count >= _MAX_PENDING_EVICTIONS:
            oldest_key = next(iter(self._pending_evictions))
            oldest_list = self._pending_evictions[oldest_key]
            _log_eviction_event(oldest_list.pop(0))
            self._pending_evictions_count -= 1
            if not oldest_list:
                del self._pending_evictions[oldest_key]
        self._pending_evictions.setdefault(raw_hash_bytes, []).append(event)
        self._pending_evictions_count += 1

    def flush_pending_evictions(self) -> int:
        if self._pending_evictions_count == 0:
            if _eviction_log_file is not None:
                _eviction_log_file.flush()
            return 0

        num_flushed = self._pending_evictions_count
        for event_list in self._pending_evictions.values():
            for event in event_list:
                _log_eviction_event(event)
        self._pending_evictions.clear()
        self._pending_evictions_count = 0
        if _eviction_log_file is not None:
            _eviction_log_file.flush()
        logger.info("Flushed %d pending eviction events.", num_flushed)
        return num_flushed

    def _flush_pending_evictions(self) -> None:
        self.flush_pending_evictions()

    def touch(
        self,
        blocks: Sequence[KVCacheBlock],
        request: Request | None = None,
    ) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
            request: (QuotaServe PR0) cache hit을 일으킨 요청(hit-side
                workload). owner tag는 hit으로 바꾸지 않는다. Future PR에서
                필요 시 hit-side workload만 별도로 기록한다. 호출자
                (single_type_kv_cache_manager.add_new_computed_blocks)가 아직
                request_id만 가진 경로라 default는 None이다.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            #
            # NOTE(QuotaServe future PR): ref_cnt 0 → 1 전이는 block이 evictable
            # 후보에서 빠지는 순간이다. Occupancy counter를 추가하면 이 transition
            # 에서 -1 해야 하지만, PR0에서는 ref_cnt만 조정한다.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            block.last_access_time = time.time()
            # ===== QuotaServe Hook #4: on_block_accessed (hit-side workload) =====
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block, request)

    def free_blocks(
        self,
        ordered_blocks: Iterable[KVCacheBlock],
        request: Request | None = None,
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            request: (QuotaServe PR 0) 이 block들을 free하는 요청. owner
                attribution 자체는 on_block_freed가 block 단위로 처리하므로
                필수는 아니지만, 향후 free-side workload 계측을 위해 시그니처에
                포함한다. 호출자가 request_id만 가진 경로가 있어 default는 None.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            # ===== QuotaServe Hook #5: on_block_freed (ref_cnt → 0 transition) =====
            # ref_cnt 감소 전/후 값을 함께 넘긴다. new_ref_cnt == 0 이고 block이
            # cached면 그 block은 free queue로 들어가 evictable 후보가 된다.
            # Future PR에서 occupancy counter를 추가하면 이 transition에서 +1 한다.
            prev_ref_cnt = block.ref_cnt
            block.ref_cnt -= 1
            if self.metrics_collector:
                self.metrics_collector.on_block_freed(
                    block, prev_ref_cnt, block.ref_cnt
                )
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()
        self.flush_pending_evictions()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
