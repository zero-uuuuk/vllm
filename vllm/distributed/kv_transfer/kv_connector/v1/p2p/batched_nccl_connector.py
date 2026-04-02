# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BatchedNcclConnector: P2pNcclConnector with configurable transfer granularity.

granularity=1  → identical to P2pNcclConnector (per-layer send, baseline)
granularity=8  → concat 8 layers into one tensor, single NCCL send (4 calls)
granularity=32 → concat all layers into one tensor, single NCCL send (1 call)
"""

import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class BatchedNcclConnector(P2pNcclConnector):
    """P2pNcclConnector with configurable KV transfer granularity.

    Controls how many layers' KV caches are concatenated before a single
    NCCL send, trading off transfer-compute overlap against per-call overhead.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._granularity: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_granularity", 1
            )
        )
        # Producer: accumulate layer KVs before concat+send
        self._layer_buffer: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._layer_count: dict[str, int] = {}
        self._batch_idx: dict[str, int] = {}

        logger.info(
            "BatchedNcclConnector initialized with granularity=%d",
            self._granularity,
        )

    # ================================================================
    #  Producer (Prefill) side
    # ================================================================

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if not self.is_producer:
            return

        if self._granularity <= 1:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return

        assert self.p2p_nccl_engine is not None
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)

        for request in connector_metadata.requests:
            request_id = request.request_id
            kv_cache = self._extract_kv(kv_layer, request.block_ids,
                                        attn_metadata)
            if kv_cache is None:
                continue

            if request_id not in self._layer_buffer:
                self._layer_buffer[request_id] = []
                self._layer_count[request_id] = 0
                self._batch_idx[request_id] = 0

            self._layer_buffer[request_id].append((layer_name, kv_cache))
            self._layer_count[request_id] += 1

            if self._layer_count[request_id] % self._granularity == 0:
                self._flush_concat(request_id, request)

    def _flush_concat(self, request_id: str, request: Any) -> None:
        """Concat buffered layer KVs into one tensor and send."""
        assert self.p2p_nccl_engine is not None

        ip, port = self.parse_request_id(request_id, True)
        remote_address = f"{ip}:{port + self._rank}"

        kvs = [kv for _, kv in self._layer_buffer[request_id]]
        merged = torch.cat(kvs, dim=0)

        batch_idx = self._batch_idx[request_id]
        tensor_id = f"{request_id}#batch_{batch_idx}"
        self.p2p_nccl_engine.send_tensor(tensor_id, merged, remote_address)

        self._batch_idx[request_id] += 1
        self._layer_buffer[request_id].clear()

        logger.debug(
            "Flushed batch_%d for %s, shape=%s",
            batch_idx,
            request_id,
            merged.shape,
        )

    def wait_for_save(self) -> None:
        if self.is_producer and self._granularity > 1:
            connector_metadata = self._get_connector_metadata()
            assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
            for request in connector_metadata.requests:
                rid = request.request_id
                if rid in self._layer_buffer and self._layer_buffer[rid]:
                    self._flush_concat(rid, request)
            self._layer_count.clear()
            self._batch_idx.clear()
        super().wait_for_save()

    # ================================================================
    #  Consumer (Decode) side
    # ================================================================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        if self.is_producer:
            return

        if self._granularity <= 1:
            super().start_load_kv(forward_context, **kwargs)
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)

        attn_layer_names: list[str] = []
        for layer_name in forward_context.no_compile_layers:
            layer_obj = forward_context.no_compile_layers[layer_name]
            if getattr(layer_obj, "kv_cache", None) is not None:
                attn_layer_names.append(layer_name)

        num_attn_layers = len(attn_layer_names)
        num_batches = (
            (num_attn_layers + self._granularity - 1) // self._granularity
        )

        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)

            for batch_idx in range(num_batches):
                start = batch_idx * self._granularity
                end = min(start + self._granularity, num_attn_layers)
                batch_layer_names = attn_layer_names[start:end]

                tensor_id = f"{request_id}#batch_{batch_idx}"
                merged = self.p2p_nccl_engine.recv_tensor(
                    tensor_id, remote_address
                )
                logger.info(
                    "⏱️KV recv: ts=%.3f, tensor_id:%s",
                    time.time(),
                    tensor_id,
                )

                if merged is None:
                    logger.warning("batch recv failed: %s", tensor_id)
                    continue

                chunks = self._split_kv(
                    merged, len(batch_layer_names), attn_metadata
                )

                for layer_name, kv_chunk in zip(batch_layer_names, chunks):
                    layer_obj = forward_context.no_compile_layers[layer_name]
                    kv_cache_tensor = layer_obj.kv_cache
                    self._inject_kv(
                        kv_cache_tensor,
                        kv_chunk,
                        request.block_ids,
                        request_id,
                        attn_metadata,
                    )

    # ================================================================
    #  Helpers
    # ================================================================

    @staticmethod
    def _extract_kv(
        kv_layer: torch.Tensor,
        block_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor | None:
        if isinstance(attn_metadata, MLACommonMetadata) or kv_layer.shape[1] == 2:
            return kv_layer[block_ids, ...]
        if kv_layer.shape[0] == 2:
            return kv_layer[:, block_ids, ...]
        return None

    @staticmethod
    def _split_kv(
        merged: torch.Tensor,
        num_layers: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, ...]:
        """Split a concatenated tensor back into per-layer tensors (views)."""
        if isinstance(attn_metadata, MLACommonMetadata) or merged.shape[1] == 2:
            per_layer = merged.shape[0] // num_layers
            return merged.split(per_layer, dim=0)
        if merged.shape[0] == 2:
            per_layer = merged.shape[1] // num_layers
            return merged.split(per_layer, dim=1)
        return (merged,)

    @staticmethod
    def _inject_kv(
        layer: torch.Tensor,
        kv_cache: torch.Tensor,
        block_ids: torch.Tensor,
        request_id: str,
        attn_metadata: AttentionMetadata,
    ) -> None:
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            num_block = kv_cache.shape[0]
            if len(block_ids) == num_block:
                layer[block_ids, ...] = kv_cache
            else:
                layer[block_ids[:num_block], ...] = kv_cache
                logger.warning(
                    "kv_cache block mismatch, block_ids:%d, "
                    "num_block:%d, request_id:%s",
                    len(block_ids),
                    num_block,
                    request_id,
                )
        elif layer.shape[0] == 2:
            num_block = kv_cache.shape[1]
            if len(block_ids) == num_block:
                layer[:, block_ids, ...] = kv_cache
            else:
                layer[:, block_ids[:num_block], ...] = kv_cache
                logger.warning(
                    "kv_cache block mismatch, block_ids:%d, "
                    "num_block:%d, request_id:%s",
                    len(block_ids),
                    num_block,
                    request_id,
                )
