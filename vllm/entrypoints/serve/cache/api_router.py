# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """
    Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Return HTTP 409 if active requests prevent the cache from being reset.

    Example:
       POST /reset_prefix_cache?reset_external=true
    """
    logger.info("Resetting prefix cache...")

    reset = await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    if not reset:
        raise HTTPException(status_code=409, detail="Prefix cache reset failed.")
    return Response(status_code=200)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """
    Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """
    Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


@router.post("/workload_eviction_report")
async def write_workload_eviction_report(
    raw_request: Request, path: str | None = Query(default=None)
):
    """Write workload evictions as JSONL and return summary statistics.

    The fixed endpoint can be called by an external benchmark client. The
    output path may be passed as ``?path=...`` or configured with
    ``VLLM_WORKLOAD_EVICTION_LOG_PATH``.
    """
    path = path or envs.VLLM_WORKLOAD_EVICTION_LOG_PATH
    if not path:
        raise HTTPException(
            status_code=400,
            detail="Provide a report path with ?path= or "
            "VLLM_WORKLOAD_EVICTION_LOG_PATH.",
        )
    logger.info("Writing workload eviction report to %s", path)
    return await engine_client(raw_request).write_workload_eviction_report(path)


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
