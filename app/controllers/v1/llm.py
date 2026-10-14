import threading
from contextlib import contextmanager

from fastapi import Depends, Request

from app.config import config

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import llm
from app.utils import utils

# LLM 接口与视频接口共用同一鉴权规则，避免新增端点时遗漏保护。
# api_key 为空时 verify_token 直接放行，不改变默认本地使用体验。
router = new_router(dependencies=[Depends(base.verify_token)])

# These endpoints are synchronous and block a threadpool worker for the whole
# provider round trip. Bound the number in flight so a flood of requests cannot
# starve every other endpoint; excess callers get 429 instead of queueing.
_DEFAULT_MAX_CONCURRENT_LLM_REQUESTS = 8
_llm_slots_lock = threading.Lock()
_llm_slots: threading.BoundedSemaphore | None = None
_llm_slots_size = 0


def _llm_semaphore() -> threading.BoundedSemaphore | None:
    global _llm_slots, _llm_slots_size
    configured = config.app.get(
        "max_concurrent_llm_requests", _DEFAULT_MAX_CONCURRENT_LLM_REQUESTS
    )
    if isinstance(configured, bool) or not isinstance(configured, int):
        configured = _DEFAULT_MAX_CONCURRENT_LLM_REQUESTS
    if configured <= 0:
        return None
    with _llm_slots_lock:
        if _llm_slots is None or _llm_slots_size != configured:
            _llm_slots = threading.BoundedSemaphore(configured)
            _llm_slots_size = configured
        return _llm_slots


@contextmanager
def _llm_slot(request: Request):
    semaphore = _llm_semaphore()
    if semaphore is None:
        yield
        return
    if not semaphore.acquire(blocking=False):
        raise HttpException(
            task_id=base.get_task_id(request),
            status_code=429,
            message="too many concurrent LLM requests, retry later",
        )
    try:
        yield
    finally:
        semaphore.release()


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Create a script for the video",
)
def generate_video_script(request: Request, body: VideoScriptRequest):
    with _llm_slot(request):
        try:
            video_script = llm.generate_script(
                video_subject=body.video_subject,
                language=body.video_language,
                paragraph_number=body.paragraph_number,
                video_script_prompt=body.video_script_prompt,
                custom_system_prompt=body.custom_system_prompt,
            )
        except llm.LLMError as exc:
            raise HttpException(
                task_id=base.get_task_id(request), status_code=503,
                message=f"script generation failed: {exc}",
            )
    if not video_script or video_script.startswith("Error: "):
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="script generation is unavailable",
        )
    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generate video terms based on the video script",
)
def generate_video_terms(request: Request, body: VideoTermsRequest):
    with _llm_slot(request):
        try:
            video_terms = llm.generate_terms(
                video_subject=body.video_subject,
                video_script=body.video_script,
                amount=body.amount,
                match_script_order=body.match_materials_to_script,
            )
        except llm.LLMError as exc:
            raise HttpException(
                task_id=base.get_task_id(request), status_code=503,
                message=f"search term generation failed: {exc}",
            )
    if not video_terms:
        raise HttpException(
            task_id=base.get_task_id(request), status_code=503,
            message="search term generation is unavailable",
        )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generate social publishing metadata",
)
def generate_video_social_metadata(
    request: Request, body: VideoSocialMetadataRequest
):
    with _llm_slot(request):
        metadata = llm.generate_social_metadata(
            video_subject=body.video_subject,
            video_script=body.video_script,
            language=body.language,
            platform=body.platform,
        )
    return utils.get_response(200, metadata)
