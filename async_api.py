# Shared async API infrastructure for OpenRouter calls.
# Used by screen.py (pydantic_ai Agent), classify.py, classify_single.py (pydantic_ai Agent),
# and embed.py (aiohttp for /embeddings endpoint).
#  
# This file has two parallel stacks:
# 1) pydantic_ai Agent stack (httpx-based) - used by screen.py, classify.py, classify_single.py
#   create_retrying_client → 
#   client singleton → 
#   create_agent → 
#   _call_agent → 
#   process_batch_agent → 
#   process_all_models_agent — 
# 2) aiohttp stack: — used by embed.py (which hits the /embeddings endpoint directly)
#   retry_aiohttp_call → 
#   process_batch_aiohttp + make_openrouter_headers 

import asyncio
import logging
import random
import sys
from typing import Any, Awaitable, Callable, Iterable, List, Optional, TypeVar

import aiohttp
import httpx
from httpx import AsyncClient, HTTPStatusError, Response
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --- Constants ---

RETRYABLE_STATUSES: set[int] = {429, 502, 503, 504}
PERMANENT_ERROR_STATUSES: set[int] = {400, 401, 403, 404, 405, 422}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# --- httpx retry client (for pydantic_ai Agent) ---


# httpx event hook that logs retryable HTTP errors (429, 502-504).
# Attached to the module-level httpx client used by all pydantic_ai Agent calls.
async def log_response(response: Response) -> None:
    if response.status_code in RETRYABLE_STATUSES:
        logger.error(
            "Retryable status %s for %s %s",
            response.status_code,
            response.request.method,
            str(response.request.url),
        )


# Build an httpx AsyncClient with automatic retry on rate-limit (429) and server
# errors (502-504) using tenacity exponential backoff. Called once at module level
# to create the singleton `client` that all pydantic_ai Agents share.
def create_retrying_client(
    *,
    max_wait_seconds: float = 300,
    max_attempts: int = 6,
    retryable_statuses: Iterable[int] = RETRYABLE_STATUSES,
    timeout: httpx.Timeout | None = httpx.Timeout(120.0),
) -> AsyncClient:
    retryable = set(retryable_statuses)

    def validate_response(response: Response) -> None:
        if response.status_code in retryable:
            response.raise_for_status()

    transport = AsyncTenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential_jitter(
                    initial=1, max=max_wait_seconds, jitter=5
                ),
                max_wait=max_wait_seconds,
            ),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        ),
        validate_response=validate_response,
    )
    return AsyncClient(
        transport=transport,
        timeout=timeout,
        event_hooks={"response": [log_response]},
    )


# Module-level singleton client for all pydantic_ai Agent calls
client = create_retrying_client()


# --- Agent factory ---


# Create a pydantic_ai Agent configured for an OpenRouter model with deterministic
# settings (temperature=0, top_p=0.1) and structured output. The agent uses the
# module-level retrying httpx client for HTTP resilience.
# Called by: screen.py (via process_all_models_agent), classify.py, classify_single.py.
def create_agent(
    model_name: str,
    api_key: str,
    *,
    system_prompt: str = "",
    output_type: Any,
    retries: int = 3,
    output_retries: int = 5,
) -> Agent:
    settings = OpenRouterModelSettings(
        extra_headers={
            "X-Title": "AISysRev",
            "HTTP-Referer": "https://github.com/EvoTestOps/AISysRev",
        },
        temperature=0,
        top_p=0.1,
    )
    model = OpenRouterModel(
        model_name,
        provider=OpenRouterProvider(api_key=api_key, http_client=client),
        settings=settings,
    )
    return Agent(
        model,
        system_prompt=system_prompt,
        retries=retries,
        output_retries=output_retries,
        output_type=output_type,
    )


# --- Permanent error detection ---


# Check if an exception indicates a permanent HTTP error (400, 401, 403, 404, 405, 422)
# that means further API calls will also fail (e.g. bad API key, invalid model).
# Used by _call_agent to trigger early abort of remaining batch items.
def _is_permanent_error(exc: Exception) -> bool:
    """Check if an error is permanent (not worth retrying other calls)."""
    error_str = str(exc)
    for status in PERMANENT_ERROR_STATUSES:
        if f"status_code: {status}" in error_str:
            return True
    return False


# --- pydantic_ai Agent concurrency orchestration ---


# Send a single prompt to a pydantic_ai Agent, respecting the concurrency semaphore.
# Tracks errors in shared error_state dict and signals abort on permanent errors
# so sibling tasks in the same batch can exit early.
# Internal helper — called only by process_batch_agent.
async def _call_agent(
    prompt: str,
    agent: Agent,
    model_name: str,
    semaphore: asyncio.Semaphore,
    error_state: dict,
    abort: asyncio.Event,
) -> Optional[Any]:
    if abort.is_set():
        return None
    async with semaphore:
        if abort.is_set():
            return None
        try:
            result = await agent.run(prompt)
            return result.output
        except Exception as e:
            error_state["count"] += 1
            if error_state["count"] == 1:
                error_state["first_error"] = str(e)
                logger.error(f"LLM call failed for model {model_name}: {e}")
                print(f"LLM call failed for model {model_name}: {e}")
                if _is_permanent_error(e):
                    abort.set()
    return None


# Run all prompts through a single pydantic_ai Agent with semaphore-limited concurrency.
# Returns a list of parsed outputs (or None for failures) in the same order as prompts.
# Aborts remaining calls early on permanent errors; deduplicates repeated error messages.
# Called by: classify.py, classify_single.py (directly), screen.py (via process_all_models_agent).
async def process_batch_agent(
    prompts: List[str],
    agent: Agent,
    model_name: str,
    max_concurrent: int = 20,
) -> List[Optional[Any]]:
    """Process a batch of prompts through a pydantic_ai Agent with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    error_state = {"count": 0, "first_error": None}
    abort = asyncio.Event()
    tasks = [
        _call_agent(prompt, agent, model_name, semaphore, error_state, abort)
        for prompt in prompts
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {model_name}")
    if abort.is_set():
        skipped = sum(1 for r in results if r is None)
        print(
            f"Permanent error for model {model_name}, skipped {skipped}/{len(prompts)} calls"
        )
    elif error_state["count"] > 1:
        print(
            f"... error repeated {error_state['count']} times for model {model_name}"
        )
    return results


# Top-level orchestrator: creates one pydantic_ai Agent per model and runs all models
# concurrently (each model's prompts are further concurrency-limited by process_batch_agent).
# Returns a list-of-lists: outer = models, inner = per-prompt results.
# Called by: screen.py only (the main screening pipeline).
async def process_all_models_agent(
    prompts: List[str],
    models: List[str],
    api_key: str,
    *,
    system_prompt: str,
    output_type: Any,
    max_concurrent_per_model: int = 20,
) -> List[List[Optional[Any]]]:
    """Process prompts through multiple models concurrently."""
    logger.info(
        f"Processing {len(models)} models with {max_concurrent_per_model} concurrent prompts per model..."
    )
    print(
        f"Processing {len(models)} models with {max_concurrent_per_model} concurrent prompts per model..."
    )
    model_tasks = []
    for model_name in models:
        agent = create_agent(
            model_name,
            api_key,
            system_prompt=system_prompt,
            output_type=output_type,
        )
        model_tasks.append(
            process_batch_agent(prompts, agent, model_name, max_concurrent_per_model)
        )
    return await tqdm_asyncio.gather(*model_tasks, desc="Processing models")


# --- aiohttp retry (for embed.py and other raw HTTP callers) ---


class PermanentAPIError(Exception):
    """Raised when the API returns a non-retryable error status code."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Permanent API error {status_code}: {message}")


# Make a single aiohttp POST with manual retry logic (exponential backoff for 429/5xx,
# immediate abort for permanent errors like 401/404). This is the low-level HTTP layer
# for callers that don't use pydantic_ai (which has its own httpx-based retry).
# Called by: embed.py to hit the OpenRouter /embeddings endpoint.
async def retry_aiohttp_call(
    session: aiohttp.ClientSession,
    url: str,
    *,
    json_payload: dict,
    headers: dict[str, str],
    timeout: aiohttp.ClientTimeout = aiohttp.ClientTimeout(total=120),
    max_retries: int = 6,
    max_wait_seconds: float = 300,
) -> Optional[dict]:
    """Make an aiohttp request with retry logic for rate limits and server errors."""
    for attempt in range(max_retries):
        try:
            async with session.post(
                url, headers=headers, json=json_payload, timeout=timeout
            ) as response:
                if response.status in PERMANENT_ERROR_STATUSES:
                    body = await response.text()
                    raise PermanentAPIError(response.status, body)

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    jitter = random.uniform(0, 5)
                    wait = min(
                        retry_after + jitter + attempt * retry_after, max_wait_seconds
                    )
                    await asyncio.sleep(wait)
                    continue

                if response.status in {502, 503, 504}:
                    wait = min(2 ** (attempt + 1) + random.uniform(0, 5), max_wait_seconds)
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return await response.json()

        except PermanentAPIError:
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2 ** (attempt + 2) + random.uniform(0, 5), max_wait_seconds)
            await asyncio.sleep(wait)
    return None


# Generic concurrency orchestrator for aiohttp-based calls. Same pattern as
# process_batch_agent (semaphore, abort-on-permanent-error, error dedup) but takes
# an arbitrary async function instead of a pydantic_ai Agent.
# Called by: embed.py to run embedding requests for all papers concurrently.
async def process_batch_aiohttp(
    items: list,
    async_fn: Callable[[Any], Awaitable[Optional[T]]],
    *,
    description: str = "Processing",
    max_concurrent: int = 20,
) -> List[Optional[T]]:
    """Generic concurrency orchestrator for aiohttp-based calls with error dedup and abort."""
    semaphore = asyncio.Semaphore(max_concurrent)
    error_state = {"count": 0, "first_error": None}
    abort = asyncio.Event()

    async def wrapped(item: Any) -> Optional[T]:
        if abort.is_set():
            return None
        async with semaphore:
            if abort.is_set():
                return None
            try:
                return await async_fn(item)
            except PermanentAPIError as e:
                error_state["count"] += 1
                if error_state["count"] == 1:
                    error_state["first_error"] = str(e)
                    logger.error(f"{description} failed: {e}")
                    print(f"{description} failed: {e}")
                abort.set()
                return None
            except Exception as e:
                error_state["count"] += 1
                if error_state["count"] == 1:
                    error_state["first_error"] = str(e)
                    logger.error(f"{description} failed: {e}")
                    print(f"{description} failed: {e}")
                return None

    tasks = [wrapped(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc=description)
    if abort.is_set():
        skipped = sum(1 for r in results if r is None)
        print(f"Permanent error, skipped {skipped}/{len(items)} calls")
    elif error_state["count"] > 1:
        print(f"... error repeated {error_state['count']} times")
    return results


# Build the standard Authorization + Content-Type headers for raw OpenRouter API calls.
# Called by: embed.py (aiohttp calls don't go through the pydantic_ai Agent which adds
# headers automatically via OpenRouterModelSettings).
def make_openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
