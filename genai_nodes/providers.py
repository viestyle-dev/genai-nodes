"""Cloud LLM providers behind one small async interface.

Each provider takes a ``Request`` (prompt, optional image, optional JSON
schema) and returns a ``Response`` (text, parsed JSON if any, usage). Vendor
SDKs are imported lazily, so the plugin loads even where an SDK is missing,
and in a worker thread (``_import_sdk``), so the first request does not stall
the engine's event loop.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import re
from dataclasses import dataclass, field
from typing import Any


async def _import_sdk(module: str):
    """Import a vendor SDK in a worker thread.

    The first import of an SDK takes seconds (anthropic several, google-genai
    two); on the event loop it would stall every other node of the engine for
    that long, and an audio output runs dry.
    """
    return await asyncio.to_thread(importlib.import_module, module)


class ProviderError(Exception):
    """A request failed and should not be retried as-is."""


class RateLimited(ProviderError):
    """The provider refused for rate or quota reasons (HTTP 429)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class Request:
    prompt: str
    api_key: str = field(repr=False)  # never printed
    model: str
    system: str = ""
    image: bytes | None = None
    image_mime: str = "image/jpeg"
    schema: dict | None = None
    max_tokens: int = 1024
    timeout: float = 30.0
    retries_left: int = 1


@dataclass
class Response:
    text: str
    data: Any = None  # parsed JSON when a schema was requested
    usage: dict = field(default_factory=dict)


def extract_json(text: str) -> Any:
    """Parse JSON from a reply that may wrap it in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except ValueError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    return None


def _json_instruction(schema: dict) -> str:
    return (
        "\n\nReply with only a JSON object that matches this JSON Schema, "
        "without code fences or other text:\n" + json.dumps(schema, ensure_ascii=False)
    )


def _strict_schema(schema: Any) -> Any:
    """Copy of a schema with additionalProperties=false on every object."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
            if "properties" in out:
                out.setdefault("required", list(out["properties"]))
        return out
    if isinstance(schema, list):
        return [_strict_schema(v) for v in schema]
    return schema


def _preload_sdk(module: str):
    try:
        importlib.import_module(module)
    except ImportError:
        pass  # reported by the first request, which needs it


class Provider:
    name = ""
    default_model = ""
    default_key_env = ""
    key_url = ""  # where to get an API key
    sdk_module = ""  # imported on first use; preload() imports it ahead

    async def generate(self, req: Request) -> Response:  # pragma: no cover - interface
        raise NotImplementedError

    def preload(self, loop: asyncio.AbstractEventLoop):
        """Import the SDK in a worker thread now, while nothing time-critical runs.

        Even off the event loop, a big SDK's first import holds the GIL for tens of
        milliseconds at a time; done during a request, an audio callback can miss
        its cycle.
        """
        if self.sdk_module:
            loop.run_in_executor(None, _preload_sdk, self.sdk_module)


class GeminiProvider(Provider):
    """Google Gemini API (free tier available) via the google-genai SDK."""

    name = "gemini"
    default_model = "gemini-3.5-flash-lite"
    default_key_env = "GEMINI_API_KEY"
    key_url = "https://aistudio.google.com/apikey"
    sdk_module = "google.genai"

    async def generate(self, req: Request) -> Response:
        try:
            genai = await _import_sdk(self.sdk_module)
        except ImportError as e:
            raise ProviderError("google-genai is not installed") from e

        parts: list[dict] = [{"type": "text", "text": req.prompt}]
        if req.image is not None:
            parts.append(
                {"type": "image", "data": base64.b64encode(req.image).decode("ascii"), "mime_type": req.image_mime}
            )
        kwargs: dict[str, Any] = {
            "model": req.model,
            "input": parts,
            "generation_config": {"max_output_tokens": req.max_tokens},
            "store": False,  # one-off requests: nothing to keep on the server
            # One counted request is one HTTP call: no continuation hops (each bills the input again)
            "automatic_continuation": False,
            "timeout": req.timeout,
        }
        if req.system:
            kwargs["system_instruction"] = req.system
        if req.schema:
            kwargs["response_format"] = {"type": "text", "mime_type": "application/json", "schema": req.schema}

        def call():
            # attempts=1: the runner owns retries and pacing; the SDK must not retry on its own
            client = genai.Client(api_key=req.api_key, http_options={"retry_options": {"attempts": 1}})
            return client.interactions.create(**kwargs)

        try:
            interaction = await asyncio.to_thread(call)
        except Exception as e:  # SDK errors carry an HTTP code
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code == 429 or "RESOURCE_EXHAUSTED" in str(e):
                raise RateLimited(f"gemini: {e}") from e
            raise ProviderError(f"gemini: {e}") from e

        reply = getattr(interaction, "output_text", "") or ""
        usage = getattr(interaction, "usage", None)
        dump = getattr(usage, "model_dump", None)
        usage_dict = dump() if dump is not None else (usage if isinstance(usage, dict) else {})
        return Response(reply, extract_json(reply) if req.schema else None, usage_dict)


class OpenRouterProvider(Provider):
    """OpenRouter (free ':free' models available) via its REST API."""

    name = "openrouter"
    default_model = "openrouter/free"
    default_key_env = "OPENROUTER_API_KEY"
    key_url = "https://openrouter.ai/settings/keys"
    url = "https://openrouter.ai/api/v1/chat/completions"
    sdk_module = "httpx2"

    async def generate(self, req: Request) -> Response:
        httpx2 = await _import_sdk(self.sdk_module)

        prompt = req.prompt + (_json_instruction(req.schema) if req.schema else "")
        content: list[dict] = [{"type": "text", "text": prompt}]
        if req.image is not None:
            data_url = f"data:{req.image_mime};base64," + base64.b64encode(req.image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": content})
        body = {"model": req.model, "messages": messages, "max_tokens": req.max_tokens}

        try:
            async with httpx2.AsyncClient(timeout=req.timeout) as client:
                r = await client.post(self.url, headers={"Authorization": f"Bearer {req.api_key}"}, json=body)
        except Exception as e:
            raise ProviderError(f"openrouter: {e}") from e
        if r.status_code == 429:
            retry = r.headers.get("retry-after")
            raise RateLimited("openrouter: rate limited", float(retry) if retry and retry.isdigit() else None)
        if r.status_code >= 400:
            raise ProviderError(f"openrouter: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            reply = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"openrouter: unexpected reply: {str(data)[:300]}") from e
        return Response(reply, extract_json(reply) if req.schema else None, data.get("usage") or {})


# Models that accept the server-side refusal fallback (scalar "default" form).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


class AnthropicProvider(Provider):
    """Anthropic Claude (paid) via the anthropic SDK."""

    name = "anthropic"
    default_model = "claude-opus-5-5"
    default_key_env = "ANTHROPIC_API_KEY"
    key_url = "https://platform.claude.com/settings/keys"
    sdk_module = "anthropic"

    async def generate(self, req: Request) -> Response:
        try:
            anthropic = await _import_sdk(self.sdk_module)
        except ImportError as e:
            raise ProviderError("anthropic is not installed") from e

        content: list[dict] = []
        if req.image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": req.image_mime,
                        "data": base64.standard_b64encode(req.image).decode("utf-8"),
                    },
                }
            )
        content.append({"type": "text", "text": req.prompt})
        output_config: dict[str, Any] = {"effort": "low"}
        if req.schema:
            output_config["format"] = {"type": "json_schema", "schema": _strict_schema(req.schema)}
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_config": output_config,
        }
        if req.system:
            kwargs["system"] = req.system

        # The runner owns retries and pacing; the SDK must not retry on its own.
        client = anthropic.AsyncAnthropic(api_key=req.api_key, timeout=req.timeout, max_retries=0)
        try:
            if req.model in _FALLBACK_MODELS:
                response = await client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = await client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            retry = e.response.headers.get("retry-after") if e.response is not None else None
            raise RateLimited("anthropic: rate limited", float(retry) if retry and retry.isdigit() else None) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"anthropic: HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"anthropic: connection error: {e}") from e
        finally:
            await client.close()

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise ProviderError(f"anthropic: request declined ({category})")
        reply = next((b.text for b in response.content if b.type == "text"), "")
        usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        return Response(reply, extract_json(reply) if req.schema else None, usage)


PROVIDERS: dict[str, Provider] = {
    p.name: p for p in (GeminiProvider(), OpenRouterProvider(), AnthropicProvider())
}
