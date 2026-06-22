"""DeepInfra VLM client (OpenAI-compatible) with on-disk caching and retry/backoff.

Uses the ``openai`` SDK pointed at DeepInfra's OpenAI-compatible endpoint. Structured
output is requested via ``response_format`` json_schema, with a json_object fallback for
robustness. Responses are cached on disk keyed by a hash of (model, messages, schema) so
repeated runs and dev iterations never re-bill the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

# DeepSeek-V4-Flash is text-only on DeepInfra (rejects image input), so the default is a
# verified vision-capable model. Override with --model on the CLI.
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
BASE_URL = "https://api.deepinfra.com/v1/openai"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Module-level usage accounting (for the operational analysis in the eval report).
USAGE = {"api_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _api_key() -> str | None:
    return os.environ.get("DEEPINFRA_TOKEN") or os.environ.get("DEEPINFRA_API_KEY")


def _cache_key(model: str, messages: list[dict], schema: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "schema": schema},
                      sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".json")


def _read_cache(key: str):
    path = _cache_path(key)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _write_cache(key: str, payload: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


class VLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, *, use_cache: bool = True,
                 max_retries: int = 5, timeout: float = 120.0):
        self.model = model
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None  # lazily created so import/cache-only runs need no key

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily
            key = _api_key()
            if not key:
                raise RuntimeError(
                    "No API key found. Set DEEPINFRA_TOKEN (or DEEPINFRA_API_KEY) to call the model."
                )
            self._client = OpenAI(api_key=key, base_url=BASE_URL, timeout=self.timeout)
        return self._client

    def complete(self, messages: list[dict], schema: dict, *, schema_name: str = "evidence_review") -> dict:
        """Return parsed structured output as a dict. Uses cache when available."""
        key = _cache_key(self.model, messages, schema)
        if self.use_cache:
            cached = _read_cache(key)
            if cached is not None:
                USAGE["cache_hits"] += 1
                cu = cached.get("usage") or {}
                USAGE["prompt_tokens"] += cu.get("prompt_tokens", 0) or 0
                USAGE["completion_tokens"] += cu.get("completion_tokens", 0) or 0
                return cached["parsed"]

        parsed, usage = self._call_with_retry(messages, schema, schema_name)
        USAGE["api_calls"] += 1
        if usage:
            USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            USAGE["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        if self.use_cache:
            _write_cache(key, {"parsed": parsed, "usage": usage})
        return parsed

    def _call_with_retry(self, messages, schema, schema_name):
        client = self._get_client()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                )
                content = resp.choices[0].message.content
                usage = self._usage_dict(resp)
                return json.loads(content), usage
            except Exception as err:  # noqa: BLE001 - broad to handle SDK + HTTP errors
                last_err = err
                msg = str(err).lower()
                # If json_schema isn't accepted, retry once with json_object then plain.
                if "response_format" in msg or "json_schema" in msg:
                    response_format = self._degrade_format(response_format, messages)
                # Backoff on rate limits / transient server errors.
                sleep = min(2 ** attempt, 30)
                time.sleep(sleep)
        raise RuntimeError(f"VLM call failed after {self.max_retries} attempts: {last_err}")

    def _degrade_format(self, current, messages):
        """Fall back from json_schema -> json_object -> none, nudging the model via prompt."""
        if current and current.get("type") == "json_schema":
            # Ask for a JSON object instead and remind the model of the keys.
            return {"type": "json_object"}
        return None  # last resort: rely on prompt to produce JSON

    @staticmethod
    def _usage_dict(resp):
        u = getattr(resp, "usage", None)
        if not u:
            return {}
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }
