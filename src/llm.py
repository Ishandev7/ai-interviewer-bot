"""Thin wrapper around the Groq chat API (free tier friendly)."""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def chat(messages, temperature=0.6, json_mode=False, max_tokens=800, retries=3):
    """Send messages to the LLM and return the reply text (or a dict in json_mode)."""
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if MODEL.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = "low"  # fewer hidden reasoning tokens = lower cost and latency

    for attempt in range(retries):
        try:
            reply = _client.chat.completions.create(**kwargs).choices[0].message.content or ""
            return json.loads(reply) if json_mode else reply.strip()
        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  (rate limited, retrying in {wait}s)")
            time.sleep(wait)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                raise
    raise RuntimeError("LLM call failed after retries")
