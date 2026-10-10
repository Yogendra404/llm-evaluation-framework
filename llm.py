import json, re, time
from openai import OpenAI, RateLimitError, APIConnectionError, InternalServerError

import config

_client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)


def call_llm(prompt, model, system="", max_tokens=1500):
    """Single LLM call with retry/backoff and a small delay for free-tier rate limits."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    last_err = None
    for attempt in range(6):
        try:
            r = _client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens, temperature=0
            )
            time.sleep(config.CALL_DELAY)
            return r.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            last_err = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def parse_json(text):
    """Extract the first JSON object/array from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"No JSON found in: {text[:200]}")
    return json.loads(m.group(0))