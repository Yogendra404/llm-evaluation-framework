import os
from openai import OpenAI

PROVIDER = os.getenv("PROVIDER", "groq").lower()   # groq | gemini | ollama

PRESETS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
        "prefer": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                   "openai/gpt-oss-20b", "llama-3.1-8b-instant"],
        "delay": 2.5,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "prefer": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
        "delay": 7,
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1", "key_env": None,
        "prefer": ["llama3.2"], "delay": 0,
    },
}
_p = PRESETS[PROVIDER]

BASE_URL = _p["base_url"]
API_KEY = os.getenv(_p["key_env"], "") if _p["key_env"] else "ollama"
CALL_DELAY = _p["delay"]

_NOT_CHAT = ("whisper", "tts", "guard", "embed", "image", "vision", "audio",
             "imagen", "veo", "aqa", "safeguard", "orpheus", "playai", "compound")


def _available_models():
    try:
        ids = [m.id for m in OpenAI(api_key=API_KEY, base_url=BASE_URL).models.list().data]
        return [i.replace("models/", "") for i in ids]
    except Exception as e:
        print(f"[config] Could not list models for {PROVIDER}: {e}")
        return []


def _pick(avail):
    for name in _p["prefer"]:
        if name in avail:
            return name
    chat = [i for i in avail if not any(b in i.lower() for b in _NOT_CHAT)]
    return chat[0] if chat else _p["prefer"][0]


_env_app, _env_judge = os.getenv("APP_MODEL"), os.getenv("JUDGE_MODEL")
if _env_app and _env_judge:
    APP_MODEL, JUDGE_MODEL = _env_app, _env_judge
else:
    _avail = _available_models()
    _auto = _pick(_avail)
    APP_MODEL = _env_app or _auto
    JUDGE_MODEL = _env_judge or _auto
    print(f"[config] provider={PROVIDER} app={APP_MODEL} judge={JUDGE_MODEL}")
    if _avail:
        print(f"[config] models your key can use: {', '.join(sorted(_avail))}")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = int(os.getenv("TOP_K", "3"))

DOCS_DIR = "data/docs"
TESTSET = "data/testset.json"
RESULTS_DIR = "results"