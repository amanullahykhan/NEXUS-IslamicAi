"""
NEXUS — Self-Learning Islamic AI Agent
=======================================
Author  : Amanullah Khan
GitHub  : https://github.com/amanullahykhan
Email   : amanullahykhan@gmail.com

A privacy-first, self-learning Islamic AI agent that answers questions
with Quran citations, Hadith references, and 4-Madhab Fiqh analysis.

Scholarly methodology:
  - Maulana Sayyid Abul Ala Maududi (Tafheem ul Quran)
  - Engineer Muhammad Ali Mirza (Quran + Sahih Hadith only, non-sectarian)
  - Ibn Kathir, Imam Nawawi, Ibn Taymiyyah, Ibn Hazm (classical scholarship)

Provider priority: Ollama (local) → Gemini (cloud) → OpenRouter (fallback)
Speed modes:       fast (~2-5s)  |  smart (~15s)  |  deep (~60s)

QUICK START:
  1. Install Ollama:  https://ollama.com
  2. Pull a model:    ollama pull qwen2.5:3b
  3. Start server:    ollama serve
  4. Run:             python self_learning_agent.py

OPTIONAL CLOUD KEYS (free):
  Gemini:      https://aistudio.google.com/app/apikey
  OpenRouter:  https://openrouter.ai

  Save to .env.nexus in the same folder:
    GEMINI_KEY_1=AIza...
    OPENROUTER_KEY=sk-or-...
"""

import os
import re
import json
import time
import random
import hashlib
import sqlite3
import textwrap
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency — run: pip install requests")

try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    RICH = True
except ImportError:
    RICH = False
    console = None

try:
    import google.generativeai as genai
    GEMINI_SDK = True
except ImportError:
    GEMINI_SDK = False


# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

GEMINI_KEYS:    list = []
OPENROUTER_KEY: str  = ""
OLLAMA_URL:     str  = "http://localhost:11434"

# Gemini models (tried in order)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

# Valid Gemini model names — anything else is rejected without making an API call
VALID_GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
}

# OpenRouter free-tier models (no billing required)
OPENROUTER_FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "mistralai/devstral-2512:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "mistralai/mistral-7b-instruct:free",
    "arcee-ai/arcee-trinity-large:free",
    "stepfun/step-3.5-flash:free",
]

# Core parameters
DB_PATH          = "nexus_memory.db"
ENSEMBLE_SIZE    = 3
MIN_CONFIDENCE   = 0.65
TEMPERATURE_BASE = 0.7
COOLDOWN_SECS    = 65

# Speed modes — trade-off between latency and answer quality
SPEED_MODES = {
    "fast":  {"use_ensemble": False, "use_critique": False},   # ~2-5s
    "smart": {"use_ensemble": False, "use_critique": True},    # ~10-20s
    "deep":  {"use_ensemble": True,  "use_critique": True},    # ~30-90s
}
DEFAULT_CHAT_MODE  = "fast"    # used when answering questions interactively
DEFAULT_TRAIN_MODE = "smart"   # used during /train sessions

# Ollama model filtering
# Any model whose name contains one of these keywords is skipped automatically
SKIP_MODEL_KEYWORDS = [
    "base",        # not instruction-tuned — returns raw autocomplete
    "code-base",   # base code models
    "nsfw",        # inappropriate for Islamic content
    "uncensored",  # unreliable for Islamic content
    "goonsai",     # known low-quality model
    "embed",       # embedding models — not chat models
    "minilm",      # embedding models
    "vl",          # vision-language models — slow for text-only tasks
    "vision",      # vision models
    "llava",       # LLaVA vision models
    "moondream",   # vision model
    "bakllava",    # vision model
    "gemma2",      # consistently times out on most hardware
    "mistral",     # slow on CPU-only systems
]

# Models larger than this (in GB) are skipped to prevent out-of-memory errors.
# Increase this value if you have more RAM available (e.g. MAX_MODEL_SIZE_GB = 8).
MAX_MODEL_SIZE_GB = 5

# Default Ollama request timeout in seconds.
# _call_ollama() scales this automatically based on model size.
OLLAMA_TIMEOUT = 60


# ════════════════════════════════════════════════════════════════
#  ISLAMIC TRAINING CURRICULUM
#  94 topics across 8 categories — used by /train
# ════════════════════════════════════════════════════════════════

ISLAMIC_CURRICULUM = {
    "quran": [
        "Quran tafsir methodology",
        "Surah Al-Fatiha meaning and importance",
        "Surah Al-Baqarah key themes",
        "Surah Al-Imran lessons",
        "Surah An-Nisa rulings on women and family",
        "Surah Al-Maida halal and haram",
        "Makki and Madani surahs differences",
        "Quran revelation history and asbab al-nuzul",
        "Tajweed rules and Quran recitation",
        "Quran memorization and its virtues",
        "Quran miracles and scientific signs",
        "Seven Qiraat (readings) of the Quran",
    ],
    "hadith": [
        "Sahih Bukhari major hadiths",
        "Sahih Muslim major hadiths",
        "Hadith classification Sahih Hasan Daif Mawdu",
        "Six major hadith books Kutub al-Sittah",
        "Science of hadith narrators Rijal al-Hadith",
        "Chain of narrators isnad evaluation",
        "Famous fabricated and weak hadiths to avoid",
        "Hadith of Jibreel on Islam Iman Ihsan",
        "Forty Nawawi hadiths",
        "Hadith on intentions niyyah",
        "Hadith on character and akhlaq",
        "Hadith on prayer salah importance",
    ],
    "aqeedah": [
        "Islamic monotheism Tawheed categories",
        "Pillars of Islam five pillars",
        "Pillars of Iman six pillars",
        "99 names of Allah Asma ul Husna",
        "Belief in angels in Islam",
        "Belief in prophets and messengers",
        "Belief in divine books",
        "Belief in Day of Judgment",
        "Belief in Qadar divine decree",
        "Shirk types and how to avoid it",
        "Islamic concept of afterlife Akhirah",
        "Signs of the Day of Judgment",
    ],
    "fiqh": [
        "Islamic prayer salah conditions and rules",
        "Wudu ablution steps and conditions",
        "Ghusl ritual bath when required",
        "Islamic fasting Ramadan rules Hanafi Shafi",
        "Zakat calculation and who receives it",
        "Hajj and Umrah rituals step by step",
        "Halal and haram food Islamic dietary laws",
        "Islamic marriage nikah rules and conditions",
        "Islamic divorce talaq rules",
        "Inheritance rules in Islam Mirath",
        "Business transactions halal and haram",
        "Four madhabs differences Hanafi Shafi Maliki Hanbali",
    ],
    "seerah": [
        "Prophet Muhammad early life in Mecca",
        "Revelation of the Quran first revelation",
        "Hijra migration from Mecca to Madinah",
        "Battle of Badr lessons and outcome",
        "Battle of Uhud lessons and outcome",
        "Battle of Khandaq trench",
        "Treaty of Hudaybiyyah",
        "Conquest of Mecca Fath al-Makkah",
        "Farewell pilgrimage Hajjat al-Wada",
        "Ten companions promised paradise",
        "Wives of Prophet Muhammad mothers of believers",
        "Companions sahabah major contributions",
    ],
    "akhlaq": [
        "Islamic character and manners adab",
        "Patience sabr in Islam Quran and Hadith",
        "Gratitude shukr in Islam",
        "Honesty and truthfulness in Islam",
        "Islamic family values rights of parents",
        "Rights of neighbors in Islam",
        "Islamic brotherhood and unity",
        "Controlling anger in Islam",
        "Generosity and charity sadaqah",
        "Humility tawadu in Islam",
        "Forgiveness and pardoning in Islam",
        "Jealousy and envy hasad prohibition",
    ],
    "history": [
        "Four rightly guided caliphs Khulafa Rashidun",
        "Abu Bakr caliphate achievements",
        "Umar ibn al-Khattab caliphate achievements",
        "Uthman ibn Affan caliphate",
        "Ali ibn Abi Talib caliphate",
        "Umayyad caliphate history",
        "Abbasid caliphate golden age",
        "Islamic golden age science and knowledge",
        "Ottoman empire Islamic history",
        "Islamic scholars Ibn Sina Ibn Rushd Al-Ghazali",
        "Spread of Islam in different regions",
        "Islamic civilization contributions to world",
    ],
    "contemporary": [
        "Engineer Muhammad Ali Mirza methodology",
        "Maulana Maududi Tafheem ul Quran approach",
        "Islam and modern science",
        "Islamic finance and banking",
        "Muslims in the modern world challenges",
        "Islamic perspective on democracy",
        "Rights of non-Muslims in Islamic state",
        "Islam and human rights",
        "Islamic perspective on environment",
        "Media and Islam misconceptions",
    ],
}


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _log(msg: str, color: str = "white"):
    if RICH:
        console.print(f"[{color}]{msg}[/{color}]")
    else:
        print(msg)


class RateLimitError(Exception):
    pass


def _parse_questions(raw: str) -> list:
    """
    Robustly extract a list of question strings from LLM output.
    Handles JSON arrays, numbered lists, and plain lines containing '?'.
    Filters out dict-like garbage returned by base/autocomplete models.
    """
    # 1. Try JSON array
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            questions = [str(q).strip() for q in parsed
                         if isinstance(q, str) and len(q.strip()) > 10]
            if questions:
                return questions
    except Exception:
        pass

    # 2. Numbered list  (1. ...  2. ...  3. ...)
    numbered = re.findall(r'\d+[\.\)]\s+(.+?)(?=\d+[\.\)]|\Z)', raw, re.DOTALL)
    if numbered:
        questions = [q.strip() for q in numbered if len(q.strip()) > 10]
        if questions:
            return questions

    # 3. Any line containing '?'
    lines = [l.strip() for l in raw.split("\n") if "?" in l and len(l.strip()) > 10]
    return lines[:3] if lines else []


# ════════════════════════════════════════════════════════════════
#  OLLAMA DETECTION & FILTERING
# ════════════════════════════════════════════════════════════════

def _is_good_model(model_info: dict) -> tuple[bool, str]:
    """
    Decide whether an Ollama model should be used for chat.
    Returns (should_use: bool, reason: str).
    """
    name    = model_info.get("name", "").lower()
    size_gb = model_info.get("size", 0) / (1024 ** 3)

    for kw in SKIP_MODEL_KEYWORDS:
        if kw in name:
            return False, f"skipped ({kw} model)"

    if size_gb > MAX_MODEL_SIZE_GB:
        return False, f"too large ({size_gb:.1f} GB > {MAX_MODEL_SIZE_GB} GB limit)"

    return True, "ok"


def get_ollama_models() -> list:
    """Return filtered list of Ollama models suitable for chat."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            all_models = resp.json().get("models", [])
            good, skipped = [], []
            for m in all_models:
                ok, reason = _is_good_model(m)
                if ok:
                    good.append(m["name"])
                else:
                    skipped.append(f"{m['name']} ({reason})")
            if skipped:
                _log(f"   Skipped models: {skipped}", "yellow")
            return good
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        _log(f"   Ollama check error: {e}", "yellow")
    return []


def get_all_ollama_models() -> list:
    """Return all Ollama models including filtered ones (used by /ollama)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return []


# ════════════════════════════════════════════════════════════════
#  SLOT — one (provider, key, model) unit in the load-balancer pool
# ════════════════════════════════════════════════════════════════

@dataclass
class Slot:
    provider:       str
    key:            str       # "local" for Ollama; API key for cloud providers
    model:          str
    calls:          int    = 0
    errors:         int    = 0
    cooldown_until: object = None

    @property
    def is_available(self) -> bool:
        return not (self.cooldown_until and datetime.now() < self.cooldown_until)

    @property
    def label(self) -> str:
        return f"{self.provider.upper():12} {self.model[:50]}"

    def mark_rate_limited(self, permanent: bool = False):
        if permanent:
            self.cooldown_until = datetime.now() + timedelta(hours=24)
            _log(f"   DISABLED (too slow): {self.label}", "red")
        else:
            self.cooldown_until = datetime.now() + timedelta(seconds=COOLDOWN_SECS)
            _log(f"   RATE-LIMITED: {self.label} — cooldown {COOLDOWN_SECS}s", "red")
        self.errors += 1

    def mark_success(self):
        self.calls += 1
        self.errors = max(0, self.errors - 1)
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            self.cooldown_until = None

    def status_str(self) -> str:
        if not self.is_available:
            secs = int((self.cooldown_until - datetime.now()).total_seconds())
            return f"COOLING {secs:>3}s"
        return f"WARN  e={self.errors}" if self.errors else "READY"


# ════════════════════════════════════════════════════════════════
#  LOAD BALANCER — three-tier provider pool with auto-failover
# ════════════════════════════════════════════════════════════════

class LoadBalancer:
    """
    Tier 1 — Ollama    : local, unlimited, 100 % private, no rate limits
    Tier 2 — Gemini    : cloud, fast, free API keys available
    Tier 3 — OpenRouter: 24 free cloud models, fallback only

    Round-robins within available slots. On rate-limit or repeated timeout
    the slot enters cooldown and the next slot takes over automatically.
    """

    def __init__(self):
        self.slots: list[Slot] = []
        self._idx  = 0
        self._build()

    def _build(self):
        self.slots.clear()

        # Tier 1: Ollama
        ollama_models = get_ollama_models()
        if ollama_models:
            _log(f"\n   Ollama running! Found {len(ollama_models)} model(s):", "bold green")
            for m in ollama_models:
                _log(f"      • {m}", "green")
                self.slots.append(Slot("ollama", "local", m))
        else:
            _log("\n   Ollama not detected at http://localhost:11434", "yellow")
            _log("   Run:  ollama serve", "yellow")
            _log("   Falling back to cloud providers...\n", "yellow")

        # Tier 2: Gemini
        for key in GEMINI_KEYS:
            for model in GEMINI_MODELS:
                if model in VALID_GEMINI_MODELS:
                    self.slots.append(Slot("gemini", key, model))
                else:
                    _log(f"   Skipped unknown Gemini model: {model}", "yellow")

        # Tier 3: OpenRouter
        if OPENROUTER_KEY:
            for model in OPENROUTER_FREE_MODELS:
                self.slots.append(Slot("openrouter", OPENROUTER_KEY, model))

        if not self.slots:
            raise ValueError(
                "\nNo providers available!\n\n"
                "Option 1 — Local (recommended, free, private):\n"
                "  ollama serve\n"
                "  ollama pull qwen2.5:3b\n\n"
                "Option 2 — Cloud (free API key):\n"
                "  Add GEMINI_KEY_1=AIza... to .env.nexus\n"
                "  Get key: https://aistudio.google.com/app/apikey"
            )

        ol  = sum(1 for s in self.slots if s.provider == "ollama")
        gm  = sum(1 for s in self.slots if s.provider == "gemini")
        orr = sum(1 for s in self.slots if s.provider == "openrouter")
        _log(f"\n   Pool ready: {ol} Ollama | {gm} Gemini | {orr} OpenRouter = {len(self.slots)} slots", "bold green")

    def _pick(self) -> Slot:
        while True:
            available = [s for s in self.slots if s.is_available]
            if available:
                slot = available[self._idx % len(available)]
                self._idx += 1
                return slot
            soonest = min(
                (s for s in self.slots if s.cooldown_until),
                key=lambda s: s.cooldown_until,
            )
            wait = max(1, (soonest.cooldown_until - datetime.now()).total_seconds() + 1)
            _log(f"   All slots cooling — waiting {wait:.0f}s...", "yellow")
            time.sleep(wait)

    def call(self, prompt: str, temp: float = TEMPERATURE_BASE) -> tuple[str, str]:
        """
        Send prompt to the best available provider.
        Returns (response_text, 'provider/model').
        Automatically retries other slots on failure.
        """
        tried: set = set()
        while True:
            slot = self._pick()
            if id(slot) in tried and len(tried) >= len(self.slots):
                raise RuntimeError("All providers tried and failed.")
            tried.add(id(slot))

            _log(f"   [{slot.provider.upper()}] {slot.model}", "dim")
            try:
                if slot.provider == "ollama":
                    text = _call_ollama(slot.model, prompt, temp)
                elif slot.provider == "gemini":
                    text = _call_gemini(slot.key, slot.model, prompt, temp)
                else:
                    text = _call_openrouter(slot.key, slot.model, prompt, temp)
                slot.mark_success()
                return text, f"{slot.provider}/{slot.model}"

            except RateLimitError:
                slot.mark_rate_limited()
                tried.discard(id(slot))

            except Exception as e:
                err        = str(e)
                is_timeout = any(w in err.lower() for w in ("timed out", "read timeout", "timeout"))
                _log(f"   Error ({slot.label[:40]}): {err[:80]}", "yellow")
                slot.errors += 1
                if is_timeout and slot.errors >= 2:
                    slot.mark_rate_limited(permanent=True)
                elif slot.errors >= 3:
                    slot.mark_rate_limited()

    def refresh_ollama(self) -> list:
        """Re-scan Ollama for newly pulled models without restarting NEXUS."""
        existing   = {s.model for s in self.slots if s.provider == "ollama"}
        new_models = get_ollama_models()
        added = 0
        for m in new_models:
            if m not in existing:
                self.slots.insert(0, Slot("ollama", "local", m))
                _log(f"   + New Ollama model added: {m}", "green")
                added += 1
        if not added:
            _log(f"   Ollama models unchanged: {new_models}", "cyan")
        return new_models

    def add_gemini_key(self, key: str):
        if key not in GEMINI_KEYS:
            GEMINI_KEYS.append(key)
            added = sum(
                1 for model in GEMINI_MODELS
                if model in VALID_GEMINI_MODELS
                and not self.slots.append(Slot("gemini", key, model))
            )
            _log(f"   Added Gemini key — {len(GEMINI_KEYS)} key(s), {added} new slots", "green")

    def add_openrouter_key(self, key: str):
        global OPENROUTER_KEY
        OPENROUTER_KEY = key
        for model in OPENROUTER_FREE_MODELS:
            self.slots.append(Slot("openrouter", key, model))
        _log(f"   Added OpenRouter — {len(OPENROUTER_FREE_MODELS)} new slots", "green")

    def status(self) -> list:
        return [
            {"status": s.status_str(), "provider": s.provider,
             "model": s.model, "calls": s.calls, "errors": s.errors}
            for s in self.slots
        ]


# ════════════════════════════════════════════════════════════════
#  PROVIDER IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════

def _call_ollama(model: str, prompt: str, temp: float) -> str:
    """Call Ollama's OpenAI-compatible endpoint. No API key required."""
    # Scale timeout based on model file size
    timeout = OLLAMA_TIMEOUT
    try:
        info = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        for m in info.get("models", []):
            if m.get("name") == model:
                gb = m.get("size", 0) / (1024 ** 3)
                timeout = 45 if gb < 2 else 60 if gb < 5 else 90 if gb < 10 else 120
                break
    except Exception:
        pass

    resp = requests.post(
        f"{OLLAMA_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model":       model,
            "temperature": temp,
            "stream":      False,
            "messages":    [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    if resp.status_code == 404:
        raise RuntimeError(f"Model '{model}' not found. Run: ollama pull {model}")
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:300]}")
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty response from Ollama model '{model}'")
    return content


def _call_gemini(key: str, model: str, prompt: str, temp: float) -> str:
    """Call Google Gemini API via the official SDK."""
    if not GEMINI_SDK:
        raise RuntimeError("Run: pip install google-generativeai")
    if model not in VALID_GEMINI_MODELS:
        raise RuntimeError(f"Unknown Gemini model '{model}' — update VALID_GEMINI_MODELS")
    genai.configure(api_key=key)
    cfg = genai.types.GenerationConfig(temperature=temp, max_output_tokens=8192)
    try:
        resp = genai.GenerativeModel(model).generate_content(prompt, generation_config=cfg)
        return resp.text
    except Exception as e:
        err = str(e)
        if any(w in err.lower() for w in ("429", "quota", "rate", "limit")):
            raise RateLimitError(err)
        # Network / gRPC errors — cool down and retry rather than crashing
        if any(w in err.lower() for w in ("503", "unavailable", "handshaker", "grpc", "tcp")):
            raise RateLimitError(f"Gemini network error: {err[:120]}")
        raise


def _call_openrouter(key: str, model: str, prompt: str, temp: float) -> str:
    """Call OpenRouter API (OpenAI-compatible)."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "nexus-islamic-agent",
            "X-Title":       "NEXUS",
        },
        json={
            "model":       model,
            "temperature": temp,
            "max_tokens":  8192,
            "messages":    [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    if resp.status_code == 429:
        raise RateLimitError(f"OpenRouter 429: {resp.text[:200]}")
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:300]}")
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty response from OpenRouter")
    return content


# ════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ════════════════════════════════════════════════════════════════

@dataclass
class Memory:
    query_hash: str
    query:      str
    strategy:   str
    answer:     str
    confidence: float
    critique:   str
    provider:   str
    timestamp:  str
    success:    int = -1


@dataclass
class AgentConfig:
    use_cot:      bool  = True
    use_ensemble: bool  = True
    use_critique: bool  = True
    use_memory:   bool  = True
    persona:      str   = "scholar"
    temperature:  float = TEMPERATURE_BASE
    chat_mode:    str   = DEFAULT_CHAT_MODE    # fast | smart | deep


# ════════════════════════════════════════════════════════════════
#  KNOWLEDGE DATABASE
# ════════════════════════════════════════════════════════════════

class KnowledgeDB:
    def __init__(self, path: str = DB_PATH):
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT,
                query      TEXT,
                strategy   TEXT,
                answer     TEXT,
                confidence REAL,
                critique   TEXT,
                provider   TEXT DEFAULT '',
                timestamp  TEXT,
                success    INTEGER DEFAULT -1
            );
            CREATE TABLE IF NOT EXISTS strategies (
                name      TEXT PRIMARY KEY,
                win_count INTEGER DEFAULT 0,
                total     INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_qhash ON memories(query_hash);
        """)
        # Migrate older databases that may be missing the provider column
        try:
            self.conn.execute("ALTER TABLE memories ADD COLUMN provider TEXT DEFAULT ''")
            self.conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore

    def save(self, m: Memory):
        """Insert a memory record, coercing all values to safe SQL types."""
        self.conn.execute(
            "INSERT INTO memories "
            "(query_hash,query,strategy,answer,confidence,critique,provider,timestamp,success) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(m.query_hash), str(m.query),     str(m.strategy),
                str(m.answer),     float(m.confidence), str(m.critique),
                str(m.provider),   str(m.timestamp), int(m.success),
            ),
        )
        self.conn.commit()

    def find_cached(self, qhash: str):
        row = self.conn.execute(
            "SELECT query_hash,query,strategy,answer,confidence,critique,"
            "provider,timestamp,success "
            "FROM memories WHERE query_hash=? AND success=1 "
            "ORDER BY confidence DESC LIMIT 1",
            (qhash,),
        ).fetchone()
        if not row:
            return None
        keys = ["query_hash", "query", "strategy", "answer", "confidence",
                "critique", "provider", "timestamp", "success"]
        return Memory(**dict(zip(keys, row)))

    def update(self, qhash: str, success: int):
        self.conn.execute("UPDATE memories SET success=? WHERE query_hash=?", (success, qhash))
        self.conn.commit()

    def record_strategy(self, name: str, win: bool):
        self.conn.execute(
            "INSERT INTO strategies(name,win_count,total) VALUES(?,?,1) "
            "ON CONFLICT(name) DO UPDATE SET win_count=win_count+?,total=total+1",
            (name, 1 if win else 0, 1 if win else 0),
        )
        self.conn.commit()

    def recent_mistakes(self, n: int = 5) -> list:
        rows = self.conn.execute(
            "SELECT query,critique FROM memories WHERE success=0 ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [f"Q: {r[0]}\nCritique: {r[1]}" for r in rows]

    def total(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def stats(self) -> dict:
        rows = self.conn.execute("SELECT * FROM strategies").fetchall()
        return {r[0]: {"wins": r[1], "total": r[2]} for r in rows}


# ════════════════════════════════════════════════════════════════
#  PROMPT FACTORY
# ════════════════════════════════════════════════════════════════

class PromptFactory:

    ISLAMIC_PERSONAS = {"scholar", "mufti", "ustadh"}

    PERSONAS = {
        "expert":    "You are a world-class expert with deep domain knowledge and exceptional clarity.",
        "scientist": "You are a meticulous scientist who shows all work, cites reasoning, and flags uncertainty.",
        "tutor":     "You are a brilliant Socratic tutor who breaks problems into clear, digestible steps.",
        "engineer":  "You are a senior software engineer writing clean, efficient, well-documented solutions.",
        "critic":    "You are an intellectual critic who challenges assumptions and finds edge cases.",

        "scholar": """You are an Islamic scholar trained in classical and contemporary Islamic sciences.

Scholarly methodology:
  * Maulana Sayyid Abul Ala Maududi (Tafheem ul Quran — deep Quranic tafsir)
  * Engineer Muhammad Ali Mirza (strict Quran + Sahih Hadith only, non-sectarian)
  * Ibn Kathir, Imam Nawawi, Ibn Taymiyyah, Ibn Hazm (classical scholarship)
  * Dr. Zakir Naik, Sheikh Yusuf al-Qaradawi (contemporary)

Strict citation rules:
  1. Cite Quran:  [Surah Name Chapter:Verse] + accurate English translation
  2. Cite Hadith: [Book, Hadith #, Narrator, Grade: Sahih/Hasan/Daif/Mawdu]
  3. Use ONLY Sahih or Hasan hadith as primary evidence
  4. State Ijma (scholarly consensus) or Ikhtilaf (disagreement) clearly
  5. Give all 4 Madhab positions for every Fiqh question
  6. Use Arabic terms with English translations in parentheses
  7. Always end with a verification disclaimer""",

        "mufti": """You are a senior Mufti giving detailed fatawa (Islamic legal opinions).
Strict Quran + Sahih Hadith only. Reject blind taqlid.
Show complete daleel (evidence chain). State ruling: Wajib / Mustahab / Mubah / Makruh / Haram.
Acknowledge ikhtilaf (disagreement) among scholars honestly.
Always close by advising the questioner to consult a qualified local mufti.""",

        "ustadh": """You are a warm, patient Islamic teacher (Ustadh).
Explain in simple English first, then introduce Arabic terminology.
Use real-life examples. Back every point with Quran and Hadith.
Acknowledge scholarly disagreement. Close with a verification reminder.""",
    }

    ISLAMIC_CRITIQUE_PROMPT = """You are an Islamic scholarly peer-reviewer performing hadith-style verification.
Check the answer below for:
  - QURAN   : Are Surah names and verse numbers correct? Is the translation faithful?
  - HADITH  : Are grades correct? Is any weak (daif) hadith used as primary evidence?
  - FIQH    : Are all 4 madhab positions covered? Is ikhtilaf acknowledged?
  - SOURCES : Are scholars (Maududi, Engineer Ali Mirza, Ibn Kathir) correctly cited?
  - CLOSING : Does the answer end with a reminder to verify with qualified scholars?"""

    @staticmethod
    def cot(query: str, persona: str, mistakes: list) -> str:
        pt = PromptFactory.PERSONAS.get(persona, PromptFactory.PERSONAS["expert"])
        pb = ("\n\nPAST MISTAKES TO AVOID:\n" + "\n---\n".join(mistakes[:3])) if mistakes else ""

        if persona in PromptFactory.ISLAMIC_PERSONAS:
            return f"""{pt}{pb}

QUESTION: {query}

Bismillah ir-Rahman ir-Rahim

Structure your answer as follows:

1. UNDERSTAND
   Domain: Aqeedah | Fiqh | Tafsir | Hadith Sciences | Seerah | Akhlaq

2. QURANIC EVIDENCE (Daleel min al-Quran)
   Format: [Surah Name Chapter:Verse] + English translation + brief tafsir

3. HADITH EVIDENCE (Daleel min al-Sunnah)
   Format: [Book, Hadith #, Narrator, Grade: Sahih/Hasan/Daif/Mawdu]
   Primary evidence must be Sahih or Hasan only.
   Flag any popular but weak or fabricated hadiths on this topic.

4. SCHOLARLY OPINIONS (Aqwal al-Ulama)
   Four Madhabs : Hanafi | Shafi'i | Maliki | Hanbali
   Classical    : Ibn Taymiyyah, Imam Nawawi, Ibn Kathir
   Contemporary : Maududi, Engineer Muhammad Ali Mirza
   Is there Ijma (consensus) or Ikhtilaf (disagreement)?

5. CONCLUSION (Al-Khulasah)
   Clear, balanced ruling with all conditions and nuances.

6. VERIFICATION REMINDER
   "This is AI-generated content. Please verify with:
   Tafheem ul Quran, Sahih Bukhari, Sahih Muslim,
   qualified scholars/muftis, IslamQA.info, or Dar al-Ifta."

7. CONFIDENCE: Rate 0.0–1.0 based on strength of evidence.

Begin:"""

        return f"""{pt}{pb}

TASK: {query}

1. UNDERSTAND   — Restate the problem clearly.
2. DECOMPOSE    — Break into sub-problems.
3. REASON       — Work through each part methodically.
4. SYNTHESIZE   — Deliver a polished final answer.
5. CONFIDENCE   — Rate 0.0–1.0.

Begin:"""

    @staticmethod
    def critique(answer: str, query: str, persona: str) -> str:
        prefix = (
            PromptFactory.ISLAMIC_CRITIQUE_PROMPT
            if persona in PromptFactory.ISLAMIC_PERSONAS
            else "You are a rigorous intellectual critic."
        )
        return f"""{prefix}

Original question: {query}

Answer to review:
{answer}

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "errors":         "factual or reference errors found",
  "gaps":           "missing evidence or context",
  "improvements":   "specific corrections needed",
  "revised_answer": "complete improved answer with all fixes applied",
  "confidence":     0.0
}}"""

    @staticmethod
    def ensemble_judge(candidates: list, query: str) -> str:
        items = "\n\n".join(
            f"=== CANDIDATE {i+1} ===\n{c}" for i, c in enumerate(candidates)
        )
        return f"""You are an expert judge evaluating multiple answers to the same question.
Synthesise the strongest elements from all candidates into one final answer.

QUESTION: {query}

{items}

Respond ONLY in JSON (no markdown, no extra text):
{{
  "best_candidate": 1,
  "reasoning":      "why this candidate or synthesis is best",
  "final_answer":   "complete synthesised answer",
  "confidence":     0.0
}}"""

    @staticmethod
    def extract_confidence(text: str) -> float:
        for pat in [
            r'"confidence"\s*:\s*([0-9.]+)',
            r'confidence[:\s]+([0-9.]+)',
            r'([0-9.]+)\s*/\s*1\.0',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return min(max(float(m.group(1)), 0.0), 1.0)
                except ValueError:
                    pass
        return 0.5


# ════════════════════════════════════════════════════════════════
#  NEXUS AGENT
# ════════════════════════════════════════════════════════════════

class NexusAgent:
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.lb     = LoadBalancer()
        self.db     = KnowledgeDB()
        _log("\nNEXUS ready!", "bold green")

    # ── Internal reasoning ────────────────────────────────────────

    def _hash(self, q: str) -> str:
        return hashlib.md5(q.strip().lower().encode()).hexdigest()

    def _run_cot(self, query: str) -> tuple:
        mistakes   = self.db.recent_mistakes() if self.config.use_memory else []
        text, prov = self.lb.call(
            PromptFactory.cot(query, self.config.persona, mistakes),
            self.config.temperature,
        )
        return text, PromptFactory.extract_confidence(text), prov

    def _run_ensemble(self, query: str) -> tuple:
        _log(f"   Generating {ENSEMBLE_SIZE} candidates...", "blue")
        candidates, providers = [], []
        mistakes = self.db.recent_mistakes() if self.config.use_memory else []
        for i in range(ENSEMBLE_SIZE):
            temp = min(1.0, max(0.1, self.config.temperature + random.uniform(-0.2, 0.3)))
            try:
                text, prov = self.lb.call(
                    PromptFactory.cot(query, self.config.persona, mistakes), temp
                )
                candidates.append(text)
                providers.append(prov)
                time.sleep(0.5)
            except Exception as e:
                _log(f"   Candidate {i+1} failed: {e}", "yellow")

        if not candidates:
            raise RuntimeError("All ensemble candidates failed.")
        if len(candidates) == 1:
            return candidates[0], PromptFactory.extract_confidence(candidates[0]), providers[0]

        judgment, prov = self.lb.call(
            PromptFactory.ensemble_judge(candidates, query), 0.2
        )
        try:
            m = re.search(r'\{.*\}', judgment, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return (
                    data.get("final_answer", judgment),
                    float(data.get("confidence", 0.75)),
                    prov,
                )
        except Exception:
            pass
        return judgment, 0.75, prov

    def _run_critique(self, answer: str, query: str) -> tuple:
        _log("   Running peer-review critique...", "blue")
        raw, _ = self.lb.call(
            PromptFactory.critique(answer, query, self.config.persona), 0.3
        )
        revised  = raw
        conf     = 0.75
        critique = ""
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data     = json.loads(m.group())
                revised  = str(data.get("revised_answer") or raw)
                conf     = float(data.get("confidence") or 0.75)
                critique = (
                    f"Errors: {data.get('errors', '')}\n"
                    f"Gaps: {data.get('gaps', '')}\n"
                    f"Fixes: {data.get('improvements', '')}"
                )
        except Exception:
            critique = raw[:500]
        return str(revised), conf, str(critique)

    def _choose_strategy(self) -> str:
        stats = self.db.stats()
        if not stats:
            return "cot_ensemble" if self.config.use_ensemble else "cot"
        def wr(name):
            s = stats.get(name, {"wins": 0, "total": 1})
            return s["wins"] / max(s["total"], 1)
        strategies = ["cot", "cot_ensemble", "cot_only"]
        if random.random() < 0.2:
            return random.choice(strategies)
        return max(strategies, key=wr)

    # ── Public API ────────────────────────────────────────────────

    def query(self, user_query: str, mode: str = None) -> str:
        """
        Answer a question using the configured reasoning pipeline.

        mode: "fast" | "smart" | "deep" | None
          fast  — single CoT call                (~2-5s)
          smart — CoT + self-critique            (~10-20s)
          deep  — ensemble sampling + critique   (~30-90s)
        """
        if mode is None:
            mode = getattr(self.config, "chat_mode", DEFAULT_CHAT_MODE)

        use_ensemble = SPEED_MODES[mode]["use_ensemble"]
        use_critique = SPEED_MODES[mode]["use_critique"]
        qhash        = self._hash(user_query)

        if self.config.use_memory:
            cached = self.db.find_cached(qhash)
            if cached:
                _log("   Serving from memory.", "green")
                return f"[From Memory]\n{cached.answer}"

        strategy = self._choose_strategy()
        _log(f"   Mode: {mode} | Strategy: {strategy}", "magenta")

        if strategy == "cot_ensemble" and use_ensemble:
            answer, conf, provider = self._run_ensemble(user_query)
        else:
            answer, conf, provider = self._run_cot(user_query)

        if conf < MIN_CONFIDENCE and use_ensemble:
            _log(f"   Confidence low ({conf:.2f}) — escalating to ensemble...", "yellow")
            answer, conf, provider = self._run_ensemble(user_query)

        critique = ""
        if use_critique:
            answer, conf, critique = self._run_critique(answer, user_query)

        if self.config.use_memory:
            self.db.save(Memory(
                query_hash = qhash,
                query      = user_query,
                strategy   = f"{mode}_{strategy}",
                answer     = answer,
                confidence = conf,
                critique   = critique,
                provider   = provider,
                timestamp  = datetime.now().isoformat(),
                success    = -1,
            ))

        _log(f"   Done — conf={conf:.2f} | {provider}", "green")
        return answer

    def feedback(self, query: str, good: bool):
        """Mark the last answer as correct or incorrect to guide future strategy selection."""
        qhash = self._hash(query)
        self.db.update(qhash, 1 if good else 0)
        row = self.db.conn.execute(
            "SELECT strategy FROM memories WHERE query_hash=? ORDER BY id DESC LIMIT 1",
            (qhash,),
        ).fetchone()
        if row:
            self.db.record_strategy(row[0], good)
        _log("Marked correct!" if good else "Noted — will avoid this pattern.", "green" if good else "red")

    def self_train(self, topics: list, train_mode: str = DEFAULT_TRAIN_MODE):
        """Run self-supervised training over a list of Islamic topics."""
        QPT             = 3   # questions generated per topic
        total_topics    = len(topics)
        total_questions = total_topics * QPT
        done_questions  = 0
        done_topics     = 0
        skipped         = 0
        start_time      = time.time()

        def _bar(pct, width=30):
            filled = int(width * pct / 100)
            return "[" + "█" * filled + "░" * (width - filled) + "]"

        def _eta(completed):
            elapsed = time.time() - start_time
            if completed == 0:
                return "estimating..."
            rem = (total_questions - completed) * (elapsed / completed)
            return f"{int(rem//60)}m {int(rem%60)}s" if rem >= 60 else f"{int(rem)}s"

        def _show(completed, status=""):
            pct  = completed / total_questions * 100 if total_questions else 0
            line = (
                f"  {_bar(pct)} {pct:5.1f}%  "
                f"Q:{completed}/{total_questions}  "
                f"Topics:{done_topics}/{total_topics}  "
                f"Skip:{skipped}  ETA:{_eta(completed)}"
            )
            _log(line + (f"  | {status}" if status else ""), "cyan")

        _log("\n" + "═" * 65, "bold yellow")
        _log("  NEXUS ISLAMIC SELF-TRAINING", "bold yellow")
        _log(
            f"  Topics: {total_topics}  |  Questions: {total_questions}"
            f"  |  Mode: {train_mode}  |  Persona: {self.config.persona}",
            "yellow",
        )
        _log("═" * 65, "bold yellow")

        for t_idx, topic in enumerate(topics, 1):
            _log(f"\n── TOPIC {t_idx}/{total_topics}: {topic} ──", "bold magenta")
            q_start = (t_idx - 1) * QPT
            _show(q_start, f"Generating questions for topic {t_idx}...")

            prompt = (
                f"Generate exactly 3 Islamic knowledge questions about: {topic}\n\n"
                "Each question must require Quran and Hadith references to answer.\n"
                "Return ONLY a JSON array of 3 strings:\n"
                '["Question one?", "Question two?", "Question three?"]\n'
                "No explanation, no markdown. Only the JSON array."
            )

            try:
                raw, _    = self.lb.call(prompt, 0.6)
                questions = _parse_questions(raw)

                if not questions:
                    _log("   Could not parse questions — skipping topic", "yellow")
                    skipped += QPT; done_questions += QPT; done_topics += 1
                    _show(done_questions, "topic skipped")
                    continue

                for q_idx, q in enumerate(questions[:QPT], 1):
                    q = str(q).strip()
                    completed_so_far = q_start + (q_idx - 1)

                    if len(q) < 15 or "?" not in q:
                        _log(f"   Q{q_idx}: Bad question skipped: {q[:55]}", "yellow")
                        skipped += 1; done_questions += 1
                        _show(q_start + q_idx, f"T{t_idx} Q{q_idx} skipped")
                        continue

                    _log(f"\n  Q{q_idx}/{QPT}: {q[:78]}{'...' if len(q) > 78 else ''}", "white")
                    _show(completed_so_far, f"Training T{t_idx} Q{q_idx} of {total_questions}...")

                    try:
                        self.query(q, mode=train_mode)
                    except Exception as e:
                        _log(f"   Error: {e}", "yellow")
                        skipped += 1

                    done_questions += 1
                    _show(q_start + q_idx, f"T{t_idx} Q{q_idx} done ✓")
                    time.sleep(0.3)

            except Exception as e:
                _log(f"   Topic error: {e}", "yellow")
                skipped += QPT; done_questions += QPT

            done_topics += 1

        elapsed = time.time() - start_time
        _log("\n" + "═" * 65, "bold green")
        _log("  TRAINING COMPLETE!", "bold green")
        _log(f"  {_bar(100)} 100.0%", "green")
        _log(f"  Questions trained : {total_questions - skipped}/{total_questions}", "green")
        _log(f"  Topics covered    : {done_topics}/{total_topics}", "green")
        _log(f"  Skipped           : {skipped}", "yellow" if skipped else "green")
        _log(f"  Total time        : {int(elapsed//60)}m {int(elapsed%60)}s", "green")
        _log(f"  Memories in DB    : {self.db.total()}", "green")
        _log("═" * 65 + "\n", "bold green")

    def show_stats(self):
        """Print provider pool status and memory/strategy statistics."""
        _log("\n  NEXUS STATUS", "bold cyan")
        _log(f"  Memories in DB : {self.db.total()}", "cyan")
        _log("\n  PROVIDER POOL:", "cyan")
        for s in self.lb.status():
            _log(
                f"    {s['status']:14} {s['provider']:12} "
                f"{s['model'][:48]:48}  calls={s['calls']}  errors={s['errors']}",
                "cyan",
            )
        _log("\n  STRATEGY WIN RATES:", "cyan")
        for name, s in self.db.stats().items():
            wr = s["wins"] / max(s["total"], 1)
            _log(f"    '{name}': {wr:.0%}  ({s['wins']}/{s['total']})", "cyan")

    def show_ollama_info(self):
        """Print all Ollama models with size and active/skipped status."""
        all_models = get_all_ollama_models()
        if not all_models:
            _log("Ollama not running or no models installed.", "yellow")
            return

        _log(f"\n  Ollama models ({len(all_models)} installed):", "bold cyan")
        active = {s.model for s in self.lb.slots if s.provider == "ollama"}
        for m in all_models:
            name    = m.get("name", "")
            size_gb = m.get("size", 0) / (1024 ** 3)
            ok, reason = _is_good_model(m)
            if ok and name in active:
                status, color = "ACTIVE", "green"
            elif ok:
                status, color = "available (not loaded)", "cyan"
            else:
                status, color = f"SKIPPED — {reason}", "yellow"
            _log(f"    {status:38}  {size_gb:5.1f} GB  {name}", color)

        _log(f"\n  RAM limit : {MAX_MODEL_SIZE_GB} GB  (edit MAX_MODEL_SIZE_GB to change)", "dim")
        _log(f"  Timeout   : {OLLAMA_TIMEOUT}s    (edit OLLAMA_TIMEOUT to change)", "dim")

    def add_provider_interactive(self):
        """Interactive menu to add API keys or rescan Ollama."""
        print("\n  1) Add Gemini API key")
        print("  2) Add OpenRouter API key")
        print("  3) Refresh Ollama models (rescan without restarting)")
        choice = input("  Choice: ").strip()
        if choice == "1":
            key = input("  Gemini key (https://aistudio.google.com/app/apikey): ").strip()
            if key:
                self.lb.add_gemini_key(key)
                _save_key(f"GEMINI_KEY_{len(GEMINI_KEYS)}", key)
        elif choice == "2":
            key = input("  OpenRouter key (https://openrouter.ai): ").strip()
            if key:
                self.lb.add_openrouter_key(key)
                _save_key("OPENROUTER_KEY", key)
        elif choice == "3":
            models = self.lb.refresh_ollama()
            if not models:
                _log("  Ollama not running. Start with: ollama serve", "yellow")


# ════════════════════════════════════════════════════════════════
#  ENVIRONMENT LOADING
# ════════════════════════════════════════════════════════════════

def _save_key(var: str, val: str):
    """Persist an API key to .env.nexus for future sessions."""
    lines = []
    if os.path.exists(".env.nexus"):
        with open(".env.nexus") as f:
            lines = [l for l in f.readlines() if not l.startswith(var + "=")]
    lines.append(f"{var}={val}\n")
    with open(".env.nexus", "w") as f:
        f.writelines(lines)
    _log(f"  Saved {var} to .env.nexus", "green")


def load_env():
    """Load API keys from environment variables and .env.nexus (env vars take priority)."""
    global OPENROUTER_KEY, OLLAMA_URL

    for i in range(1, 20):
        k = os.getenv(f"GEMINI_KEY_{i}", "").strip()
        if k and k not in GEMINI_KEYS:
            GEMINI_KEYS.append(k)
    if k := os.getenv("GEMINI_API_KEY", "").strip():
        if k not in GEMINI_KEYS:
            GEMINI_KEYS.append(k)
    if k := os.getenv("OPENROUTER_KEY", "").strip():
        OPENROUTER_KEY = k
    if k := os.getenv("OLLAMA_URL", "").strip():
        OLLAMA_URL = k

    if os.path.exists(".env.nexus"):
        with open(".env.nexus") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k.startswith("GEMINI_KEY_") and v and v not in GEMINI_KEYS:
                    GEMINI_KEYS.append(v)
                elif k == "OPENROUTER_KEY" and v:
                    OPENROUTER_KEY = v
                elif k == "OLLAMA_URL" and v:
                    OLLAMA_URL = v


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   NEXUS — Self-Learning Islamic AI Agent                         ║
║   Ollama (local) · Google Gemini · OpenRouter (24 free models)   ║
║   Quran + Hadith + Fiqh  ·  Maududi  ·  Engineer Ali Mirza      ║
╠══════════════════════════════════════════════════════════════════╣
║  COMMANDS                                                        ║
║   /mode        fast (2s) | smart (15s) | deep (60s)             ║
║   /train       quran | hadith | fiqh | full | custom            ║
║   /stats       provider pool + memory + strategy stats          ║
║   /ollama      all Ollama models with size and status           ║
║   /addkey      add API key or refresh Ollama                    ║
║   /good /bad   teach NEXUS from the last answer                 ║
║   /persona X   scholar | mufti | ustadh | expert | tutor        ║
║   /quit        exit                                             ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main():
    load_env()
    print(BANNER)

    _log("Checking Ollama...", "cyan")
    models = get_ollama_models()
    if models:
        _log("Ollama running! Active models:", "bold green")
        for m in models:
            _log(f"  • {m}", "green")
    else:
        _log("Ollama not detected. Run 'ollama serve' to enable local inference.", "yellow")

    try:
        agent = NexusAgent()
    except ValueError as e:
        print(f"\n{e}")
        raise SystemExit(1)

    last_query = ""

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAs-salamu alaykum! Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        if cmd == "/quit":
            print("As-salamu alaykum! Goodbye!")
            break

        elif cmd in ("/stats", "/providers"):
            agent.show_stats()

        elif cmd == "/ollama":
            agent.show_ollama_info()

        elif cmd == "/addkey":
            agent.add_provider_interactive()

        elif cmd == "/good" and last_query:
            agent.feedback(last_query, True)

        elif cmd == "/bad" and last_query:
            agent.feedback(last_query, False)

        elif cmd == "/train":
            print("""
  Training options:
    custom       — comma-separated topics of your choice
    full         — complete curriculum (94 topics, 282 questions)
    quran        — Quran & Tafsir          (12 topics)
    hadith       — Hadith sciences         (12 topics)
    aqeedah      — Islamic belief          (12 topics)
    fiqh         — Fiqh & rulings          (12 topics)
    seerah       — Prophet's biography     (12 topics)
    akhlaq       — Islamic character       (12 topics)
    history      — Islamic history         (12 topics)
    contemporary — Modern Islam            (10 topics)
""")
            sub = input("  Choice: ").strip().lower()

            if sub == "custom":
                raw    = input("  Topics (comma-separated): ").strip()
                topics = [t.strip() for t in raw.split(",") if t.strip()]
            elif sub == "full":
                topics = [t for cat in ISLAMIC_CURRICULUM.values() for t in cat]
                _log(f"  Full curriculum: {len(topics)} topics, {len(topics)*3} questions", "yellow")
                if input("  This will take a while. Continue? (y/n): ").strip().lower() != "y":
                    topics = []
            elif sub in ISLAMIC_CURRICULUM:
                topics = ISLAMIC_CURRICULUM[sub]
                _log(f"  {sub.title()}: {len(topics)} topics, {len(topics)*3} questions", "cyan")
            else:
                _log("  Unknown option — try: custom, full, quran, hadith, fiqh...", "yellow")
                topics = []

            if topics:
                mode = input("  Training mode (fast/smart/deep) [smart]: ").strip().lower()
                agent.self_train(topics, train_mode=mode if mode in SPEED_MODES else DEFAULT_TRAIN_MODE)

        elif cmd == "/mode":
            current = getattr(agent.config, "chat_mode", DEFAULT_CHAT_MODE)
            print(f"""
  Current mode: {current}

  fast   — single call, no critique   (~2-5s)    ← default for chat
  smart  — CoT + self-critique        (~10-20s)
  deep   — ensemble + critique        (~30-90s)
""")
            m = input("  New mode: ").strip().lower()
            if m in SPEED_MODES:
                agent.config.chat_mode = m
                _log(f"  Mode set to: {m}", "green")
            else:
                _log("  Invalid. Choose: fast / smart / deep", "yellow")

        elif user_input.startswith("/persona "):
            p = user_input.split(" ", 1)[1].strip()
            if p in PromptFactory.PERSONAS:
                agent.config.persona = p
                tag = "Islamic Scholar" if p in PromptFactory.ISLAMIC_PERSONAS else "General"
                print(f"  [{tag}] Persona → {p}")
            else:
                print(f"  Available: {', '.join(PromptFactory.PERSONAS.keys())}")

        else:
            last_query = user_input
            print("\n" + "─" * 65)
            print("NEXUS thinking...\n")
            t0 = time.time()
            try:
                answer = agent.query(user_input)
            except Exception as e:
                print(f"Error: {e}")
                continue

            print("\nNEXUS:\n")
            if RICH:
                console.print(Markdown(answer))
            else:
                for para in answer.split("\n"):
                    print(textwrap.fill(para, 90) if len(para) > 90 else para)

            print(f"\n{'─'*65}")
            print(f"Time: {time.time()-t0:.1f}s  |  /good /bad  |  /stats  |  /mode  |  /train")


if __name__ == "__main__":
    main()
