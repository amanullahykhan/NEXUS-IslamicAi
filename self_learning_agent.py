"""
╔══════════════════════════════════════════════════════════════════════╗
║        NEXUS - Self-Learning Islamic AI Agent v4                     ║
║        Gemini + OpenRouter (24 free) + Ollama Local Models           ║
╠══════════════════════════════════════════════════════════════════════╣
║  YOUR OLLAMA MODELS (auto-detected):                                 ║
║    gpt-oss           OpenAI open-weight (best general)               ║
║    llama3.2          Meta Llama 3.2 (fast, good)                     ║
║    llama2-uncensored Unrestricted Llama 2                            ║
║    qwen2.5-coder     Strong reasoning + structured answers           ║
║    qwen3-coder       Best for analytical Islamic research            ║
║    qwen3-vl          Vision + language (can read Arabic images)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PROVIDER PRIORITY:                                                  ║
║    1. Ollama  (local, FREE, unlimited, private, NO rate limits!)     ║
║    2. Gemini  (cloud, fast, free API keys)                           ║
║    3. OpenRouter (24 free cloud models, fallback)                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  INTELLIGENCE:                                                       ║
║    Chain-of-Thought + Islamic scholarly reasoning                    ║
║    Self-critique / Islamic peer-review loop                          ║
║    Ensemble sampling across your local models                        ║
║    Confidence scoring + auto-escalation                              ║
║    SQLite episodic memory + mistake-aware learning                   ║
╚══════════════════════════════════════════════════════════════════════╝

OLLAMA QUICK START:
  Make sure Ollama is running:
    ollama serve          <- run this in a terminal, keep it open

  Check your models:
    ollama list           <- shows all installed models

  If a model is missing, pull it:
    ollama pull llama3.2
    ollama pull qwen2.5-coder

POWERSHELL (optional cloud keys):
  $env:GEMINI_KEY_1="AIza..."       <- https://aistudio.google.com/app/apikey
  $env:OPENROUTER_KEY="sk-or-..."   <- https://openrouter.ai
  python self_learning_agent.py
"""

import os, json, time, sqlite3, hashlib, textwrap, random, re
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import requests
except ImportError:
    raise SystemExit("Run: pip install requests")

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
#  CONFIG
# ════════════════════════════════════════════════════════════════
GEMINI_KEYS:    list = []
OPENROUTER_KEY: str  = ""
OLLAMA_URL:     str  = "http://localhost:11434"

# Your exact Ollama models from the screenshot
# These are used as fallback if auto-detect fails
YOUR_OLLAMA_MODELS = [
    "qwen3-coder",       # best for structured Islamic research answers
    "qwen2.5-coder",     # strong reasoning
    "gpt-oss",           # great general purpose
    "llama3.2",          # fast general
    "llama2-uncensored", # unrestricted (use carefully for sensitive fiqh topics)
    "qwen3-vl",          # vision+language
]

GEMINI_MODELS = [
    "gemini-2.5-flash",
]

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

DB_PATH          = "nexus_memory.db"
ENSEMBLE_SIZE    = 3
MIN_CONFIDENCE   = 0.65
TEMPERATURE_BASE = 0.7
COOLDOWN_SECS    = 65

# ── Speed modes ───────────────────────────────────────────────────
# FAST  = single CoT call, no ensemble, no critique  (2-5s per answer)
# SMART = CoT + self-critique                        (10-20s per answer)
# DEEP  = full ensemble + critique                   (30-90s per answer)
SPEED_MODES = {
    "fast":  {"use_ensemble": False, "use_critique": False},  # chat mode
    "smart": {"use_ensemble": False, "use_critique": True},   # balanced
    "deep":  {"use_ensemble": True,  "use_critique": True},   # training
}
DEFAULT_CHAT_MODE  = "fast"    # used when you ask questions in chat
DEFAULT_TRAIN_MODE = "smart"   # used during /train (good quality, not too slow)

# ── Full Islamic training curriculum ─────────────────────────────
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

# ── Ollama model quality filters ─────────────────────────────────
# Skip these — base models (not instruction-tuned) return raw text
# not proper answers, they just autocomplete
SKIP_MODEL_KEYWORDS = [
    # ── Base/non-instruction models ──────────────────────────────
    "base",        # qwen2.5-coder:1.5b-base — NOT instruction tuned
    "code-base",   # base code models
    # ── Inappropriate content ─────────────────────────────────────
    "nsfw",        # not suitable for Islamic content
    "uncensored",  # llama2-uncensored — unreliable for Islamic content
    "goonsai",     # junk model
    # ── Embedding models (not chat) ───────────────────────────────
    "embed",       # embedding models
    "minilm",      # embedding
    # ── Vision/multimodal models (slow for text-only tasks) ───────
    "vl",          # qwen3-vl, qwen2.5-vl etc.
    "vision",      # any vision model
    "llava",       # LLaVA
    "moondream",   # vision
    "bakllava",    # vision
    # ── Large/slow models that time out ───────────────────────────
    "gpt-oss",     # needs 10.8GB+, always times out
    "gemma2",      # gemma2:latest — consistently timing out (NOT gemma3!)
    "mistral",     # mistral:latest — timing out on your system
]

# Max model size in GB — models larger than this are skipped
MAX_MODEL_SIZE_GB = 5   # keeps only small fast models (llama3.2:3b=2GB, qwen2.5-coder:3b=2GB)

# Timeout per Ollama request — kept short since only small models remain
OLLAMA_TIMEOUT = 60


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════
def _log(msg, color="white"):
    if RICH:
        console.print(f"[{color}]{msg}[/{color}]")
    else:
        print(msg)


class RateLimitError(Exception):
    pass


def _parse_questions(raw: str) -> list:
    """
    Robustly extract a list of question strings from LLM output.
    Handles JSON arrays, numbered lists, and plain text questions.
    Filters out dict-like garbage from base models.
    """
    # 1. Try JSON array first
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            # Filter: only keep plain strings, not dicts
            questions = [
                str(q).strip() for q in parsed
                if isinstance(q, str) and len(q.strip()) > 10
            ]
            if questions:
                return questions
    except Exception:
        pass

    # 2. Try numbered list  (1. ... 2. ... 3. ...)
    numbered = re.findall(r'\d+[\.\)]\s+(.+?)(?=\d+[\.\)]|\Z)', raw, re.DOTALL)
    if numbered:
        questions = [q.strip() for q in numbered if len(q.strip()) > 10]
        if questions:
            return questions

    # 3. Try splitting by lines and picking lines with "?"
    lines = [l.strip() for l in raw.split("\n") if "?" in l and len(l.strip()) > 10]
    if lines:
        return lines[:3]

    return []


# ════════════════════════════════════════════════════════════════
#  OLLAMA DETECTION + SMART FILTERING
# ════════════════════════════════════════════════════════════════
def _is_good_model(model_info: dict) -> tuple[bool, str]:
    """
    Returns (should_use, reason).
    Filters out:
      - Base models (not instruction-tuned)
      - NSFW / junk models
      - Models too large for available RAM
    """
    name = model_info.get("name", "").lower()

    # Check for bad keywords in name
    for kw in SKIP_MODEL_KEYWORDS:
        if kw in name:
            return False, f"skipped ({kw} model)"

    # Check model size — Ollama reports size in bytes
    size_bytes = model_info.get("size", 0)
    size_gb    = size_bytes / (1024 ** 3)
    if size_gb > MAX_MODEL_SIZE_GB:
        return False, f"too large ({size_gb:.1f}GB > {MAX_MODEL_SIZE_GB}GB limit)"

    return True, "ok"


def get_ollama_models() -> list:
    """
    Query Ollama API, filter out base/nsfw/oversized models.
    Returns clean list of model names safe to use for chat.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            all_models  = resp.json().get("models", [])
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
    """Return ALL models including filtered ones — for /ollama list command."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return []


# ════════════════════════════════════════════════════════════════
#  SLOT
# ════════════════════════════════════════════════════════════════
@dataclass
class Slot:
    provider:       str    # ollama | gemini | openrouter
    key:            str    # "local" for ollama, api key for others
    model:          str
    calls:          int    = 0
    errors:         int    = 0
    cooldown_until: object = None

    @property
    def is_available(self):
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False
        return True

    @property
    def label(self):
        return f"{self.provider.upper():12} {self.model[:50]}"

    def mark_rate_limited(self, permanent=False):
        if permanent:
            # Set cooldown so far in future it's effectively disabled this session
            self.cooldown_until = datetime.now() + timedelta(hours=24)
            _log(f"   DISABLED (too slow): {self.label}", "red")
        else:
            self.cooldown_until = datetime.now() + timedelta(seconds=COOLDOWN_SECS)
            _log(f"   RATE-LIMITED: {self.label} -> cooldown {COOLDOWN_SECS}s", "red")
        self.errors += 1

    def mark_success(self):
        self.calls += 1
        self.errors = max(0, self.errors - 1)
        if self.cooldown_until and datetime.now() >= self.cooldown_until:
            self.cooldown_until = None

    def status_str(self):
        if not self.is_available:
            secs = int((self.cooldown_until - datetime.now()).total_seconds())
            return f"COOLING {secs:>3}s"
        return f"WARN  e={self.errors}  " if self.errors else "READY        "


# ════════════════════════════════════════════════════════════════
#  LOAD BALANCER
# ════════════════════════════════════════════════════════════════
class LoadBalancer:
    """
    Priority order:
      Tier 1: Ollama    (local, unlimited, private — YOUR models first!)
      Tier 2: Gemini    (cloud API keys)
      Tier 3: OpenRouter (free cloud fallback)

    Round-robin within each tier.
    On 429 -> slot cools for 65s, next slot takes over.
    Ollama never rate-limits, so it's the BEST primary provider.
    """

    def __init__(self):
        self.slots = []
        self._idx  = 0
        self._build()

    def _build(self):
        self.slots.clear()

        # ── Tier 1: Ollama (YOUR local models — no limits!) ──────
        ollama_models = get_ollama_models()

        if ollama_models:
            _log(f"\n   Ollama running! Found {len(ollama_models)} model(s):", "bold green")
            for m in ollama_models:
                _log(f"      • {m}", "green")
                self.slots.append(Slot("ollama", "local", m))
        else:
            # Ollama not running or no models — warn and suggest
            _log("\n   Ollama not detected. Trying to connect to http://localhost:11434 ...", "yellow")
            _log("   Make sure Ollama is running:  ollama serve", "yellow")
            _log("   Falling back to cloud providers...", "yellow")

        # ── Tier 2: Gemini ────────────────────────────────────────
        for key in GEMINI_KEYS:
            for model in GEMINI_MODELS:
                if model in VALID_GEMINI_MODELS:
                    self.slots.append(Slot("gemini", key, model))
                else:
                    _log(f"   Skipped invalid Gemini model: {model}", "yellow")

        # ── Tier 3: OpenRouter ────────────────────────────────────
        if OPENROUTER_KEY:
            for model in OPENROUTER_FREE_MODELS:
                self.slots.append(Slot("openrouter", OPENROUTER_KEY, model))

        if not self.slots:
            raise ValueError(
                "\nNo providers available!\n\n"
                "OPTION 1 (Recommended - Your local models):\n"
                "  Make sure Ollama is running:\n"
                "    ollama serve\n"
                "  Verify your models:\n"
                "    ollama list\n\n"
                "OPTION 2 (Cloud):\n"
                "  Set GEMINI_KEY_1 environment variable\n"
                "  Get free key: https://aistudio.google.com/app/apikey\n\n"
                "OPTION 3 (Cloud fallback):\n"
                "  Set OPENROUTER_KEY environment variable\n"
                "  Get free key: https://openrouter.ai"
            )

        ol = sum(1 for s in self.slots if s.provider == "ollama")
        g  = sum(1 for s in self.slots if s.provider == "gemini")
        o  = sum(1 for s in self.slots if s.provider == "openrouter")
        _log(f"\n   Pool ready: {ol} Ollama | {g} Gemini | {o} OpenRouter = {len(self.slots)} total slots", "bold green")

    def _pick(self):
        while True:
            available = [s for s in self.slots if s.is_available]
            if available:
                slot = available[self._idx % len(available)]
                self._idx += 1
                return slot
            # All cooling — wait for soonest
            soonest = min(
                (s for s in self.slots if s.cooldown_until),
                key=lambda s: s.cooldown_until
            )
            wait = max(1, (soonest.cooldown_until - datetime.now()).total_seconds() + 1)
            _log(f"   All cloud slots cooling. Waiting {wait:.0f}s... (tip: Ollama never rate-limits!)", "yellow")
            time.sleep(wait)

    def call(self, prompt, temp=TEMPERATURE_BASE):
        """
        Returns (response_text, 'provider/model').
        Tries slots in priority order. Auto-rotates on any failure.
        """
        tried = set()
        while True:
            slot = self._pick()
            if id(slot) in tried and len(tried) >= len(self.slots):
                raise RuntimeError("All slots tried and failed.")
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
                err = str(e)
                is_timeout = "timed out" in err.lower() or "read timeout" in err.lower() or "timeout" in err.lower()
                _log(f"   Error ({slot.label[:40]}): {err[:80]}", "yellow")
                slot.errors += 1
                if is_timeout and slot.errors >= 2:
                    # Timed out twice — permanently disable this session
                    slot.mark_rate_limited(permanent=True)
                elif slot.errors >= 3:
                    slot.mark_rate_limited()

    def refresh_ollama(self):
        """Re-scan Ollama for models (useful if you pulled a new model)."""
        existing = {s.model for s in self.slots if s.provider == "ollama"}
        new_models = get_ollama_models()
        added = 0
        for m in new_models:
            if m not in existing:
                # Insert at front so Ollama stays as Tier 1
                self.slots.insert(0, Slot("ollama", "local", m))
                _log(f"   + New Ollama model: {m}", "green")
                added += 1
        if added == 0:
            _log(f"   Ollama models unchanged: {new_models}", "cyan")
        return new_models

    def add_gemini_key(self, key):
        if key not in GEMINI_KEYS:
            GEMINI_KEYS.append(key)
            added = 0
            for model in GEMINI_MODELS:
                if model in VALID_GEMINI_MODELS:
                    self.slots.append(Slot("gemini", key, model))
                    added += 1
            _log(f"   Added Gemini key -> {len(GEMINI_KEYS)} keys, {added} new slots", "green")

    def add_openrouter_key(self, key):
        global OPENROUTER_KEY
        OPENROUTER_KEY = key
        for model in OPENROUTER_FREE_MODELS:
            self.slots.append(Slot("openrouter", key, model))
        _log(f"   Added OpenRouter -> {len(OPENROUTER_FREE_MODELS)} new slots", "green")

    def status(self):
        rows = []
        for s in self.slots:
            rows.append({
                "status":   s.status_str(),
                "provider": s.provider,
                "model":    s.model,
                "calls":    s.calls,
                "errors":   s.errors,
            })
        return rows


# ════════════════════════════════════════════════════════════════
#  PROVIDER IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════
def _call_ollama(model, prompt, temp):
    """
    Ollama OpenAI-compatible API at http://localhost:11434
    No API key. No rate limits. 100% private.
    Timeout scales with model size to avoid hanging on slow models.
    """
    # Smart timeout: check model size from Ollama, scale accordingly
    timeout = OLLAMA_TIMEOUT
    try:
        info = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        for m in info.get("models", []):
            if m.get("name") == model:
                size_gb = m.get("size", 0) / (1024 ** 3)
                if size_gb < 2:
                    timeout = 45    # tiny models: 45s
                elif size_gb < 5:
                    timeout = 60    # small (3-4B): 60s
                elif size_gb < 10:
                    timeout = 90    # medium (7B): 90s
                else:
                    timeout = 120   # large (13B+): 120s
                break
    except Exception:
        pass   # use default OLLAMA_TIMEOUT

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
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n"
            f"Run: ollama pull {model}"
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty response from Ollama model '{model}'")
    return content


VALID_GEMINI_MODELS = {
    # Gemini 2.5
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro",
    # Gemini 2.0
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    # Gemini 1.5
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
}

def _call_gemini(key, model, prompt, temp):
    if not GEMINI_SDK:
        raise RuntimeError("Run: pip install google-generativeai")
    # Reject obviously wrong model names immediately — no API call needed
    if model not in VALID_GEMINI_MODELS:
        raise RuntimeError(f"Unknown Gemini model '{model}' — skipping")
    genai.configure(api_key=key)
    cfg = genai.types.GenerationConfig(temperature=temp, max_output_tokens=8192)
    try:
        resp = genai.GenerativeModel(model).generate_content(prompt, generation_config=cfg)
        return resp.text
    except Exception as e:
        err = str(e)
        # Rate limit / quota
        if "429" in err or "quota" in err.lower() or "rate" in err.lower() or "limit" in err.lower():
            raise RateLimitError(err)
        # Network / connectivity errors — treat as rate limit so it cools and retries
        if "503" in err or "unavailable" in err.lower() or "handshaker" in err.lower() \
                or "connect" in err.lower() or "grpc" in err.lower() or "tcp" in err.lower():
            raise RateLimitError(f"Gemini network error (no internet?): {err[:120]}")
        raise


def _call_openrouter(key, model, prompt, temp):
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
#  DATA
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
    chat_mode:    str   = DEFAULT_CHAT_MODE   # fast | smart | deep


# ════════════════════════════════════════════════════════════════
#  KNOWLEDGE DATABASE
# ════════════════════════════════════════════════════════════════
class KnowledgeDB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT, query TEXT, strategy TEXT,
                answer TEXT, confidence REAL, critique TEXT,
                provider TEXT DEFAULT '', timestamp TEXT,
                success INTEGER DEFAULT -1
            );
            CREATE TABLE IF NOT EXISTS strategies (
                name TEXT PRIMARY KEY,
                win_count INTEGER DEFAULT 0,
                total INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_qhash ON memories(query_hash);
        """)
        # Auto-migrate old DBs that might be missing provider column
        try:
            self.conn.execute("ALTER TABLE memories ADD COLUMN provider TEXT DEFAULT ''")
            self.conn.commit()
            _log("   DB migrated: added provider column.", "green")
        except Exception:
            pass   # column already exists — that's fine

    def save(self, m):
        self.conn.execute(
            "INSERT INTO memories "
            "(query_hash,query,strategy,answer,confidence,critique,provider,timestamp,success) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(m.query_hash),
                str(m.query),
                str(m.strategy),
                str(m.answer),
                float(m.confidence),
                str(m.critique),   # always string — guards against dict/list
                str(m.provider),
                str(m.timestamp),
                int(m.success),
            )
        )
        self.conn.commit()

    def find_cached(self, qhash):
        row = self.conn.execute(
            "SELECT query_hash,query,strategy,answer,confidence,critique,provider,timestamp,success "
            "FROM memories WHERE query_hash=? AND success=1 "
            "ORDER BY confidence DESC LIMIT 1",
            (qhash,)
        ).fetchone()
        if not row:
            return None
        keys = ["query_hash","query","strategy","answer","confidence",
                "critique","provider","timestamp","success"]
        return Memory(**dict(zip(keys, row)))

    def update(self, qhash, success):
        self.conn.execute(
            "UPDATE memories SET success=? WHERE query_hash=?", (success, qhash)
        )
        self.conn.commit()

    def record_strategy(self, name, win):
        self.conn.execute(
            "INSERT INTO strategies(name,win_count,total) VALUES(?,?,1) "
            "ON CONFLICT(name) DO UPDATE SET win_count=win_count+?,total=total+1",
            (name, 1 if win else 0, 1 if win else 0)
        )
        self.conn.commit()

    def recent_mistakes(self, n=5):
        rows = self.conn.execute(
            "SELECT query,critique FROM memories WHERE success=0 ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [f"Q: {r[0]}\nCritique: {r[1]}" for r in rows]

    def total(self):
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def stats(self):
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
        "tutor":     "You are a brilliant Socratic tutor who breaks problems into clear digestible steps.",
        "engineer":  "You are a senior software engineer writing clean, efficient, well-documented solutions.",
        "critic":    "You are an intellectual critic who challenges assumptions and finds edge cases.",

        "scholar": """You are an Islamic scholar trained in classical and contemporary Islamic sciences.
You follow the methodology of:
  * Maulana Sayyid Abul Ala Maududi (Tafheem ul Quran - deep Quranic tafsir)
  * Engineer Muhammad Ali Mirza (strict Quran + Sahih Hadith, non-sectarian, evidence-based)
  * Ibn Kathir, Imam Nawawi, Ibn Taymiyyah, Ibn Hazm (classical scholarship)
  * Dr. Zakir Naik, Sheikh Yusuf al-Qaradawi (contemporary)

STRICT RULES:
1. Cite Quran: [Surah Name Chapter:Verse] + accurate English translation
2. Cite Hadith: [Book, Hadith#, Narrator, Grade: Sahih/Hasan/Daif/Mawdu]
3. ONLY Sahih or Hasan hadith as primary daleel (Engineer Ali Mirza method)
4. State Ijma (consensus) or Ikhtilaf (disagreement) clearly
5. Give all 4 Madhab positions for Fiqh questions (Hanafi/Shafi'i/Maliki/Hanbali)
6. Use Arabic terms with English translation in parentheses
7. ALWAYS end with verification disclaimer""",

        "mufti": """You are a senior Mufti giving detailed fatawa.
Strict Quran + Sahih Hadith only. Reject blind taqlid.
Show complete daleel (evidence). Give ruling: Wajib/Mustahab/Mubah/Makruh/Haram.
Acknowledge ikhtilaf honestly.
ALWAYS end: Advise consulting a qualified local mufti for personal application.""",

        "ustadh": """You are a warm, patient Islamic teacher (Ustadh).
Inspired by Engineer Ali Mirza's clarity and Maududi's depth.
Explain in simple English first, then introduce Arabic terms.
Use real-life examples. Back up with Quran + Hadith always.
Acknowledge scholarly disagreement. End with verification reminder.""",
    }

    ISLAMIC_CRITIQUE = """You are an Islamic scholarly peer-reviewer doing hadith-style verification.
Check:
- QURAN: Surah names and verse numbers correct? Translation faithful?
- HADITH: Grades correct? No weak hadith used as primary daleel?
- FIQH: All 4 madhab positions covered? Ikhtilaf acknowledged?
- ATTRIBUTION: Maududi, Engineer Ali Mirza, Ibn Kathir correctly quoted?
- VERIFICATION: Does answer end with reminder to verify with scholars/books?"""

    @staticmethod
    def cot(query, persona, mistakes):
        pt = PromptFactory.PERSONAS.get(persona, PromptFactory.PERSONAS["expert"])
        pb = "\n\nPAST MISTAKES TO AVOID:\n" + "\n---\n".join(mistakes[:3]) if mistakes else ""

        if persona in PromptFactory.ISLAMIC_PERSONAS:
            return f"""{pt}{pb}

QUESTION: {query}

Bismillah ir-Rahman ir-Rahim

Answer using this Islamic scholarly structure:

1. UNDERSTAND
   Identify the domain: Aqeedah | Fiqh | Tafsir | Hadith Sciences | Seerah | Akhlaq

2. QURANIC EVIDENCE (Daleel min al-Quran)
   - All relevant ayaat: [Surah Name Chapter:Verse]
   - Accurate English translation
   - Brief tafsir from Maududi (Tafheem), Ibn Kathir, or classical mufassireen

3. HADITH EVIDENCE (Daleel min al-Sunnah)
   - Format: [Book Name, Hadith #, Narrator, Grade: Sahih/Hasan/Daif/Mawdu]
   - ONLY Sahih or Hasan as PRIMARY evidence (Engineer Ali Mirza methodology)
   - Flag any popular but weak or fabricated hadiths on this topic

4. SCHOLARLY OPINIONS (Aqwal al-Ulama)
   - Four Madhabs: Hanafi | Shafi'i | Maliki | Hanbali
   - Classical scholars: Ibn Taymiyyah, Imam Nawawi, Ibn Hazm, Ibn Kathir
   - Contemporary: Maududi, Engineer Muhammad Ali Mirza
   - Is there Ijma (consensus) or Ikhtilaf (disagreement)?

5. CONCLUSION (Al-Khulasah)
   Clear, balanced answer. Conditions, exceptions, nuances.
   Arabic terms with English translation in parentheses.

6. VERIFICATION REMINDER
   End with: "This is AI-generated content. Please verify with:
   Tafheem ul Quran, Sahih Bukhari, Sahih Muslim,
   qualified scholars/muftis, or IslamQA.info / Dar al-Ifta."

7. CONFIDENCE: Rate 0.0-1.0 based on strength of evidence.

Begin:"""

        return f"""{pt}{pb}

TASK: {query}

1. UNDERSTAND   - Restate the problem clearly.
2. DECOMPOSE    - Break into sub-problems.
3. REASON       - Work through methodically. Show all reasoning.
4. SYNTHESIZE   - Final polished answer.
5. CONFIDENCE   - Rate 0.0-1.0.

Begin:"""

    @staticmethod
    def critique(answer, query, persona):
        prefix = PromptFactory.ISLAMIC_CRITIQUE \
                 if persona in PromptFactory.ISLAMIC_PERSONAS \
                 else "You are a rigorous intellectual critic."
        return f"""{prefix}

Original question: {query}

Answer to review:
{answer}

Respond ONLY in this JSON format (no other text):
{{
  "errors": "factual or reference errors found",
  "gaps": "missing evidence or context",
  "improvements": "specific corrections needed",
  "revised_answer": "complete improved answer with all fixes applied",
  "confidence": 0.0
}}"""

    @staticmethod
    def ensemble_judge(candidates, query):
        items = "\n\n".join(
            f"=== CANDIDATE {i+1} ===\n{c}" for i, c in enumerate(candidates)
        )
        return f"""You are an expert judge evaluating multiple answers to the same question.
Pick the best or synthesize the strongest elements from all candidates.

QUESTION: {query}

{items}

Respond ONLY in JSON (no other text):
{{
  "best_candidate": 1,
  "reasoning": "why this is best",
  "final_answer": "complete synthesized best answer",
  "confidence": 0.0
}}"""

    @staticmethod
    def extract_confidence(text):
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
    def __init__(self, config=None):
        self.config = config or AgentConfig()
        self.lb     = LoadBalancer()
        self.db     = KnowledgeDB()
        _log("\nNEXUS v4 ready!", "bold green")

    def _hash(self, q):
        return hashlib.md5(q.strip().lower().encode()).hexdigest()

    def _run_cot(self, query):
        mistakes   = self.db.recent_mistakes() if self.config.use_memory else []
        prompt     = PromptFactory.cot(query, self.config.persona, mistakes)
        text, prov = self.lb.call(prompt, self.config.temperature)
        return text, PromptFactory.extract_confidence(text), prov

    def _run_ensemble(self, query):
        _log(f"   Generating {ENSEMBLE_SIZE} candidates across your models...", "blue")
        candidates, providers = [], []
        mistakes = self.db.recent_mistakes() if self.config.use_memory else []
        for i in range(ENSEMBLE_SIZE):
            temp = min(1.0, max(0.1, self.config.temperature + random.uniform(-0.2, 0.3)))
            try:
                prompt     = PromptFactory.cot(query, self.config.persona, mistakes)
                text, prov = self.lb.call(prompt, temp)
                candidates.append(text)
                providers.append(prov)
                time.sleep(0.5)
            except Exception as e:
                _log(f"   Candidate {i+1} failed: {e}", "yellow")

        if not candidates:
            raise RuntimeError("All ensemble candidates failed.")
        if len(candidates) == 1:
            return candidates[0], PromptFactory.extract_confidence(candidates[0]), providers[0]

        # Judge best answer
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
                    prov
                )
        except Exception:
            pass
        return judgment, 0.75, prov

    def _run_critique(self, answer, query):
        _log("   Running Islamic peer-review critique...", "blue")
        prompt  = PromptFactory.critique(answer, query, self.config.persona)
        raw, _  = self.lb.call(prompt, 0.3)
        revised = raw
        conf    = 0.75
        critique = ""   # always a string
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                # Ensure all values are strings — models sometimes return dicts/lists
                revised  = str(data.get("revised_answer") or raw)
                conf     = float(data.get("confidence") or 0.75)
                errors   = str(data.get("errors")       or "")
                gaps     = str(data.get("gaps")         or "")
                fixes    = str(data.get("improvements") or "")
                critique = f"Errors: {errors}\nGaps: {gaps}\nFixes: {fixes}"
        except Exception:
            critique = raw[:500]   # fallback: store first 500 chars of raw as string
        # Final safety: ensure critique is always a plain string
        if not isinstance(critique, str):
            critique = str(critique)
        if not isinstance(revised, str):
            revised = str(revised)
        return revised, conf, critique

    def _choose_strategy(self):
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

    def query(self, user_query, mode=None):
        """
        mode: "fast" | "smart" | "deep" | None (uses agent default)
        fast  = single CoT call only              ~2-5s
        smart = CoT + self-critique               ~10-20s
        deep  = ensemble + critique               ~30-90s
        """
        # Determine effective mode
        if mode is None:
            mode = getattr(self.config, "chat_mode", DEFAULT_CHAT_MODE)

        use_ensemble = SPEED_MODES[mode]["use_ensemble"]
        use_critique = SPEED_MODES[mode]["use_critique"]

        qhash = self._hash(user_query)

        # Memory cache — always instant
        if self.config.use_memory:
            cached = self.db.find_cached(qhash)
            if cached:
                _log("   From memory.", "green")
                return f"[From Memory]\n{cached.answer}"

        strategy = self._choose_strategy()
        _log(f"   Mode: {mode} | Strategy: {strategy}", "magenta")

        if strategy == "cot_ensemble" and use_ensemble:
            answer, conf, provider = self._run_ensemble(user_query)
        else:
            answer, conf, provider = self._run_cot(user_query)

        # Confidence gate — only escalate in smart/deep mode
        if conf < MIN_CONFIDENCE and use_ensemble:
            _log(f"   Confidence low ({conf:.2f}), escalating...", "yellow")
            answer, conf, provider = self._run_ensemble(user_query)

        # Self-critique — only in smart/deep mode
        critique = ""
        if use_critique:
            answer, conf, critique = self._run_critique(answer, user_query)

        # Save to memory
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
                success    = -1
            ))

        _log(f"   Done — conf={conf:.2f} | {provider}", "green")
        return answer

    def feedback(self, query, good):
        qhash = self._hash(query)
        self.db.update(qhash, 1 if good else 0)
        row = self.db.conn.execute(
            "SELECT strategy FROM memories WHERE query_hash=? ORDER BY id DESC LIMIT 1",
            (qhash,)
        ).fetchone()
        if row:
            self.db.record_strategy(row[0], good)
        _log("Marked correct!" if good else "Noted — avoiding this pattern.", "green" if good else "red")

    def self_train(self, topics, train_mode=DEFAULT_TRAIN_MODE):
        QUESTIONS_PER_TOPIC = 3
        total_topics    = len(topics)
        total_questions = total_topics * QUESTIONS_PER_TOPIC
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
            per_q     = elapsed / completed
            remaining = (total_questions - completed) * per_q
            if remaining < 60:
                return f"{int(remaining)}s"
            return f"{int(remaining // 60)}m {int(remaining % 60)}s"

        def _show(completed, status=""):
            pct  = completed / total_questions * 100 if total_questions else 0
            line = (
                f"  {_bar(pct)} {pct:5.1f}%  "
                f"Q:{completed}/{total_questions}  "
                f"Topics:{done_topics}/{total_topics}  "
                f"Skip:{skipped}  "
                f"ETA:{_eta(completed)}"
            )
            if status:
                line += f"  | {status}"
            _log(line, "cyan")

        _log("\n" + "═" * 65, "bold yellow")
        _log("  NEXUS ISLAMIC SELF-TRAINING", "bold yellow")
        _log(f"  Topics: {total_topics}  |  Questions: {total_questions}  |  Mode: {train_mode}  |  Persona: {self.config.persona}", "yellow")
        _log("═" * 65, "bold yellow")

        for t_idx, topic in enumerate(topics, 1):
            _log(f"\n── TOPIC {t_idx}/{total_topics}: {topic} ──", "bold magenta")
            q_start = (t_idx - 1) * QUESTIONS_PER_TOPIC
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
                    _log(f"   Could not parse questions — skipping topic", "yellow")
                    skipped        += QUESTIONS_PER_TOPIC
                    done_questions += QUESTIONS_PER_TOPIC
                    done_topics    += 1
                    _show(done_questions, "topic skipped")
                    continue

                for q_idx, q in enumerate(questions[:QUESTIONS_PER_TOPIC], 1):
                    q = str(q).strip()
                    completed_so_far = q_start + (q_idx - 1)

                    if len(q) < 15 or "?" not in q:
                        _log(f"   Q{q_idx}: Bad question skipped: {q[:55]}", "yellow")
                        skipped        += 1
                        done_questions += 1
                        _show(q_start + q_idx, f"T{t_idx} Q{q_idx} skipped")
                        continue

                    _log(f"\n  Q{q_idx}/{QUESTIONS_PER_TOPIC}: {q[:78]}{'...' if len(q)>78 else ''}", "white")
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
                skipped        += QUESTIONS_PER_TOPIC
                done_questions += QUESTIONS_PER_TOPIC

            done_topics += 1

        elapsed = time.time() - start_time
        success = total_questions - skipped
        _log("\n" + "═" * 65, "bold green")
        _log("  TRAINING COMPLETE!", "bold green")
        _log(f"  {_bar(100)} 100.0%", "green")
        _log(f"  Questions trained : {success}/{total_questions}", "green")
        _log(f"  Topics covered    : {done_topics}/{total_topics}", "green")
        _log(f"  Skipped           : {skipped}", "yellow" if skipped else "green")
        _log(f"  Total time        : {int(elapsed//60)}m {int(elapsed%60)}s", "green")
        _log(f"  Memories in DB    : {self.db.total()}", "green")
        _log("═" * 65 + "\n", "bold green")

    def show_ollama_info(self):
        """Show all Ollama models with size and whether they're being used."""
        all_models = get_all_ollama_models()
        if not all_models:
            _log("Ollama not running or no models installed.", "yellow")
            _log("Start: ollama serve", "yellow")
            return

        _log(f"\nAll Ollama models ({len(all_models)} total):", "bold cyan")
        active = {s.model for s in self.lb.slots if s.provider == "ollama"}

        for m in all_models:
            name     = m.get("name", "")
            size_gb  = m.get("size", 0) / (1024**3)
            ok, reason = _is_good_model(m)
            if ok and name in active:
                status = "ACTIVE"
                color  = "green"
            elif ok:
                status = "available"
                color  = "cyan"
            else:
                status = f"SKIPPED - {reason}"
                color  = "yellow"
            _log(f"   {status:28} {size_gb:5.1f}GB  {name}", color)

        _log(f"\nRAM limit: {MAX_MODEL_SIZE_GB}GB (edit MAX_MODEL_SIZE_GB in script to change)", "dim")
        _log(f"Timeout per request: {OLLAMA_TIMEOUT}s (edit OLLAMA_TIMEOUT in script to change)", "dim")
        _log("\nNEXUS STATUS", "bold cyan")
        _log(f"   Islamic memories stored: {self.db.total()}", "cyan")
        _log("\n   PROVIDER POOL:", "cyan")
        for s in self.lb.status():
            _log(
                f"      {s['status']:15} {s['provider']:12} "
                f"{s['model'][:48]:48} calls={s['calls']} errors={s['errors']}",
                "cyan"
            )
        _log("\n   STRATEGY WIN RATES:", "cyan")
        for name, s in self.db.stats().items():
            wr = s["wins"] / max(s["total"], 1)
            _log(f"      '{name}': {wr:.0%}  ({s['wins']}/{s['total']})", "cyan")

    def add_provider_interactive(self):
        print("\n   1) Add Gemini key")
        print("   2) Add OpenRouter key")
        print("   3) Refresh Ollama (rescan for new/pulled models)")
        choice = input("   Choice: ").strip()
        if choice == "1":
            key = input("   Gemini API key: ").strip()
            if key:
                self.lb.add_gemini_key(key)
                _save_key(f"GEMINI_KEY_{len(GEMINI_KEYS)}", key)
        elif choice == "2":
            key = input("   OpenRouter API key: ").strip()
            if key:
                self.lb.add_openrouter_key(key)
                _save_key("OPENROUTER_KEY", key)
        elif choice == "3":
            models = self.lb.refresh_ollama()
            if models:
                _log(f"   Ollama models: {models}", "green")
            else:
                _log("   Ollama not running. Start it: ollama serve", "yellow")


# ════════════════════════════════════════════════════════════════
#  ENV / SETUP
# ════════════════════════════════════════════════════════════════
def _save_key(var, val):
    lines = []
    if os.path.exists(".env.nexus"):
        with open(".env.nexus") as f:
            lines = [l for l in f.readlines() if not l.startswith(var + "=")]
    lines.append(f"{var}={val}\n")
    with open(".env.nexus", "w") as f:
        f.writelines(lines)
    _log(f"   Saved {var} to .env.nexus", "green")


def load_env():
    global OPENROUTER_KEY, OLLAMA_URL
    for i in range(1, 20):
        k = os.getenv(f"GEMINI_KEY_{i}", "").strip()
        if k and k not in GEMINI_KEYS:
            GEMINI_KEYS.append(k)
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single and single not in GEMINI_KEYS:
        GEMINI_KEYS.append(single)
    ok = os.getenv("OPENROUTER_KEY", "").strip()
    if ok:
        OPENROUTER_KEY = ok
    ou = os.getenv("OLLAMA_URL", "").strip()
    if ou:
        OLLAMA_URL = ou

    if os.path.exists(".env.nexus"):
        with open(".env.nexus") as f:
            for line in f:
                line = line.strip()
                if "=" not in line or line.startswith("#"):
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
#  BANNER + MAIN
# ════════════════════════════════════════════════════════════════
BANNER = """
+------------------------------------------------------------------+
|  NEXUS v4 - Islamic AI Agent                                     |
|  Ollama (local) + Gemini + OpenRouter (24 free models)           |
|  Quran + Hadith + Fiqh | Maududi | Engineer Ali Mirza            |
+------------------------------------------------------------------+

COMMANDS:
  /mode            -> switch speed (fast=2s | smart=15s | deep=60s)
  /stats           -> provider pool + memory stats
  /ollama          -> show all Ollama models + sizes
  /addkey          -> add Gemini/OpenRouter key or refresh Ollama
  /train           -> Islamic self-training (custom/full/quran/hadith/fiqh...)
  /good  /bad      -> reinforce last answer
  /persona X       -> scholar | mufti | ustadh | expert | tutor | critic
  /quit            -> exit

Default chat mode : FAST (single call, ~2-5s response)
Default train mode: SMART (CoT + critique, best quality/speed balance)
Provider priority : Ollama -> Gemini -> OpenRouter
"""


def main():
    load_env()
    print(BANNER)

    # Show Ollama status before starting
    _log("Checking Ollama...", "cyan")
    ollama_models = get_ollama_models()

    if ollama_models:
        _log(f"Ollama is running! Your models:", "bold green")
        for m in ollama_models:
            _log(f"  • {m}", "green")
    else:
        _log("\nOllama not detected!", "bold red")
        _log("Please run in a separate terminal:  ollama serve", "yellow")
        _log("Then restart NEXUS, or add cloud keys below.\n", "yellow")

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
  /train custom     -> enter your own topics
  /train full       -> full Islamic curriculum (80+ topics, all categories)
  /train quran      -> Quran & tafsir topics
  /train hadith     -> Hadith sciences topics
  /train aqeedah    -> Islamic belief topics
  /train fiqh       -> Fiqh & rulings topics
  /train seerah     -> Prophet's biography topics
  /train akhlaq     -> Islamic character topics
  /train history    -> Islamic history topics
  /train contemporary -> Modern Islamic topics
""")
            sub = input("  Choice: ").strip().lower()

            if sub == "custom":
                raw    = input("  Topics (comma-separated): ").strip()
                topics = [t.strip() for t in raw.split(",") if t.strip()]
            elif sub == "full":
                topics = [t for cat in ISLAMIC_CURRICULUM.values() for t in cat]
                _log(f"  Full curriculum: {len(topics)} topics, {len(topics)*3} questions total", "yellow")
                confirm = input("  This will take a while. Continue? (y/n): ").strip().lower()
                if confirm != "y":
                    topics = []
            elif sub in ISLAMIC_CURRICULUM:
                topics = ISLAMIC_CURRICULUM[sub]
                _log(f"  {sub.title()} curriculum: {len(topics)} topics, {len(topics)*3} questions", "cyan")
            else:
                _log("  Unknown option. Use: custom, full, quran, hadith, aqeedah, fiqh, seerah, akhlaq, history, contemporary", "yellow")
                topics = []

            if topics:
                mode = input("  Training mode (fast/smart/deep) [smart]: ").strip().lower()
                if mode not in SPEED_MODES:
                    mode = DEFAULT_TRAIN_MODE
                agent.self_train(topics, train_mode=mode)

        elif cmd == "/mode":
            print(f"""
  Current chat mode: {getattr(agent.config, 'chat_mode', DEFAULT_CHAT_MODE)}

  /mode fast   -> single call, no critique  (~2-5s)   <- default for chat
  /mode smart  -> CoT + self-critique       (~10-20s)
  /mode deep   -> ensemble + critique       (~30-90s)
""")
            m = input("  New mode: ").strip().lower()
            if m in SPEED_MODES:
                agent.config.chat_mode = m
                _log(f"  Chat mode set to: {m}", "green")
            else:
                _log("  Invalid mode. Choose: fast / smart / deep", "yellow")

        elif user_input.startswith("/persona "):
            p = user_input.split(" ", 1)[1].strip()
            if p in PromptFactory.PERSONAS:
                agent.config.persona = p
                icon = "Scholar" if p in PromptFactory.ISLAMIC_PERSONAS else "General"
                print(f"[{icon}] Persona -> {p}")
            else:
                print(f"Available: {list(PromptFactory.PERSONAS.keys())}")

        else:
            last_query = user_input
            print("\n" + "-" * 65)
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

            elapsed = time.time() - t0
            print(f"\n{'-'*65}")
            print(f"Time: {elapsed:.1f}s  |  /good /bad to teach  |  /stats  |  /addkey")


if __name__ == "__main__":
    main()