# 🕌 NEXUS — Self-Learning Islamic AI Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-black?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5-orange?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![SQLite](https://img.shields.io/badge/Memory-SQLite-lightblue?style=for-the-badge&logo=sqlite)

**A privacy-first, self-learning Islamic AI agent powered by local Ollama models + Google Gemini.**  
Answers Islamic questions with Quran citations, Hadith references, and 4-Madhab Fiqh analysis.

[Features](#-features) • [Quick Start](#-quick-start) • [Commands](#-commands) • [Training](#-self-training) • [Architecture](#-architecture) • [Author](#-author)

</div>

---

## 📖 Purpose

NEXUS was created to provide **accurate, evidence-based Islamic knowledge** using the methodology of:

- **Maulana Sayyid Abul Ala Maududi** — Tafheem ul Quran (deep Quranic Tafsir)
- **Engineer Muhammad Ali Mirza** — Strict Quran + Sahih Hadith only, non-sectarian, evidence-based
- **Ibn Kathir, Imam Nawawi, Ibn Taymiyyah, Ibn Hazm** — Classical scholarship
- **Dr. Zakir Naik, Sheikh Yusuf al-Qaradawi** — Contemporary scholarship

Every answer must cite:
- Quran verses with Surah name, chapter, and verse number
- Hadith with book, hadith number, narrator, and grade (Sahih/Hasan/Daif/Mawdu)
- All 4 Madhab positions for Fiqh questions (Hanafi / Shafi'i / Maliki / Hanbali)

The goal is a **100% private, offline-capable** Islamic research assistant that keeps improving itself through self-training.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🦙 **Ollama-First** | Runs entirely on your local machine — no data leaves your PC |
| 🧠 **Self-Learning** | Trains itself on 80+ Islamic topics across 8 categories |
| 📚 **Islamic Personas** | Scholar / Mufti / Ustadh with strict citation rules |
| ⛓️ **Chain-of-Thought** | Step-by-step Islamic scholarly reasoning |
| 🔍 **Self-Critique** | AI peer-review loop verifies Quran/Hadith references |
| 🎯 **Ensemble Sampling** | Multiple models vote on best answer |
| 💾 **Episodic Memory** | SQLite database remembers past Q&A and mistakes |
| ⚡ **Speed Modes** | Fast (2s) / Smart (15s) / Deep (60s) — your choice |
| 🔄 **Multi-Provider** | Ollama → Gemini → OpenRouter auto-failover |
| 🚫 **Smart Filtering** | Auto-skips base models, vision models, and oversized models |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- At least one Ollama model pulled

### 1. Install Ollama & Pull Models

```bash
# Install Ollama (Windows/Mac/Linux)
# https://ollama.com/download

# Start Ollama server
ollama serve

# Pull recommended models (in a new terminal)
ollama pull qwen2.5:3b          # Best for Arabic/Islamic content
ollama pull deepseek-r1:1.5b    # Tiny but brilliant reasoning
ollama pull llama3.2:3b         # Fast general purpose
ollama pull gemma3:4b           # Google's multilingual model
```

### 2. Install Python Dependencies

```bash
pip install requests rich google-generativeai
```

### 3. Run NEXUS

```bash
python self_learning_agent.py
```

### 4. Optional — Add Gemini API Key (Free)

```powershell
# PowerShell (Windows)
$env:GEMINI_KEY_1="AIza..."    # https://aistudio.google.com/app/apikey
python self_learning_agent.py

# Or add it at runtime with /addkey command
```

---

## 💻 Commands

| Command | Description |
|---|---|
| `/mode` | Switch speed: `fast` (2s) \| `smart` (15s) \| `deep` (60s) |
| `/train` | Open training menu — full curriculum or custom topics |
| `/stats` | Show provider pool, memory stats, and win rates |
| `/ollama` | Show all Ollama models with size and active status |
| `/addkey` | Add Gemini/OpenRouter API key, or refresh Ollama models |
| `/good` | Mark last answer as correct — NEXUS learns from this |
| `/bad` | Mark last answer as wrong — NEXUS avoids this pattern |
| `/persona X` | Switch persona: `scholar` \| `mufti` \| `ustadh` \| `expert` \| `tutor` |
| `/quit` | Exit gracefully |

### Example Session

```
You: What is the ruling on music in Islam according to different scholars?

NEXUS thinking...

NEXUS:
Bismillah ir-Rahman ir-Rahim

1. UNDERSTAND
   Domain: Fiqh (Islamic jurisprudence) + Aqwal al-Ulama

2. QURANIC EVIDENCE
   [Surah Luqman 31:6] — "And of the people is he who buys...idle talk..."
   Ibn Kathir: "idle talk" (lahw al-hadith) includes forbidden musical entertainment.

3. HADITH EVIDENCE
   [Sahih Bukhari, Hadith 5590, Abu Malik Al-Ash'ari, Grade: Sahih]
   "There will be people from my Ummah who will consider...musical instruments as lawful..."

4. SCHOLARLY OPINIONS
   Hanafi:  Generally prohibited (haram), especially with singing
   Shafi'i: Prohibited if accompanied by sinful content
   Maliki:  Largely prohibited, some allowance for drums in permissible contexts
   Hanbali: Generally prohibited; Ibn Taymiyyah held strictest view
   ...

Time: 4.2s  |  /good /bad to teach  |  /stats
```

---

## 🎓 Self-Training

NEXUS can train itself on the complete Islamic curriculum. Each topic generates 3 scholarly questions and answers them with full Quran + Hadith citations.

```
/train
```

**Training categories:**

| Category | Topics | Questions |
|---|---|---|
| `quran` | 12 topics (Tafsir, Qiraat, Revelation...) | 36 |
| `hadith` | 12 topics (Bukhari, Muslim, Rijal...) | 36 |
| `aqeedah` | 12 topics (Tawheed, Angels, Qadar...) | 36 |
| `fiqh` | 12 topics (Salah, Zakat, Nikah...) | 36 |
| `seerah` | 12 topics (Battles, Companions...) | 36 |
| `akhlaq` | 12 topics (Sabr, Shukr, Adab...) | 36 |
| `history` | 12 topics (Caliphs, Golden Age...) | 36 |
| `contemporary` | 10 topics (Modern Islam, Finance...) | 30 |
| **`full`** | **All 94 topics** | **282** |

**Training modes:**

```
fast   → single call per question     (~2-5s each, quick coverage)
smart  → CoT + self-critique          (~15s each, recommended)
deep   → ensemble + full critique     (~60s each, highest quality)
```

**Real-time progress display:**

```
── TOPIC 3/8: Islamic prayer salah conditions ──
  [████████░░░░░░░░░░░░░░░░░░░░░░]  25.0%  Q:6/24  Topics:2/8  Skip:0  ETA:14m 30s  | Training T3 Q1 of 24...
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     NEXUS Agent                          │
│  ┌──────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │ Personas │  │PromptFactory│  │  KnowledgeDB      │  │
│  │ Scholar  │  │  CoT prompts│  │  SQLite memory    │  │
│  │ Mufti    │  │  Critique   │  │  Strategy stats   │  │
│  │ Ustadh   │  │  Ensemble   │  │  Mistake learning │  │
│  └──────────┘  └─────────────┘  └───────────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │     Load Balancer      │
              │   Round-robin + CoT    │
              │   Auto-failover        │
              │   Cooldown management  │
              └───┬───────┬───────┬───┘
                  │       │       │
         ┌────────▼──┐ ┌──▼────┐ ┌▼──────────┐
         │  Ollama   │ │Gemini │ │OpenRouter  │
         │  LOCAL    │ │ API   │ │  24 free   │
         │  Tier 1   │ │Tier 2 │ │  Tier 3   │
         │  No limits│ │Fast   │ │  Fallback  │
         └───────────┘ └───────┘ └───────────┘
```

### Provider Priority

| Tier | Provider | Why |
|---|---|---|
| **1** | Ollama (local) | Free, unlimited, private, no rate limits |
| **2** | Google Gemini | Fast, high quality, free API keys available |
| **3** | OpenRouter | 24 free cloud models, fallback only |

### Intelligence Pipeline

```
User Question
     │
     ▼
Memory Cache ──(hit)──► Return cached answer
     │ (miss)
     ▼
Strategy Selector (learns which works best)
     │
     ├── fast:  Single CoT call ──────────────────────► Answer
     ├── smart: CoT ──► Self-Critique ────────────────► Answer
     └── deep:  Ensemble(3) ──► Judge ──► Critique ───► Answer
                                                   │
                                                   ▼
                                            Save to SQLite
```

### Smart Model Filtering

NEXUS automatically skips models that cause problems:

```python
# Skipped automatically:
"base"       # Not instruction-tuned (returns garbage)
"vl"         # Vision models (slow for text tasks)
"embed"      # Embedding models (not chat models)
"nsfw"       # Inappropriate content
"uncensored" # Unreliable for Islamic content
"gemma2"     # Consistently times out
```

Models over `MAX_MODEL_SIZE_GB` (default: 5GB) are also skipped to avoid RAM issues.

---

## ⚙️ Configuration

Edit these constants at the top of `self_learning_agent.py`:

```python
# Provider settings
OLLAMA_URL       = "http://localhost:11434"  # Ollama server URL
OLLAMA_TIMEOUT   = 60                        # Seconds per request
MAX_MODEL_SIZE_GB = 5                        # Skip models larger than this

# Speed mode defaults
DEFAULT_CHAT_MODE  = "fast"    # fast | smart | deep
DEFAULT_TRAIN_MODE = "smart"   # fast | smart | deep

# Quality settings
MIN_CONFIDENCE   = 0.65   # Escalate to ensemble below this
ENSEMBLE_SIZE    = 3      # Number of candidates in deep mode
TEMPERATURE_BASE = 0.7    # Creativity level
COOLDOWN_SECS    = 65     # Rate limit cooldown duration
```

### Environment Variables

```bash
GEMINI_KEY_1=AIza...          # Primary Gemini API key
GEMINI_KEY_2=AIza...          # Secondary key (auto-rotation)
OPENROUTER_KEY=sk-or-...      # OpenRouter API key
OLLAMA_URL=http://localhost:11434  # Custom Ollama URL
```

Keys are also saved locally in `.env.nexus` after first use.

---

## 📦 Recommended Ollama Models

| Model | Size | Speed | Islamic/Arabic | Reasoning | Command |
|---|---|---|---|---|---|
| `qwen2.5:3b` | 1.9GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull qwen2.5:3b` |
| `deepseek-r1:1.5b` | 1.1GB | ⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `ollama pull deepseek-r1:1.5b` |
| `gemma3:4b` | 3.0GB | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | `ollama pull gemma3:4b` |
| `llama3.2:3b` | 2.0GB | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | `ollama pull llama3.2:3b` |
| `phi4-mini` | 2.5GB | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `ollama pull phi4-mini` |
| `deepseek-r1:7b` | 4.7GB | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | `ollama pull deepseek-r1:7b` |

---

## 🗂️ File Structure

```
ai-agent/
├── self_learning_agent.py   # Main agent (single file, fully self-contained)
├── nexus_memory.db          # SQLite memory database (auto-created)
├── .env.nexus               # API keys storage (auto-created, gitignored)
└── README.md                # This file
```

> ⚠️ **Never commit `.env.nexus` to Git** — it contains your API keys. Add it to `.gitignore`.

---

## 🔒 Privacy

- All Ollama inference runs **100% locally** — your Islamic questions never leave your machine
- Gemini/OpenRouter are **optional** cloud fallbacks only
- The SQLite memory database is **local only**
- No telemetry, no tracking, no external logging

---

## ⚠️ Disclaimer

NEXUS is an AI assistant for Islamic research purposes only. It is **not a substitute** for qualified Islamic scholars or muftis. Always verify answers with:

- **Tafheem ul Quran** (Maududi)
- **Sahih Bukhari** and **Sahih Muslim**
- **Qualified scholars and muftis**
- **IslamQA.info** or **Dar al-Ifta**

All AI-generated content should be treated as a starting point for research, not a final ruling (fatwa).

---

## 👨‍💻 Author

**Amanullah Khan**

- 🐙 GitHub: [@amanullahykhan](https://github.com/amanullahykhan)
- 📧 Email: [amanullahykhan@gmail.com](mailto:amanullahykhan@gmail.com)

- Support me : Jazzcash / EasyPaisa : +92-324-3362515


---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Ollama** | Local LLM inference engine |
| **Google Gemini API** | Cloud AI fallback (gemini-2.5-flash) |
| **OpenRouter API** | 24 free cloud models fallback |
| **SQLite** | Episodic memory and strategy learning |
| **Rich** | Beautiful terminal output with colors |
| **Requests** | HTTP calls to Ollama and cloud APIs |
| **google-generativeai** | Official Gemini Python SDK |

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

**بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ**

*Made with ❤️ for the Muslim Ummah*

</div>
