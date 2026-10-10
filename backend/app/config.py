"""
Central configuration for VoicePilot.

All settings are read from environment variables (or a `.env` file in the
backend folder) so nothing secret is hard-coded.
python-dotenv usage: https://github.com/theskumar/python-dotenv
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

# ---------------------------------------------------------------- LLM ------
# Every provider below speaks the OpenAI-compatible Chat Completions API, so
# a single client (the official `openai` package) works for all of them.
#   ollama -> free, runs locally (https://ollama.com/blog/openai-compatibility)
#   groq   -> free tier in the cloud (https://console.groq.com/docs/openai)
#   openai -> paid, but gpt-4o-mini is very cheap
#   none   -> no LLM at all; a local intent classifier + rules are used
PROVIDERS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": "llama3.2:3b",
        "api_key_env": None,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
    },
    "openai": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "none": {"base_url": None, "model": "-", "api_key_env": None},
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
if LLM_PROVIDER not in PROVIDERS:
    LLM_PROVIDER = "none"
LLM_MODEL = os.getenv("LLM_MODEL") or PROVIDERS[LLM_PROVIDER]["model"]
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v4")
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "5"))

# ------------------------------------------------------------- speech ------
# faster-whisper model sizes: tiny, base, small, medium, large-v3
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None  # "" = auto-detect

# ------------------------------------------------------------ actions ------
# DRY_RUN=1 -> actions are validated and logged but nothing is launched.
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
NOTES_DIR = Path(os.getenv("NOTES_DIR", str(Path.home() / "VoicePilot Notes")))

# ------------------------------------------------------------ storage ------
DB_PATH = Path(os.getenv("DB_PATH", str(BACKEND_DIR / "voicepilot.db")))
CLASSIFIER_PATH = BACKEND_DIR / "models" / "intent_classifier.joblib"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# ------------------------------------------------------ agent + RAG ------
# The agent handles requests that need several steps or the user's documents.
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "1") == "1"
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "4"))
AGENT_MAX_ACTIONS = int(os.getenv("AGENT_MAX_ACTIONS", "2"))  # tools that change something

# Read-only document search (RAG). Put .pdf / .docx / .txt / .md files in DOCUMENTS_DIR.
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(PROJECT_DIR / "documents")))
RAG_INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", str(BACKEND_DIR / "rag_index.json")))
# auto = Ollama embeddings if available, otherwise TF-IDF | ollama | tfidf
RAG_BACKEND = os.getenv("RAG_BACKEND", "auto").lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
