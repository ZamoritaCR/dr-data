import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys -- check Streamlit secrets first, then .env / os.environ
def _get_secret(key):
    """Read from st.secrets (Streamlit Cloud) or os.environ (.env local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")

ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")

# Validate
def validate_keys():
    issues = []
    if not ANTHROPIC_API_KEY:
        issues.append("ANTHROPIC_API_KEY not set")
    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY not set")
    if not GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY not set (optional -- Gemini features disabled)")
    return issues


def get_available_engines():
    """Return list of available engine names based on configured keys."""
    engines = []
    if ANTHROPIC_API_KEY:
        engines.append("Claude")
    if OPENAI_API_KEY:
        engines.append("OpenAI")
    if GEMINI_API_KEY:
        engines.append("Gemini")
    return engines


def require_at_least_one_engine():
    """Raise RuntimeError if no AI engine has a key configured."""
    engines = get_available_engines()
    if not engines:
        raise RuntimeError(
            "No AI engine configured. Set at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY "
            "in your .env file or Streamlit secrets."
        )
    return engines
