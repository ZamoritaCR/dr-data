"""
Intent Classifier -- uses Claude Haiku to understand what the user wants.

Single LLM call that considers both the user message AND the file context
(what type of file is loaded, whether Tableau structure exists, etc.)
to determine intent. No keyword matching. No string contains() checks.
"""

import json
import logging

logger = logging.getLogger(__name__)

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        try:
            import anthropic
            from config.settings import _get_secret
            api_key = _get_secret("ANTHROPIC_API_KEY")
            if api_key:
                _CLIENT = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            logger.warning(f"[INTENT] Could not create client: {e}")
    return _CLIENT


# Intent categories (backward compatible with existing agent code)
BUILD_DASHBOARD = "build_dashboard"
BUILD_POWERBI = "build_powerbi"
BUILD_PDF = "build_pdf"
BUILD_PPTX = "build_pptx"
BUILD_WORD = "build_word"
BUILD_ALL = "build_all"
CHAT = "chat"
ANALYZE = "analyze"

BUILD_INTENTS = {BUILD_DASHBOARD, BUILD_POWERBI, BUILD_PDF, BUILD_PPTX, BUILD_WORD, BUILD_ALL}


_SYSTEM_PROMPT = """\
You are the intent classifier for Dr. Data, a data analytics and \
Tableau-to-Power BI migration tool. Your job: understand what the user \
wants based on what they said and what file they have loaded.

Return ONLY valid JSON. No markdown. No explanation.

Available intents:
- build_powerbi: user wants a Power BI project (.pbip)
- build_dashboard: user wants an HTML dashboard / visualization
- build_pdf: user wants a PDF report
- build_pptx: user wants PowerPoint / slides / presentation / deck
- build_word: user wants a Word document
- build_all: user wants multiple formats or "everything"
- analyze: user wants data analysis, insights, or questions about the data
- chat: user is asking a question or having a conversation

Mode (for build intents):
- replicate: user wants output that mirrors the source file exactly
- creative: user wants something new, impressive, or AI-designed

Rules:
- If a Tableau file is loaded and user says "convert", "migrate", \
"replicate", "same thing", "put this in power bi" -> build_powerbi + replicate
- If a Tableau/image file is loaded and user says "build", "create", \
"make something", "wow me", "impressive" -> build_dashboard + creative
- If CSV/Excel only and user says "power bi" -> build_powerbi + creative
- If CSV/Excel only and user says "dashboard", "visualize", "chart" -> \
build_dashboard + creative
- "report" alone -> build_pdf. "interactive report" -> build_dashboard
- "do it", "go ahead", "yes", "ok", "again", "another", "redo", \
"surprise me", "blow my mind" -> build_dashboard + creative (rebuild)
- "analyze", "insights", "what does this show", "tell me about" -> analyze
- When in doubt between chat and build, prefer build if data is loaded
- People speak naturally. Understand intent, don't match keywords."""


def classify(user_message: str, conversation_context: str = "",
             file_context: dict = None) -> dict:
    """Classify user intent using Claude Haiku with file awareness.

    Args:
        user_message: what the user just said
        conversation_context: last 2-3 messages for follow-up understanding
        file_context: dict with source_type, file_name, has_tableau_spec,
                      has_dataframe, worksheet_count, dashboard_count

    Returns:
        {"intent": str, "mode": "replicate"|"creative", "confidence": float,
         "details": str}
    """
    client = _get_client()
    if not client:
        return _minimal_fallback(user_message)

    # Build file context string
    fc = file_context or {}
    file_info = (
        f"File loaded: {fc.get('source_type', 'none')}\n"
        f"File name: {fc.get('file_name', 'none')}\n"
        f"Has Tableau structure: {fc.get('has_tableau_spec', False)}\n"
        f"Worksheets: {fc.get('worksheet_count', 0)}\n"
        f"Dashboards: {fc.get('dashboard_count', 0)}\n"
        f"Has data: {fc.get('has_dataframe', False)}"
    )

    user_prompt = f"""{file_info}

CONVERSATION CONTEXT:
{conversation_context or '(none)'}

USER MESSAGE: {user_message}

Return this JSON:
{{"intent": "build_powerbi|build_dashboard|build_pdf|build_pptx|build_word|build_all|analyze|chat", "mode": "replicate|creative", "confidence": 0.95}}"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON (may be wrapped in markdown)
        import re
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            intent = result.get("intent", CHAT)
            mode = result.get("mode", "creative")
            confidence = result.get("confidence", 0.9)

            # Validate intent is known
            valid_intents = {
                BUILD_DASHBOARD, BUILD_POWERBI, BUILD_PDF,
                BUILD_PPTX, BUILD_WORD, BUILD_ALL, CHAT, ANALYZE,
            }
            if intent not in valid_intents:
                intent = CHAT

            return {
                "intent": intent,
                "mode": mode,
                "confidence": confidence,
                "details": "haiku",
            }

    except Exception as e:
        logger.warning(f"[INTENT] Classification failed: {e}")

    return _minimal_fallback(user_message)


def _minimal_fallback(msg: str) -> dict:
    """Last-resort fallback when API is unavailable.

    Uses only the most obvious signals -- not a full keyword engine.
    """
    lower = msg.lower()

    if any(k in lower for k in ("power bi", "powerbi", "pbi", "pbip")):
        return {"intent": BUILD_POWERBI, "mode": "creative",
                "confidence": 0.7, "details": "fallback"}

    if "pdf" in lower and "report" in lower:
        return {"intent": BUILD_PDF, "mode": "creative",
                "confidence": 0.7, "details": "fallback"}

    if any(k in lower for k in ("powerpoint", "pptx", "slides", "presentation")):
        return {"intent": BUILD_PPTX, "mode": "creative",
                "confidence": 0.7, "details": "fallback"}

    # Default to dashboard for any build-like language
    if any(k in lower for k in ("build", "create", "make", "generate", "dashboard")):
        return {"intent": BUILD_DASHBOARD, "mode": "creative",
                "confidence": 0.6, "details": "fallback"}

    return {"intent": CHAT, "mode": "creative",
            "confidence": 0.5, "details": "fallback"}
