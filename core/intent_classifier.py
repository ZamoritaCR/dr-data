"""
Intent Classifier -- uses Claude Haiku to understand what the user wants.

Replaces brittle keyword matching with LLM understanding of human language.
Fast (~300ms) classification that routes to the right pipeline.
"""

import json
import anthropic
from config.settings import _get_secret


_CLIENT = None

def _get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if api_key:
            _CLIENT = anthropic.Anthropic(api_key=api_key)
    return _CLIENT


# Intent categories
BUILD_DASHBOARD = "build_dashboard"
BUILD_POWERBI = "build_powerbi"
BUILD_PDF = "build_pdf"
BUILD_PPTX = "build_pptx"
BUILD_WORD = "build_word"
BUILD_ALL = "build_all"
CHAT = "chat"
ANALYZE = "analyze"

BUILD_INTENTS = {BUILD_DASHBOARD, BUILD_POWERBI, BUILD_PDF, BUILD_PPTX, BUILD_WORD, BUILD_ALL}


def classify(user_message: str, conversation_context: str = "") -> dict:
    """Classify user intent using Claude Haiku.

    Args:
        user_message: what the user just said
        conversation_context: last 2-3 messages for follow-up understanding

    Returns:
        {"intent": str, "confidence": float, "details": str}
    """
    client = _get_client()
    if not client:
        # Fallback to keyword matching if no API key
        return _keyword_fallback(user_message)

    prompt = f"""Classify the user's intent. They are using a data analytics tool that can:
- Build interactive HTML dashboards
- Build Power BI projects (.pbip)
- Build PDF reports
- Build PowerPoint presentations
- Build Word documents
- Analyze/explore data conversationally

The user has already uploaded a data file. They may be asking for something for the first time, or following up on a previous build.

CONVERSATION CONTEXT (last messages):
{conversation_context}

USER MESSAGE: {user_message}

Respond with ONLY one of these JSON objects:
{{"intent":"build_dashboard"}} - user wants an HTML dashboard, visualization, chart, visual output, or is asking for a rebuild/different version of a previous visual output
{{"intent":"build_powerbi"}} - user explicitly wants Power BI / PBI / PBIP
{{"intent":"build_pdf"}} - user wants a PDF report
{{"intent":"build_pptx"}} - user wants PowerPoint / slides / presentation / deck
{{"intent":"build_word"}} - user wants a Word document
{{"intent":"build_all"}} - user wants multiple formats or "everything"
{{"intent":"analyze"}} - user wants data analysis, insights, or questions answered about the data
{{"intent":"chat"}} - user is chatting, asking questions about the tool, or something unrelated to building

RULES:
- "do it", "go ahead", "yes", "ok", "make it", "surprise me", "another one", "try again", "redo it", "something different", "blow my mind" after a build = REBUILD (same intent as the last build, default to build_dashboard)
- "this is terrible", "I hate it", "make it better", "not good enough" after a build = REBUILD
- Any request to CREATE, GENERATE, PRODUCE, MAKE, BUILD something visual = build_dashboard unless they specifically say Power BI, PDF, PPTX, or Word
- When in doubt between chat and build, prefer BUILD if the user has data loaded
- "report" alone = build_pdf. "interactive report" or "visual report" = build_dashboard

Output ONLY the JSON. Nothing else."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Parse JSON
        if raw.startswith("{"):
            result = json.loads(raw)
            return {
                "intent": result.get("intent", CHAT),
                "confidence": 0.95,
                "details": "",
            }
    except Exception as e:
        print(f"[INTENT] Classification failed: {e}")

    return _keyword_fallback(user_message)


def _keyword_fallback(msg: str) -> dict:
    """Ultra-fast keyword fallback if LLM is unavailable."""
    lower = msg.lower()
    if any(k in lower for k in ("power bi", "powerbi", "pbi", "pbip", "pbix")):
        return {"intent": BUILD_POWERBI, "confidence": 0.9, "details": "keyword"}
    if any(k in lower for k in ("dashboard", "html", "interactive", "visual", "chart")):
        return {"intent": BUILD_DASHBOARD, "confidence": 0.8, "details": "keyword"}
    if any(k in lower for k in ("pdf",)):
        return {"intent": BUILD_PDF, "confidence": 0.8, "details": "keyword"}
    if any(k in lower for k in ("powerpoint", "pptx", "slides", "presentation", "deck")):
        return {"intent": BUILD_PPTX, "confidence": 0.8, "details": "keyword"}
    if any(k in lower for k in ("word", "docx", "document")):
        return {"intent": BUILD_WORD, "confidence": 0.8, "details": "keyword"}
    if any(k in lower for k in ("all formats", "all three", "everything")):
        return {"intent": BUILD_ALL, "confidence": 0.8, "details": "keyword"}
    # Follow-up patterns
    if any(k in lower for k in (
        "do it", "build it", "make it", "go ahead", "yes", "ok do",
        "another", "again", "redo", "different", "better", "surprise",
        "blow my mind", "out of this world", "something else",
    )):
        return {"intent": BUILD_DASHBOARD, "confidence": 0.6, "details": "followup_keyword"}
    return {"intent": CHAT, "confidence": 0.5, "details": "default"}
