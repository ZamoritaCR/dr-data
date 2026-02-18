"""
Gemini Engine - Google Gemini integration for large-context analysis,
vision (Tableau screenshot parsing), and document understanding.

Used as a third engine alongside Claude and OpenAI for:
- Large dataset analysis (up to 800K tokens context window)
- Tableau screenshot -> structured layout description
- Document ingestion (PDF, Word, etc.)
- Chat failover when Claude/OpenAI hit rate limits
"""

import os
import io
import json
import time

try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


def _get_gemini_key():
    """Resolve Gemini API key from Streamlit secrets or environment."""
    # Streamlit secrets first
    try:
        import streamlit as st
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if key in st.secrets:
                return st.secrets[key]
    except Exception:
        pass
    # Fall back to os.environ / .env
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _retry_generate(model, content, max_retries=3):
    """Call model.generate_content with exponential-backoff retries."""
    for attempt in range(max_retries):
        try:
            return model.generate_content(content)
        except Exception as e:
            err = str(e).lower()
            # Only retry on transient / rate-limit errors
            if attempt < max_retries - 1 and any(
                kw in err for kw in ("429", "rate", "resource", "503", "500", "overloaded")
            ):
                wait = 2 ** attempt
                print(f"[GEMINI] Retry {attempt + 1}/{max_retries} after {wait}s -- {e}")
                time.sleep(wait)
            else:
                raise


class GeminiEngine:
    """Google Gemini integration for analysis, vision, and failover chat."""

    def __init__(self):
        self.available = False
        self.model = None

        if not _HAS_GENAI:
            print("[GEMINI] google-generativeai not installed. Gemini engine disabled.")
            return

        api_key = _get_gemini_key()
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            self.available = True
            print("[OK] Gemini engine ready (model: gemini-2.0-flash)")
        else:
            print("[GEMINI] No API key found. Gemini engine disabled.")

    def is_available(self):
        """Check if Gemini is configured and ready."""
        return self.available

    # ------------------------------------------------------------------ #
    #  Large dataset analysis                                              #
    # ------------------------------------------------------------------ #

    def analyze_large_dataset(self, df, question=None):
        """Analyze a full DataFrame using Gemini's large context window.

        Args:
            df: pandas DataFrame (up to ~500K rows).
            question: optional specific question to answer about the data.

        Returns:
            str: analysis text, or None on failure.
        """
        if not self.available:
            return None

        try:
            # Convert to CSV -- Gemini can handle ~800K tokens (~3.2M chars)
            csv_str = df.to_csv(index=False, max_rows=500000)

            prompt = (
                "You are Dr. Data, a senior data analyst. "
                "Analyze this complete dataset. Find the 3-5 most compelling "
                "insights with specific numbers. Look for: trends over time, "
                "correlations between columns, outliers, clusters, anomalies, "
                "distribution patterns. Be specific with percentages and values. "
                "No bullet points -- write naturally like a sharp colleague."
            )

            if question:
                prompt += (
                    f"\n\nThe user specifically asked: {question}\n"
                    "Answer this question using the data, then add any other "
                    "interesting findings."
                )

            prompt += f"\n\nDATASET ({len(df):,} rows, {len(df.columns)} columns):\n{csv_str}"

            t0 = time.time()
            response = _retry_generate(self.model, prompt)
            elapsed = time.time() - t0
            print(f"[GEMINI] Dataset analysis: {elapsed:.1f}s, "
                  f"{len(df):,} rows analyzed")
            return response.text

        except Exception as e:
            print(f"[GEMINI] Dataset analysis failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Image / vision analysis                                             #
    # ------------------------------------------------------------------ #

    def analyze_image(self, image_bytes, prompt):
        """Send an image to Gemini vision with a prompt.

        Args:
            image_bytes: raw bytes of the image (PNG, JPG, etc.).
            prompt: what to analyze about the image.

        Returns:
            str: response text, or None on failure.
        """
        if not self.available:
            return None

        try:
            response = _retry_generate(self.model, [
                prompt,
                {"mime_type": "image/png", "data": image_bytes},
            ])
            return response.text

        except Exception as e:
            print(f"[GEMINI] Image analysis failed: {e}")
            return None

    def describe_tableau_screenshot(self, image_bytes):
        """Analyze a Tableau dashboard screenshot and return structured layout.

        Args:
            image_bytes: raw bytes of the screenshot.

        Returns:
            dict: structured layout description, or None on failure.
        """
        if not self.available:
            return None

        prompt = (
            "You are analyzing a screenshot of a Tableau dashboard. "
            "Describe the EXACT visual layout in detail:\n"
            "- What chart types are visible and where are they positioned "
            "(top-left, bottom-right, etc.)\n"
            "- What colors are used for each chart element\n"
            "- What titles and labels are visible\n"
            "- What filters or parameters are shown\n"
            "- What KPIs or summary numbers are displayed and where\n"
            "- Approximate proportions of each visual "
            "(takes up 50% width, 30% height, etc.)\n"
            "- Any text tables, legends, or annotations\n\n"
            "Return your description as a structured JSON with this format:\n"
            "{\n"
            '  "dashboard_title": "string",\n'
            '  "dimensions": {"width_pct": 100, "height_pct": 100},\n'
            '  "visuals": [\n'
            "    {\n"
            '      "type": "bar_chart|line_chart|area_chart|pie_chart|map|'
            'table|kpi_card|scatter|treemap|heatmap",\n'
            '      "title": "string",\n'
            '      "position": {"x_pct": 0, "y_pct": 0, '
            '"width_pct": 50, "height_pct": 50},\n'
            '      "description": "what data it shows",\n'
            '      "colors": ["#hex1", "#hex2"],\n'
            '      "axes": {"x": "field name", "y": "field name"}\n'
            "    }\n"
            "  ],\n"
            '  "filters": [{"name": "string", '
            '"type": "dropdown|slider|date_range", '
            '"position": "top|left|right"}],\n'
            '  "color_scheme": "description of overall color palette"\n'
            "}\n\n"
            "Return ONLY the JSON. No markdown fences. No commentary."
        )

        try:
            raw = self.analyze_image(image_bytes, prompt)
            if not raw:
                return None

            # Strip markdown fences if present
            import re
            text = raw.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
                text = text.strip()

            result = json.loads(text)
            print(f"[GEMINI] Tableau screenshot: "
                  f"{len(result.get('visuals', []))} visuals detected")
            return result

        except json.JSONDecodeError as e:
            print(f"[GEMINI] Tableau screenshot JSON parse failed: {e}")
            return None
        except Exception as e:
            print(f"[GEMINI] Tableau screenshot analysis failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Document analysis                                                   #
    # ------------------------------------------------------------------ #

    def analyze_document(self, file_bytes, filename, question=None):
        """Analyze a document (PDF, Word, etc.) using Gemini.

        Args:
            file_bytes: raw bytes of the document.
            filename: original filename (used for MIME type detection).
            question: optional specific question about the document.

        Returns:
            str: analysis text, or None on failure.
        """
        if not self.available:
            return None

        try:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            mime_map = {
                "pdf": "application/pdf",
                "docx": "application/vnd.openxmlformats-officedocument"
                        ".wordprocessingml.document",
                "doc": "application/msword",
                "xlsx": "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet",
                "xls": "application/vnd.ms-excel",
                "pptx": "application/vnd.openxmlformats-officedocument"
                        ".presentationml.presentation",
                "csv": "text/csv",
                "txt": "text/plain",
                "json": "application/json",
            }
            mime = mime_map.get(ext, "application/octet-stream")

            uploaded = genai.upload_file(io.BytesIO(file_bytes), mime_type=mime)

            prompt = (
                "You are Dr. Data, a senior analyst. Read this entire document "
                "thoroughly. Provide:\n"
                "1) A concise executive summary (3-4 sentences).\n"
                "2) The key data points, metrics, or findings with specific "
                "numbers.\n"
                "3) Any tables or datasets you can extract.\n"
                "4) Recommendations or action items if apparent."
            )

            if question:
                prompt += (
                    f"\n\nThe user specifically asked: {question}\n"
                    "Answer this question first, then provide the analysis above."
                )

            t0 = time.time()
            response = _retry_generate(self.model, [prompt, uploaded])
            elapsed = time.time() - t0
            print(f"[GEMINI] Document analysis ({filename}): {elapsed:.1f}s")
            return response.text

        except Exception as e:
            print(f"[GEMINI] Document analysis failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Chat (failover)                                                     #
    # ------------------------------------------------------------------ #

    def chat(self, system_prompt, message, conversation_history=None):
        """General-purpose chat -- failover when Claude/OpenAI hit limits.

        Args:
            system_prompt: system-level instructions.
            message: the user's message.
            conversation_history: optional list of {"role": ..., "content": ...}.

        Returns:
            str: response text, or None on failure.
        """
        if not self.available:
            return None

        try:
            # Build history for Gemini chat session
            history = []
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if not content:
                        continue
                    # Gemini uses "user" and "model" roles
                    g_role = "model" if role == "assistant" else "user"
                    history.append({"role": g_role, "parts": [content]})

            chat_session = self.model.start_chat(history=history)

            # Prepend system prompt to the user message
            full_message = f"[SYSTEM]: {system_prompt}\n\n{message}"

            t0 = time.time()
            for attempt in range(3):
                try:
                    response = chat_session.send_message(full_message)
                    break
                except Exception as e:
                    err = str(e).lower()
                    if attempt < 2 and any(
                        kw in err for kw in ("429", "rate", "resource", "503", "500", "overloaded")
                    ):
                        wait = 2 ** attempt
                        print(f"[GEMINI] Chat retry {attempt + 1}/3 after {wait}s -- {e}")
                        time.sleep(wait)
                    else:
                        raise
            elapsed = time.time() - t0
            print(f"[GEMINI] Chat response: {elapsed:.1f}s")
            return response.text

        except Exception as e:
            print(f"[GEMINI] Chat failed: {e}")
            return None
