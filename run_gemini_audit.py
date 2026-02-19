"""
run_gemini_audit.py
Send the entire codebase to Gemini 2.0 Flash for an independent QA audit.
Usage:  python run_gemini_audit.py
Requires: GEMINI_API_KEY env var or .streamlit/secrets.toml entry
"""

import os
import sys
import glob
import time
import datetime

# ---------------------------------------------------------------------------
# 1. Resolve API key: env var first, then .streamlit/secrets.toml
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    secrets_path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if os.path.isfile(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY"):
                        _, _, val = line.partition("=")
                        api_key = val.strip().strip('"').strip("'")
                        break
        except Exception as e:
            print(f"WARN: Could not parse {secrets_path}: {e}")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found.")
    print("Set it with:  set GEMINI_API_KEY=AI...")
    print("Or add GEMINI_API_KEY = \"...\" to .streamlit/secrets.toml")
    sys.exit(1)

print("API key found.\n")

import google.generativeai as genai

genai.configure(api_key=api_key)

# ---------------------------------------------------------------------------
# 2. Collect files -- no truncation (Gemini handles 1M tokens)
# ---------------------------------------------------------------------------
patterns = [
    os.path.join(ROOT, "app", "**", "*.py"),
    os.path.join(ROOT, "core", "**", "*.py"),
    os.path.join(ROOT, "generators", "**", "*.py"),
]

py_files = []
for pat in patterns:
    py_files.extend(glob.glob(pat, recursive=True))
py_files = sorted(set(py_files))

# Add requirements.txt and config.toml
for extra in ["requirements.txt", os.path.join(".streamlit", "config.toml")]:
    extra_path = os.path.join(ROOT, extra)
    if os.path.isfile(extra_path):
        py_files.append(extra_path)

print(f"Collected {len(py_files)} files for audit.\n")

# ---------------------------------------------------------------------------
# 3. Read all file contents (no truncation)
# ---------------------------------------------------------------------------
file_contents = []
total_chars = 0

for fpath in py_files:
    rel = os.path.relpath(fpath, ROOT)
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print(f"  WARN: could not read {rel}: {e}")
        continue

    print(f"  {rel}  ({len(text):,} chars)")
    file_contents.append(f"===== FILE: {rel} =====\n{text}\n")
    total_chars += len(file_contents[-1])

print(f"\nTotal payload: {total_chars:,} characters (~{total_chars // 4:,} tokens est.)\n")

# ---------------------------------------------------------------------------
# 4. Build the audit prompt
# ---------------------------------------------------------------------------
AUDIT_PROMPT = """\
You are a senior software architect performing an independent cross-reference QA audit of a Streamlit-based enterprise data intelligence platform called Dr. Data. This audit runs in parallel with Claude and OpenAI audits. Your job is to catch issues others might miss.

The platform has:
- Main UI in app/streamlit_app.py (Streamlit)
- AI agent in app/dr_data_agent.py
- 12+ data quality modules in core/ directory
- Tab 1: AI workspace + chat
- Tab 2: Data Quality Engine with 8 sub-tabs

CHECK EVERY ITEM:
1. IMPORT CHAIN: Every from core.XXX import YYY - does the class and method exist?
2. METHOD CALLS: Every module.method() call - does the method exist with correct signature?
3. SESSION STATE: Every st.session_state key - initialized before access?
4. WIDGET KEYS: All Streamlit widget key= params - are all unique?
5. TAB STRUCTURE: tab1/tab2 and dq_subtab1-8 properly nested?
6. ERROR HANDLING: try/except coverage, bare excepts?
7. JSON PERSISTENCE: Corruption handling, unbounded growth?
8. SECURITY: Hardcoded creds, eval() injection, SQL injection?
9. DATA SAFETY: None/NaN/empty DataFrame handling, division by zero?
10. DEPENDENCIES: All imports covered in requirements.txt?
11. LOGIC BUGS: Wrong dictionary keys, wrong method names, type mismatches?
12. PERFORMANCE: O(n^2) algorithms, memory issues with large DataFrames?

OUTPUT FORMAT:
A. CRITICAL BUGS (will crash the app)
B. HIGH-RISK ISSUES (incorrect behavior)
C. MEDIUM ISSUES (conditional failures)
D. LOW ISSUES (code quality)
E. CROSS-REFERENCE MATRIX (module -> method -> exists YES/NO)
F. SECURITY VULNERABILITIES
G. PRIORITY FIX LIST (top 20)

Be extremely specific. File names, line numbers where possible, method names, exact issues.

Here are ALL source files:

""" + "\n".join(file_contents)

# ---------------------------------------------------------------------------
# 5. Send to Gemini 2.0 Flash
# ---------------------------------------------------------------------------
print("Sending to Gemini 2.0 Flash for audit (this may take 1-3 minutes)...")
t0 = time.time()

try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        AUDIT_PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=16000,
        ),
    )
except Exception as e:
    print(f"\nERROR: Gemini API call failed: {e}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"Response received in {elapsed:.1f}s.\n")

# ---------------------------------------------------------------------------
# 6. Extract results
# ---------------------------------------------------------------------------
try:
    result_text = response.text
except Exception as e:
    print(f"ERROR: Could not extract response text: {e}")
    if hasattr(response, "prompt_feedback"):
        print(f"Prompt feedback: {response.prompt_feedback}")
    sys.exit(1)

# Token usage (if available)
prompt_tokens = 0
completion_tokens = 0
total_tokens = 0
if hasattr(response, "usage_metadata") and response.usage_metadata:
    um = response.usage_metadata
    prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
    completion_tokens = getattr(um, "candidates_token_count", 0) or 0
    total_tokens = getattr(um, "total_token_count", 0) or 0

print(f"Token usage:  {prompt_tokens:,} input + {completion_tokens:,} output = {total_tokens:,} total")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
OUTPUT_FILE = os.path.join(ROOT, "gemini_qa_audit_results.txt")

header = f"""\
================================================================================
  GEMINI 2.0 FLASH -- CROSS-REFERENCE QA AUDIT
  Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  Model: gemini-2.0-flash
  Files analyzed: {len(py_files)}
  Total source chars: {total_chars:,}
  Tokens: {prompt_tokens:,} input + {completion_tokens:,} output = {total_tokens:,}
  Elapsed: {elapsed:.1f}s
================================================================================

Files included:
{chr(10).join("  " + os.path.relpath(f, ROOT) for f in py_files)}

================================================================================
AUDIT RESULTS
================================================================================

"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(header + result_text + "\n")

print(f"\nResults saved to: {OUTPUT_FILE}")
print("Done.")
