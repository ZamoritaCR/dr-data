"""
run_openai_audit.py
Send all project .py files + requirements.txt to GPT-4o for cross-reference QA audit.
Usage:  python run_openai_audit.py
Requires: OPENAI_API_KEY environment variable
"""

import os
import sys
import glob
import time
import datetime

# ---------------------------------------------------------------------------
# 1. Check API key early
# ---------------------------------------------------------------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    print("Set it with:  set OPENAI_API_KEY=sk-...")
    sys.exit(1)

from openai import OpenAI

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------
# 2. Collect files
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

patterns = [
    os.path.join(ROOT, "app", "**", "*.py"),
    os.path.join(ROOT, "core", "**", "*.py"),
    os.path.join(ROOT, "generators", "**", "*.py"),
]

py_files = []
for pat in patterns:
    py_files.extend(glob.glob(pat, recursive=True))
py_files = sorted(set(py_files))

req_path = os.path.join(ROOT, "requirements.txt")
if os.path.isfile(req_path):
    py_files.append(req_path)

print(f"Collected {len(py_files)} files for audit.\n")

# ---------------------------------------------------------------------------
# 3. Read and truncate -- aggressive per-file limits
#    streamlit_app.py : first 5000 + last 5000
#    dr_data_agent.py : first 3000 + last 3000
#    everything else  : first 4000 + last 4000 (max 8000)
# ---------------------------------------------------------------------------
LIMITS = {
    "streamlit_app.py": (5000, 5000),
    "dr_data_agent.py": (3000, 3000),
}
DEFAULT_LIMIT = (4000, 4000)

file_contents = []
total_chars = 0

for fpath in py_files:
    rel = os.path.relpath(fpath, ROOT)
    basename = os.path.basename(fpath)
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print(f"  WARN: could not read {rel}: {e}")
        continue

    head, tail = LIMITS.get(basename, DEFAULT_LIMIT)
    max_chars = head + tail
    original_len = len(text)

    if original_len > max_chars:
        text = (
            text[:head]
            + f"\n\n... [TRUNCATED {original_len - head - tail:,} chars] ...\n\n"
            + text[-tail:]
        )
        print(f"  {rel}  ({original_len:,} chars -> truncated to ~{len(text):,})")
    else:
        print(f"  {rel}  ({original_len:,} chars)")

    file_contents.append(f"===== FILE: {rel} =====\n{text}\n")
    total_chars += len(file_contents[-1])

print(f"\nTotal payload: {total_chars:,} characters (~{total_chars // 4:,} tokens est.)\n")

# ---------------------------------------------------------------------------
# 4. Build the audit prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior Python/Streamlit code auditor. You will receive the source
code of a multi-module Streamlit application (app/, core/, generators/) plus its
requirements.txt. Some large files are truncated (middle section removed) but
the imports, class definitions, and key methods are preserved at the head and
tail of each file.

Perform a brutally thorough cross-reference QA audit covering:

1. IMPORT CHAINS -- For every `from X import Y` or `import X`, verify X exists
   in the provided files or is a known stdlib/PyPI package listed in
   requirements.txt. Flag missing or misspelled imports.

2. METHOD CROSS-REFERENCES -- For every `obj.method()` call in streamlit_app.py
   and dr_data_agent.py, verify the target class actually defines that method
   with the correct signature. Flag missing methods, wrong argument counts, or
   renamed methods.

3. SESSION STATE KEYS -- List every `st.session_state["key"]` or
   `st.session_state.key` usage. Flag keys that are READ but never WRITTEN
   (potential KeyError) or WRITTEN but never READ (dead state).

4. WIDGET KEY UNIQUENESS -- List every Streamlit widget call with a `key=`
   parameter. Flag any duplicate keys (including dynamic keys that could
   collide at runtime).

5. TAB / EXPANDER NESTING -- Identify any `st.tabs()` nested inside another
   `st.tabs()` or improper nesting of layout containers that Streamlit does
   not support.

6. ERROR HANDLING -- Flag bare `except:`, `except Exception: pass`, or
   overly broad catches that swallow real errors silently.

7. JSON PERSISTENCE -- Check every JSON file read/write for: missing
   file-not-found handling, encoding issues, race conditions on concurrent
   writes.

8. SECURITY -- Flag SQL injection (f-string in SQL), arbitrary code execution
   (eval/exec on user input), XSS in generated HTML, hardcoded credentials,
   path traversal in file operations.

9. DATA SAFETY -- Flag any place where a DataFrame is mutated in a context
   that should be read-only (e.g., building a context string). Flag
   destructive operations (drop, del, overwrite) on shared state.

10. DEPENDENCY GAPS -- Cross-reference every `import X` against
    requirements.txt. Flag packages that are imported but NOT listed in
    requirements.txt.

Output format:
- Start with an EXECUTIVE SUMMARY (counts by severity).
- Then list every finding as:
  [SEVERITY] file:line -- description
  where SEVERITY is CRITICAL, HIGH, MEDIUM, or LOW.
- Group findings by category (1-10 above).
- End with a PRIORITY FIX ORDER (top 10 most important fixes).

Be specific: include file names, line numbers (approximate is OK), method names,
and exact code snippets where relevant. Do NOT pad with generic advice.
"""

combined_source = "\n".join(file_contents)

USER_PROMPT = f"""\
Here are all source files for the Dr. Data application.
Perform the full 10-category cross-reference audit described in your instructions.

{combined_source}
"""

# ---------------------------------------------------------------------------
# 5. Send to GPT-4o
# ---------------------------------------------------------------------------
print("Sending to GPT-4o for audit (this may take 1-3 minutes)...")
t0 = time.time()

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=16000,
        temperature=0.1,
    )
except Exception as e:
    print(f"\nERROR: OpenAI API call failed: {e}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"Response received in {elapsed:.1f}s.\n")

# ---------------------------------------------------------------------------
# 6. Extract results and token usage
# ---------------------------------------------------------------------------
result_text = response.choices[0].message.content
usage = response.usage
prompt_tokens = usage.prompt_tokens if usage else 0
completion_tokens = usage.completion_tokens if usage else 0
total_tokens = usage.total_tokens if usage else 0

# GPT-4o pricing (as of early 2026): $2.50/1M input, $10.00/1M output
cost_input = (prompt_tokens / 1_000_000) * 2.50
cost_output = (completion_tokens / 1_000_000) * 10.00
cost_total = cost_input + cost_output

print(f"Token usage:  {prompt_tokens:,} input + {completion_tokens:,} output = {total_tokens:,} total")
print(f"Estimated cost: ${cost_total:.4f}")

# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------
OUTPUT_FILE = os.path.join(ROOT, "openai_qa_audit_results.txt")

header = f"""\
================================================================================
  GPT-4o CROSS-REFERENCE QA AUDIT
  Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  Model: gpt-4o
  Files analyzed: {len(py_files)}
  Total source chars: {total_chars:,}
  Tokens: {prompt_tokens:,} input + {completion_tokens:,} output = {total_tokens:,}
  Estimated cost: ${cost_total:.4f}
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
