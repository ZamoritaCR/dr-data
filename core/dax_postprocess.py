"""
Post-processing for DAX measures after translation.

Runs AFTER all translation phases (deterministic, heuristic, spatial, AI).
Never modifies the translation pipeline itself.

Two passes:
1. Syntax validation -- fix unbalanced parens, remove Tableau-only syntax
2. Column name reconciliation -- align DAX column references with actual data

Every function is wrapped in try/except and returns measures unchanged on error.
This module CANNOT crash the pipeline.
"""
import re


# ---------------------------------------------------------------------------
#  Pass 1: DAX Syntax Validation
# ---------------------------------------------------------------------------

def _balance_parens(dax):
    """Fix unbalanced parentheses in DAX expressions.

    Adds missing closing parens at the end, or removes excess closing parens.
    """
    depth = 0
    for ch in dax:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    if depth > 0:
        dax = dax + ")" * depth
    elif depth < 0:
        # Remove excess closing parens from the end
        excess = abs(depth)
        while excess > 0 and dax.endswith(")"):
            dax = dax[:-1]
            excess -= 1
    return dax


def _strip_tableau_syntax(dax):
    """Remove leftover Tableau syntax that leaked through heuristic translation.

    Handles: IF...THEN...ELSE...END -> IF(..., ..., ...)
    Only rewrites the outermost IF block if the DAX still contains THEN/END.
    """
    # Skip if it already looks like DAX IF(cond, then, else)
    if "THEN" not in dax and "END" not in dax.split("(")[0]:
        return dax

    # Simple single-level IF THEN ELSE END
    m = re.match(
        r'^IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END\s*$',
        dax.strip(), re.IGNORECASE | re.DOTALL,
    )
    if m:
        cond = m.group(1).strip()
        then_val = m.group(2).strip()
        else_val = m.group(3).strip()
        return f"IF({cond}, {then_val}, {else_val})"

    # IF THEN END (no ELSE)
    m2 = re.match(
        r'^IF\s+(.+?)\s+THEN\s+(.+?)\s+END\s*$',
        dax.strip(), re.IGNORECASE | re.DOTALL,
    )
    if m2:
        cond = m2.group(1).strip()
        then_val = m2.group(2).strip()
        return f"IF({cond}, {then_val}, BLANK())"

    # Nested ELSEIF -- convert to nested IF
    # IF cond1 THEN val1 ELSEIF cond2 THEN val2 ELSE val3 END
    m3 = re.match(
        r'^IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSEIF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END\s*$',
        dax.strip(), re.IGNORECASE | re.DOTALL,
    )
    if m3:
        c1 = m3.group(1).strip()
        v1 = m3.group(2).strip()
        c2 = m3.group(3).strip()
        v2 = m3.group(4).strip()
        v3 = m3.group(5).strip()
        return f"IF({c1}, {v1}, IF({c2}, {v2}, {v3}))"

    return dax


def _validate_dax_syntax(dax):
    """Run all syntax fixes on a single DAX expression.

    Returns the fixed DAX string.
    """
    if not dax or "BLANK()" in dax or dax.startswith("/*"):
        return dax

    dax = _strip_tableau_syntax(dax)
    dax = _balance_parens(dax)
    return dax


# ---------------------------------------------------------------------------
#  Pass 2: Column Name Reconciliation
# ---------------------------------------------------------------------------

def _build_column_lookup(profile_col_names):
    """Build a case-insensitive lookup for fuzzy column matching.

    Returns dict: lowercase_name -> actual_name
    Also includes underscore/space/hyphen variants.
    """
    lookup = {}
    for col in profile_col_names:
        lookup[col.lower()] = col
        # Also register without spaces/underscores/hyphens
        normalized = col.lower().replace(" ", "").replace("_", "").replace("-", "")
        lookup[normalized] = col
        # Register with spaces instead of underscores
        lookup[col.lower().replace("_", " ")] = col
        # Register with underscores instead of spaces
        lookup[col.lower().replace(" ", "_")] = col
    return lookup


def _reconcile_column_refs(dax, table_name, col_lookup):
    """Fix column references in DAX to match actual data column names.

    Finds all 'Table'[Column] or Table[Column] patterns and checks if
    the column name matches. If not, tries fuzzy matching.

    Returns the fixed DAX string and count of fixes applied.
    """
    if not dax or dax.startswith("/*"):
        return dax, 0

    fixes = 0

    def fix_ref(match):
        nonlocal fixes
        prefix = match.group(1)  # 'Table' or Table
        col = match.group(2)     # column name

        # Already valid?
        if col in col_lookup.values():
            return match.group(0)

        # Try case-insensitive match
        col_lower = col.lower()
        if col_lower in col_lookup:
            actual = col_lookup[col_lower]
            if actual != col:
                fixes += 1
            return f"{prefix}[{actual}]"

        # Try normalized (no spaces/underscores)
        normalized = col_lower.replace(" ", "").replace("_", "").replace("-", "")
        if normalized in col_lookup:
            actual = col_lookup[normalized]
            fixes += 1
            return f"{prefix}[{actual}]"

        # No match found -- leave as-is
        return match.group(0)

    # Match 'TableName'[Column] or TableName[Column]
    result = re.sub(
        r"('[\w\s]+'|\w+)(\[[^\]]+\])",
        lambda m: fix_ref(m) if m.group(2) else m.group(0),
        dax,
    )

    # More precise: capture prefix and column separately
    result = re.sub(
        r"('[\w\s]+'|[\w]+)\[([^\]]+)\]",
        fix_ref,
        dax,
    )

    return result, fixes


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def postprocess_measures(measures, table_name, profile_col_names):
    """Run post-processing on translated measures.

    This is the only function called from the pipeline.
    It NEVER crashes -- all errors are caught and measures pass through unchanged.

    Args:
        measures: list of measure dicts (mutated in place)
        table_name: PBI table name
        profile_col_names: set of actual column names from the data

    Returns:
        dict with stats: {syntax_fixed, columns_fixed, total_processed}
    """
    stats = {"syntax_fixed": 0, "columns_fixed": 0, "total_processed": 0}

    try:
        col_lookup = _build_column_lookup(profile_col_names)
    except Exception:
        col_lookup = {}

    for m in measures:
        dax = m.get("dax", "")
        if not dax or dax.strip() == "BLANK()" or dax.startswith("/*"):
            continue

        stats["total_processed"] += 1
        original = dax

        # Pass 1: syntax validation
        try:
            dax = _validate_dax_syntax(dax)
            if dax != original:
                stats["syntax_fixed"] += 1
        except Exception:
            dax = original  # revert on any error

        # Pass 2: column name reconciliation
        try:
            if col_lookup:
                dax, col_fixes = _reconcile_column_refs(dax, table_name, col_lookup)
                stats["columns_fixed"] += col_fixes
        except Exception:
            pass  # leave dax as-is from pass 1

        m["dax"] = dax

    return stats
