from core.utils import remove_tableau_brackets
"""
Sprint 3 -- Tableau Formula to DAX Transpiler.

AST-based formula translation using Lark parser.
Handles LOD expressions, IIF, ZN, date functions, string functions,
and generates DAX with confidence scoring.
"""
from lark import Lark, Transformer, Token, v_args, exceptions as lark_exceptions

TABLEAU_GRAMMAR = r"""
    ?start: expr

    ?expr: if_expr
        | iif_expr
        | case_expr
        | lod_expr
        | logical_expr

    ?logical_expr: comparison
        | logical_expr AND comparison   -> and_expr
        | logical_expr OR comparison    -> or_expr
        | NOT comparison                -> not_expr

    AND: /AND/i
    OR: /OR/i
    NOT: /NOT/i

    ?comparison: arithmetic
        | comparison "=" arithmetic     -> eq
        | comparison "!=" arithmetic    -> neq
        | comparison "<>" arithmetic    -> neq
        | comparison "<" arithmetic     -> lt
        | comparison ">" arithmetic     -> gt
        | comparison "<=" arithmetic    -> lte
        | comparison ">=" arithmetic    -> gte

    ?arithmetic: term
        | arithmetic "+" term           -> add
        | arithmetic "-" term           -> sub

    ?term: factor
        | term "*" factor               -> mul
        | term "/" factor               -> div

    ?factor: "-" factor                 -> neg
        | "+" factor
        | atom

    ?atom: "(" expr ")"
        | function_call
        | field_ref
        | number
        | string
        | boolean

    if_expr: IF expr THEN expr (ELSEIF expr THEN expr)* (ELSE expr)? END
    iif_expr: IIF "(" expr "," expr "," expr ("," expr)? ")"
    case_expr: CASE expr (WHEN expr THEN expr)+ (ELSE expr)? END

    IF: /IF/i
    THEN: /THEN/i
    ELSEIF: /ELSEIF/i
    ELSE: /ELSE/i
    END: /END/i
    IIF: /IIF/i
    CASE: /CASE/i
    WHEN: /WHEN/i

    lod_expr: "{" lod_type field_ref_list ":" expr "}"
    lod_type: /FIXED/i | /INCLUDE/i | /EXCLUDE/i
    field_ref_list: field_ref ("," field_ref)*

    function_call: FUNC_NAME "(" [expr ("," expr)*] ")"
    FUNC_NAME: /[A-Za-z_][A-Za-z0-9_]*/

    field_ref: "[" FIELD_CONTENT "]" "." "[" FIELD_CONTENT "]"  -> qualified_field_ref
        | "[" FIELD_CONTENT "]"                                  -> simple_field_ref
    FIELD_CONTENT: /[^\[\]]+/

    number: SIGNED_NUMBER
    string: ESCAPED_STRING
    boolean: TRUE | FALSE | NULL
    TRUE: /TRUE/i
    FALSE: /FALSE/i
    NULL: /NULL/i

    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
    %ignore /\/\/.*/
"""

# Tableau -> DAX function mapping
FUNCTION_MAP = {
    # Aggregate
    "SUM": "SUM",
    "AVG": "AVERAGE",
    "AVERAGE": "AVERAGE",
    "MIN": "MIN",
    "MAX": "MAX",
    "COUNT": "COUNT",
    "COUNTD": "DISTINCTCOUNT",
    "MEDIAN": "MEDIAN",
    "STDEV": "STDEV.S",
    "STDEVP": "STDEV.P",
    "VAR": "VAR.S",
    "VARP": "VAR.P",
    "ATTR": None,

    # Logical
    "IIF": "IF",
    "ZN": None,
    "ISNULL": "ISBLANK",
    "IFNULL": None,
    "ISDATE": None,

    # String
    "STR": "FORMAT",
    "INT": "INT",
    "FLOAT": "CONVERT",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "MID": "MID",
    "LEN": "LEN",
    "UPPER": "UPPER",
    "LOWER": "LOWER",
    "TRIM": "TRIM",
    "LTRIM": None,
    "RTRIM": None,
    "REPLACE": "SUBSTITUTE",
    "CONTAINS": None,
    "STARTSWITH": None,
    "ENDSWITH": None,
    "FIND": "FIND",
    "REGEXP_MATCH": None,
    "SPACE": "REPT",
    "SPLIT": None,
    "ASCII": "UNICODE",
    "CHAR": "UNICHAR",

    # Date
    "YEAR": "YEAR",
    "MONTH": "MONTH",
    "DAY": "DAY",
    "HOUR": "HOUR",
    "MINUTE": "MINUTE",
    "SECOND": "SECOND",
    "TODAY": "TODAY",
    "NOW": "NOW",
    "DATEADD": "DATEADD",
    "DATEDIFF": "DATEDIFF",
    "DATEPART": None,
    "DATETRUNC": None,
    "DATE": "DATE",
    "MAKEDATE": "DATE",
    "MAKEDATETIME": None,
    "DATENAME": None,

    # Math
    "ABS": "ABS",
    "CEILING": "CEILING",
    "FLOOR": "FLOOR",
    "ROUND": "ROUND",
    "SQRT": "SQRT",
    "POWER": "POWER",
    "EXP": "EXP",
    "LN": "LN",
    "LOG": "LOG",
    "LOG2": None,
    "PI": "PI",
    "SIGN": "SIGN",
    "MOD": "MOD",
    "DIV": None,

    # Table calculations (mostly untranslatable)
    "WINDOW_SUM": None,
    "WINDOW_AVG": None,
    "RUNNING_SUM": None,
    "RUNNING_AVG": None,
    "RUNNING_COUNT": None,
    "RUNNING_MIN": None,
    "RUNNING_MAX": None,
    "LOOKUP": None,
    "FIRST": None,
    "LAST": None,
    "INDEX": None,
    "RANK": "RANKX",
    "RANK_DENSE": "RANKX",
    "SIZE": None,
    "TOTAL": None,
    "PREVIOUS_VALUE": None,
}

# Table calc functions that cannot be auto-translated
TABLE_CALC_FUNCTIONS = {
    "WINDOW_SUM", "WINDOW_AVG", "RUNNING_SUM", "RUNNING_AVG",
    "RUNNING_COUNT", "RUNNING_MIN", "RUNNING_MAX", "LOOKUP",
    "FIRST", "LAST", "INDEX", "SIZE", "TOTAL", "PREVIOUS_VALUE",
}


@v_args(inline=True)
class DAXEmitter(Transformer):
    """Transform a Lark parse tree into a DAX expression string."""

    def __init__(self, field_resolution_map=None, table_name="Data"):
        super().__init__()
        self.table_name = table_name
        self.field_map = field_resolution_map or {}
        self.warnings = []
        self.confidence = 1.0

    def _resolve_field(self, field_name):
        """Resolve a Tableau field name to a DAX field reference."""
        clean = remove_tableau_brackets(field_name)
        # Check resolution map
        if self.field_map:
            entry = self.field_map.get(field_name) or self.field_map.get(clean)
            if entry and entry.get("excel_column"):
                return f"{self.table_name}[{entry['excel_column']}]"
        return f"{self.table_name}[{clean}]"

    def simple_field_ref(self, content):
        return self._resolve_field(str(content))

    def qualified_field_ref(self, table, field):
        return self._resolve_field(str(field))

    def number(self, n):
        return str(n)

    def string(self, s):
        # Lark ESCAPED_STRING includes the quotes
        val = str(s)
        # DAX uses double quotes for strings
        if val.startswith("'") and val.endswith("'"):
            inner = val[1:-1]
            return f'"{inner}"'
        return val

    def boolean(self, b):
        val = str(b).upper()
        if val == "NULL":
            return "BLANK()"
        return val

    # Arithmetic
    def add(self, a, b):
        return f"({a} + {b})"

    def sub(self, a, b):
        return f"({a} - {b})"

    def mul(self, a, b):
        return f"({a} * {b})"

    def div(self, a, b):
        return f"DIVIDE({a}, {b})"

    def neg(self, a):
        return f"(-{a})"

    # Comparison
    def eq(self, a, b):
        return f"({a} = {b})"

    def neq(self, a, b):
        return f"({a} <> {b})"

    def lt(self, a, b):
        return f"({a} < {b})"

    def gt(self, a, b):
        return f"({a} > {b})"

    def lte(self, a, b):
        return f"({a} <= {b})"

    def gte(self, a, b):
        return f"({a} >= {b})"

    # Logical
    def and_expr(self, a, _kw, b):
        return f"({a} && {b})"

    def or_expr(self, a, _kw, b):
        return f"({a} || {b})"

    def not_expr(self, _kw, a):
        return f"NOT({a})"

    # IF / THEN / ELSE / END
    def if_expr(self, *args):
        # Filter out keyword tokens
        parts = [a for a in args if not isinstance(a, Token)]
        if len(parts) < 2:
            return "IF(TRUE(), BLANK())"

        # Simple IF: condition, then_val [, else_val]
        if len(parts) == 2:
            return f"IF({parts[0]}, {parts[1]})"
        elif len(parts) == 3:
            return f"IF({parts[0]}, {parts[1]}, {parts[2]})"
        else:
            # ELSEIF chains -> nested IF
            result = parts[-1] if len(parts) % 2 == 1 else "BLANK()"
            pairs = []
            i = 0
            while i + 1 < len(parts):
                if i + 2 < len(parts) and len(parts) % 2 == 0:
                    pairs.append((parts[i], parts[i + 1]))
                    i += 2
                elif i + 1 < len(parts):
                    pairs.append((parts[i], parts[i + 1]))
                    i += 2
                else:
                    break

            # Build nested IF from pairs
            if len(parts) % 2 == 1:
                result = parts[-1]
                pair_list = [(parts[i], parts[i + 1]) for i in range(0, len(parts) - 1, 2)]
            else:
                result = "BLANK()"
                pair_list = [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]

            for cond, val in reversed(pair_list):
                result = f"IF({cond}, {val}, {result})"
            return result

    def iif_expr(self, *args):
        parts = [a for a in args if not isinstance(a, Token)]
        if len(parts) >= 3:
            return f"IF({parts[0]}, {parts[1]}, {parts[2]})"
        return f"IF({', '.join(parts)})"

    def case_expr(self, *args):
        parts = [a for a in args if not isinstance(a, Token)]
        if len(parts) < 3:
            return f"SWITCH({', '.join(parts)})"

        switch_expr = parts[0]
        has_else = len(parts) % 2 == 0
        cases = []
        i = 1
        end = len(parts) - 1 if has_else else len(parts)
        while i + 1 <= end:
            cases.append(f"{parts[i]}, {parts[i + 1]}")
            i += 2
        else_val = parts[-1] if has_else else "BLANK()"
        return f"SWITCH({switch_expr}, {', '.join(cases)}, {else_val})"

    def lod_expr(self, *args):
        parts = [a for a in args if not isinstance(a, Token)]
        if len(parts) < 2:
            self.warnings.append("LOD expression could not be fully parsed")
            self.confidence *= 0.7
            return f"/* LOD */ {' '.join(str(p) for p in parts)}"

        lod_type = str(parts[0]).upper().strip()
        agg_expr = str(parts[-1])

        # Flatten dimension fields: parts[1:-1] may contain lists from field_ref_list
        dim_fields = []
        for p in parts[1:-1]:
            if isinstance(p, list):
                dim_fields.extend(str(x) for x in p)
            else:
                dim_fields.append(str(p))

        if lod_type == "FIXED":
            dim_refs = ", ".join(dim_fields)
            return f"CALCULATE({agg_expr}, ALLEXCEPT({self.table_name}, {dim_refs}))"
        elif lod_type == "EXCLUDE":
            dim_refs = ", ".join(f"ALL({d})" for d in dim_fields)
            return f"CALCULATE({agg_expr}, {dim_refs})"
        elif lod_type == "INCLUDE":
            dim_refs = ", ".join(f"KEEPFILTERS(VALUES({d}))" for d in dim_fields)
            return f"CALCULATE({agg_expr}, {dim_refs})"
        else:
            self.warnings.append(f"Unknown LOD type: {lod_type}")
            self.confidence *= 0.7
            return f"/* {lod_type} LOD */ CALCULATE({agg_expr})"

    def lod_type(self, t):
        return str(t).upper()

    def field_ref_list(self, *refs):
        return list(str(r) for r in refs)

    def function_call(self, func_name, *args):
        # Filter out tokens
        fn = str(func_name).upper()
        call_args = [a for a in args if a is not None and not isinstance(a, Token)]

        # Special function handling
        if fn == "ZN":
            if call_args:
                arg = call_args[0]
                return f"IF(ISBLANK({arg}), 0, {arg})"
            return "0"

        if fn == "IFNULL":
            if len(call_args) >= 2:
                return f"IF(ISBLANK({call_args[0]}), {call_args[1]}, {call_args[0]})"
            return str(call_args[0]) if call_args else "BLANK()"

        if fn == "ISNULL":
            if call_args:
                return f"ISBLANK({call_args[0]})"
            return "ISBLANK(BLANK())"

        if fn == "CONTAINS":
            if len(call_args) >= 2:
                return f"CONTAINSSTRING({call_args[0]}, {call_args[1]})"
            return f"CONTAINSSTRING({', '.join(str(a) for a in call_args)})"

        if fn == "STARTSWITH":
            if len(call_args) >= 2:
                return f"(LEFT({call_args[0]}, LEN({call_args[1]})) = {call_args[1]})"
            return f"/* STARTSWITH */ FALSE"

        if fn == "ENDSWITH":
            if len(call_args) >= 2:
                return f"(RIGHT({call_args[0]}, LEN({call_args[1]})) = {call_args[1]})"
            return f"/* ENDSWITH */ FALSE"

        if fn == "DATEPART":
            if len(call_args) >= 2:
                part = str(call_args[0]).strip('"\'').lower()
                field = call_args[1]
                part_map = {
                    "year": "YEAR", "month": "MONTH", "day": "DAY",
                    "hour": "HOUR", "minute": "MINUTE", "second": "SECOND",
                    "quarter": "QUARTER", "week": "WEEKNUM",
                    "weekday": "WEEKDAY",
                }
                dax_fn = part_map.get(part, "YEAR")
                return f"{dax_fn}({field})"
            self.warnings.append("DATEPART requires at least 2 arguments")
            return "/* DATEPART */ BLANK()"

        if fn == "DATETRUNC":
            if len(call_args) >= 2:
                part = str(call_args[0]).strip('"\'').lower()
                field = call_args[1]
                if part in ("month", "mon"):
                    self.warnings.append("DATETRUNC(month) approximated with EOMONTH pattern")
                    return f"DATE(YEAR({field}), MONTH({field}), 1)"
                elif part in ("year", "yr"):
                    return f"DATE(YEAR({field}), 1, 1)"
                elif part in ("quarter", "qtr"):
                    return f"DATE(YEAR({field}), (QUARTER({field}) - 1) * 3 + 1, 1)"
                elif part in ("week", "wk"):
                    self.warnings.append("DATETRUNC(week) approximated")
                    return f"({field} - WEEKDAY({field}, 2) + 1)"
                else:
                    return f"/* DATETRUNC({part}) */ {field}"
            return f"/* DATETRUNC */ BLANK()"

        if fn == "ATTR":
            if call_args:
                return f"FIRSTNONBLANK({call_args[0]}, 1)"
            return "BLANK()"

        if fn == "STR":
            if call_args:
                return f'FORMAT({call_args[0]}, "")'
            return '""'

        if fn == "MAKEDATETIME":
            if len(call_args) >= 2:
                return f"({call_args[0]} + {call_args[1]})"
            return str(call_args[0]) if call_args else "BLANK()"

        if fn == "DATENAME":
            if len(call_args) >= 2:
                part = str(call_args[0]).strip('"\'').lower()
                field = call_args[1]
                if part == "month":
                    return f'FORMAT({field}, "MMMM")'
                elif part == "weekday":
                    return f'FORMAT({field}, "dddd")'
                return f'FORMAT({field}, "")'
            return '""'

        if fn in ("LTRIM", "RTRIM"):
            if call_args:
                return f"TRIM({call_args[0]})"
            return 'TRIM("")'

        if fn == "REPLACE":
            if len(call_args) >= 3:
                return f"SUBSTITUTE({call_args[0]}, {call_args[1]}, {call_args[2]})"
            return f"SUBSTITUTE({', '.join(str(a) for a in call_args)})"

        if fn == "SPACE":
            if call_args:
                return f'REPT(" ", {call_args[0]})'
            return '" "'

        if fn == "DIV":
            if len(call_args) >= 2:
                return f"INT(DIVIDE({call_args[0]}, {call_args[1]}))"
            return "0"

        if fn == "LOG2":
            if call_args:
                return f"LOG({call_args[0]}, 2)"
            return "0"

        # Table calc warning
        if fn in TABLE_CALC_FUNCTIONS:
            self.warnings.append(
                f"Table calculation '{fn}' cannot be auto-translated to DAX. "
                f"Requires manual conversion using RANKX, CALCULATE, or custom measures."
            )
            self.confidence *= 0.3
            args_str = ", ".join(str(a) for a in call_args)
            return f"/* {fn}({args_str}) -- MANUAL REVIEW REQUIRED */"

        if fn == "REGEXP_MATCH":
            self.warnings.append("REGEXP_MATCH has no DAX equivalent")
            self.confidence *= 0.5
            args_str = ", ".join(str(a) for a in call_args)
            return f"/* REGEXP_MATCH({args_str}) -- no DAX equivalent */"

        # Standard function mapping
        dax_fn = FUNCTION_MAP.get(fn, fn)
        if dax_fn is None:
            self.warnings.append(f"Function '{fn}' has no direct DAX equivalent")
            self.confidence *= 0.6
            args_str = ", ".join(str(a) for a in call_args)
            return f"/* {fn} */ {fn}({args_str})"

        args_str = ", ".join(str(a) for a in call_args)
        return f"{dax_fn}({args_str})"


class TableauFormulaTranspiler:
    """Public API for transpiling Tableau formulas to DAX."""

    def __init__(self, field_resolution_map=None, table_name="Data"):
        self.field_resolution_map = field_resolution_map
        self.table_name = table_name
        self._parser = Lark(TABLEAU_GRAMMAR, start="start", parser="earley",
                           ambiguity="resolve")

    def transpile(self, tableau_formula):
        """Transpile a single Tableau formula to DAX.

        Returns:
            dict with keys: dax, confidence, warnings, original, parse_success
        """
        result = {
            "dax": "",
            "confidence": 0.0,
            "warnings": [],
            "original": tableau_formula,
            "parse_success": False,
        }

        if not tableau_formula or not tableau_formula.strip():
            result["warnings"].append("Empty formula")
            return result

        formula = tableau_formula.strip()

        try:
            tree = self._parser.parse(formula)
            emitter = DAXEmitter(
                field_resolution_map=self.field_resolution_map,
                table_name=self.table_name,
            )
            dax = emitter.transform(tree)
            result["dax"] = str(dax)
            result["confidence"] = emitter.confidence
            result["warnings"] = emitter.warnings
            result["parse_success"] = True
        except lark_exceptions.UnexpectedInput as e:
            result["warnings"].append(f"Parse error at position {e.pos_in_stream}: {str(e)[:100]}")
            # Attempt regex-based fallback
            fallback = self._regex_fallback(formula)
            if fallback:
                result["dax"] = fallback["dax"]
                result["confidence"] = fallback["confidence"]
                result["warnings"].extend(fallback["warnings"])
                result["parse_success"] = True
        except Exception as e:
            result["warnings"].append(f"Transpilation error: {str(e)[:200]}")
            fallback = self._regex_fallback(formula)
            if fallback:
                result["dax"] = fallback["dax"]
                result["confidence"] = fallback["confidence"]
                result["warnings"].extend(fallback["warnings"])

        return result

    def transpile_batch(self, formulas):
        """Transpile a dict of {field_name: tableau_formula}.

        Returns:
            dict: {field_name: transpile_result}
        """
        results = {}
        for name, formula in formulas.items():
            results[name] = self.transpile(formula)
        return results

    def _regex_fallback(self, formula):
        """Regex-based fallback transpilation for formulas that fail parsing."""
        import re
        dax = formula
        warnings = ["Used regex fallback -- review output carefully"]
        confidence = 0.5

        # ZN pattern
        dax = re.sub(
            r'ZN\s*\(([^)]+)\)',
            lambda m: f'IF(ISBLANK({m.group(1)}), 0, {m.group(1)})',
            dax, flags=re.IGNORECASE
        )

        # IIF pattern
        dax = re.sub(
            r'IIF\s*\(',
            'IF(',
            dax, flags=re.IGNORECASE
        )

        # ISNULL -> ISBLANK
        dax = re.sub(r'\bISNULL\b', 'ISBLANK', dax, flags=re.IGNORECASE)

        # COUNTD -> DISTINCTCOUNT
        dax = re.sub(r'\bCOUNTD\b', 'DISTINCTCOUNT', dax, flags=re.IGNORECASE)

        # AVG -> AVERAGE
        dax = re.sub(r'\bAVG\b', 'AVERAGE', dax, flags=re.IGNORECASE)

        # IF/THEN/ELSE/END -> IF()
        if_match = re.match(
            r'IF\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+?))?\s+END',
            dax, re.IGNORECASE | re.DOTALL
        )
        if if_match:
            cond = if_match.group(1)
            then_val = if_match.group(2)
            else_val = if_match.group(3) or "BLANK()"
            dax = f"IF({cond}, {then_val}, {else_val})"

        # LOD FIXED
        lod_match = re.search(
            r'\{FIXED\s+(.+?)\s*:\s*(.+?)\}',
            dax, re.IGNORECASE
        )
        if lod_match:
            dims_raw = lod_match.group(1)
            agg = lod_match.group(2)
            dims = [remove_tableau_brackets(d.strip()) for d in dims_raw.split(",")]
            dim_refs = ", ".join(f"{self.table_name}[{d}]" for d in dims)
            replacement = f"CALCULATE({agg}, ALLEXCEPT({self.table_name}, {dim_refs}))"
            dax = dax[:lod_match.start()] + replacement + dax[lod_match.end():]

        # LOD EXCLUDE
        lod_excl = re.search(
            r'\{EXCLUDE\s+(.+?)\s*:\s*(.+?)\}',
            dax, re.IGNORECASE
        )
        if lod_excl:
            dims_raw = lod_excl.group(1)
            agg = lod_excl.group(2)
            dims = [remove_tableau_brackets(d.strip()) for d in dims_raw.split(",")]
            dim_refs = ", ".join(f"ALL({self.table_name}[{d}])" for d in dims)
            replacement = f"CALCULATE({agg}, {dim_refs})"
            dax = dax[:lod_excl.start()] + replacement + dax[lod_excl.end():]

        # Resolve field refs: [Field] -> Table[Field]
        def resolve_ref(m):
            field = m.group(1)
            return f"{self.table_name}[{field}]"

        dax = re.sub(r'(?<!\w)\[([^\]]+)\]', resolve_ref, dax)

        if dax == formula:
            return None  # No transformation happened

        return {
            "dax": dax,
            "confidence": confidence,
            "warnings": warnings,
        }
