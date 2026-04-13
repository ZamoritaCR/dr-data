"""
Color Extractor -- Deep color extraction from Tableau workbook XML.

Goes beyond the basic parser palette extraction to capture every color
signal in the workbook: gradients, conditional formats, reference lines,
annotations, dashboard backgrounds, borders, tooltips, and title colors.

Produces a unified palette ranked by frequency with harmony analysis.

Pure functions. No AI calls. No file I/O. No side effects.
"""

import re
from collections import Counter

# Regex: matches 3 or 6 hex digit color with optional # prefix
_HEX_RE = re.compile(r'^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$')


def _normalize_hex(color_str):
    """Normalize any hex color string to lowercase 6-char without #.

    Handles: #abc, #aabbcc, abc, aabbcc, empty/invalid -> "".
    """
    if not color_str or not isinstance(color_str, str):
        return ""
    c = color_str.strip().lstrip("#")
    if len(c) == 3 and all(ch in "0123456789abcdefABCDEF" for ch in c):
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    if len(c) != 6:
        return ""
    if not all(ch in "0123456789abcdefABCDEF" for ch in c):
        return ""
    return c.lower()


def _is_noise_color(hex6):
    """Return True if color is near-white or near-black (visual noise)."""
    if not hex6 or len(hex6) != 6:
        return True
    try:
        r = int(hex6[0:2], 16)
        g = int(hex6[2:4], 16)
        b = int(hex6[4:6], 16)
    except ValueError:
        return True
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    # Near-white (>245) or near-black (<10)
    return brightness > 245 or brightness < 10


def _hex_to_hsl(hex6):
    """Convert 6-char hex to HSL tuple (h: 0-360, s: 0-1, l: 0-1)."""
    r = int(hex6[0:2], 16) / 255.0
    g = int(hex6[2:4], 16) / 255.0
    b = int(hex6[4:6], 16) / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    lightness = (cmax + cmin) / 2.0
    if delta == 0:
        hue = 0.0
        saturation = 0.0
    else:
        saturation = delta / (1 - abs(2 * lightness - 1)) if (1 - abs(2 * lightness - 1)) != 0 else 0
        if cmax == r:
            hue = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)
    return (hue, saturation, lightness)


def extract_deep_palette(root, max_colors=12):
    """Extract the complete visual palette from a Tableau workbook.

    Goes beyond extract_all_colors by scanning ALL format attributes and
    text content for hex colors, filtering noise, and building a ranked
    palette suitable for PBI theme injection.

    Args:
        root: XML Element (TWB root or workbook element).
        max_colors: maximum palette size.

    Returns:
        list of hex color strings with # prefix, ranked by visual importance.
    """
    import xml.etree.ElementTree as ET

    # Phase 1: Collect every hex color from every attribute and text node
    raw_counter = Counter()

    def _scan(el):
        for k, v in el.attrib.items():
            for m in re.findall(r'#([0-9a-fA-F]{6})', v):
                norm = _normalize_hex(m)
                if norm:
                    # Weight by attribute type
                    weight = 1
                    if k in ('mark-color', 'color'):
                        weight = 5
                    elif k in ('fill', 'background-color'):
                        weight = 3
                    elif k in ('font-color', 'border-color'):
                        weight = 2
                    raw_counter[norm] += weight
        if el.text:
            for m in re.findall(r'#([0-9a-fA-F]{6})', el.text):
                norm = _normalize_hex(m)
                if norm:
                    raw_counter[norm] += 2  # formula-embedded colors
        for child in el:
            _scan(child)

    _scan(root)

    # Phase 2: Split into chromatic (saturated) and achromatic (gray) buckets
    # Boost vivid colors — they are the brand accents even if less frequent
    chromatic = {}
    achromatic = {}
    for hex6, count in raw_counter.items():
        if _is_noise_color(hex6):
            continue
        h, s, l = _hex_to_hsl(hex6)
        if s < 0.10:
            achromatic[hex6] = count
        else:
            # Saturation boost: vivid colors (S > 0.5) get 3x weight
            boost = 3.0 if s > 0.5 else (1.5 if s > 0.25 else 1.0)
            chromatic[hex6] = count * boost

    # Phase 3: Build palette — chromatic colors first (they define the brand),
    # then fill remaining slots with achromatic
    chrom_ranked = sorted(chromatic.keys(), key=lambda c: chromatic[c], reverse=True)
    achrom_ranked = sorted(achromatic.keys(), key=lambda c: achromatic[c], reverse=True)

    palette = chrom_ranked[:max_colors]
    remaining = max_colors - len(palette)
    if remaining > 0:
        palette.extend(achrom_ranked[:remaining])

    return [f"#{c}" for c in palette[:max_colors]]


def extract_all_colors(root):
    """Extract ALL color signals from a Tableau workbook XML root.

    Args:
        root: defusedxml.ElementTree Element (TWB root or workbook element).

    Returns:
        dict with keys: palette_colors, mark_colors, gradient_stops,
        conditional_colors, reference_line_colors, annotation_colors,
        dashboard_bg, worksheet_bgs, border_colors, tooltip_colors,
        title_colors. All color values are 6-char lowercase hex (no #).
    """
    result = {
        "palette_colors": [],
        "mark_colors": [],
        "gradient_stops": [],
        "conditional_colors": [],
        "reference_line_colors": [],
        "annotation_colors": [],
        "dashboard_bg": "",
        "worksheet_bgs": {},
        "border_colors": [],
        "tooltip_colors": [],
        "title_colors": {},
    }

    # 1. Palette colors from <color-palette> elements
    for cp in root.iter("color-palette"):
        for c_el in cp.findall("color"):
            if c_el.text:
                norm = _normalize_hex(c_el.text)
                if norm:
                    result["palette_colors"].append(norm)

    # 2. Mark colors from worksheet style-rules and encoding elements
    for ws in root.iter("worksheet"):
        ws_name = ws.get("name", "")
        for sr in ws.iter("style-rule"):
            if sr.get("element") == "mark":
                # Discrete color mappings
                for enc in sr.findall("encoding"):
                    if enc.get("attr") == "color":
                        enc_type = enc.get("type", "").lower()
                        # Discrete palette maps
                        for m in enc.findall("map"):
                            color = m.get("to", "")
                            norm = _normalize_hex(color)
                            if norm:
                                result["mark_colors"].append(norm)
                        # Gradient / sequential / diverging stops
                        if enc_type in ("interpolated", "quantize", "quantile"):
                            for rng in enc.iter("range"):
                                for attr_name in ("from", "to"):
                                    val = rng.get(attr_name, "")
                                    norm = _normalize_hex(val)
                                    if norm:
                                        result["gradient_stops"].append(norm)
                # Mark-color from format attrs
                for fmt in sr.findall("format"):
                    attr = fmt.get("attr", "")
                    val = fmt.get("value", "")
                    if attr == "mark-color" and val:
                        norm = _normalize_hex(val)
                        if norm:
                            result["mark_colors"].append(norm)

    # 3. Gradient stops from datasource-level color encodings
    for ds in root.iter("datasource"):
        for enc in ds.iter("encoding"):
            if enc.get("attr") == "color":
                enc_type = enc.get("type", "").lower()
                if enc_type in ("interpolated", "quantize", "quantile"):
                    for rng in enc.iter("range"):
                        for attr_name in ("from", "to"):
                            val = rng.get(attr_name, "")
                            norm = _normalize_hex(val)
                            if norm:
                                result["gradient_stops"].append(norm)

    # 4. Reference line colors
    for rl in root.iter("reference-line"):
        for fmt in rl.iter("format"):
            attr = fmt.get("attr", "")
            val = fmt.get("value", "")
            if attr in ("line-color", "fill-color", "band-color") and val:
                norm = _normalize_hex(val)
                if norm:
                    result["reference_line_colors"].append(norm)
        # Also check line element directly
        line_color = rl.get("line-color", "")
        if line_color:
            norm = _normalize_hex(line_color)
            if norm:
                result["reference_line_colors"].append(norm)

    # 5. Annotation colors
    for ann in root.iter("point-annotation"):
        _extract_annotation_colors(ann, result)
    for ann in root.iter("mark-annotation"):
        _extract_annotation_colors(ann, result)
    for ann in root.iter("zone-annotation"):
        _extract_annotation_colors(ann, result)

    # 6. Dashboard background colors
    for dash in root.iter("dashboard"):
        for sr in dash.iter("style-rule"):
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr in ("background-color", "fill") and val:
                    norm = _normalize_hex(val)
                    if norm:
                        result["dashboard_bg"] = norm

    # 7. Worksheet background colors
    for ws in root.iter("worksheet"):
        ws_name = ws.get("name", "")
        for sr in ws.iter("style-rule"):
            if sr.get("element") in ("worksheet", "pane"):
                for fmt in sr.findall("format"):
                    attr = fmt.get("attr", "")
                    val = fmt.get("value", "")
                    if attr in ("background-color", "fill") and val:
                        norm = _normalize_hex(val)
                        if norm:
                            result["worksheet_bgs"][ws_name] = norm

    # 8. Border colors from style rules
    for sr in root.iter("style-rule"):
        for fmt in sr.findall("format"):
            attr = fmt.get("attr", "")
            val = fmt.get("value", "")
            if "border" in attr and "color" in attr and val:
                norm = _normalize_hex(val)
                if norm:
                    result["border_colors"].append(norm)

    # 9. Tooltip colors
    for sr in root.iter("style-rule"):
        if sr.get("element") == "tooltip":
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr in ("font-color", "background-color", "fill") and val:
                    norm = _normalize_hex(val)
                    if norm:
                        result["tooltip_colors"].append(norm)

    # 10. Title colors per worksheet
    for ws in root.iter("worksheet"):
        ws_name = ws.get("name", "")
        for sr in ws.iter("style-rule"):
            if sr.get("element") == "title":
                for fmt in sr.findall("format"):
                    attr = fmt.get("attr", "")
                    val = fmt.get("value", "")
                    if attr == "font-color" and val:
                        norm = _normalize_hex(val)
                        if norm:
                            result["title_colors"][ws_name] = norm

    # 11. Conditional formatting colors (calc-based color rules)
    for calc in root.iter("calculation"):
        formula = calc.get("formula", "")
        # Extract hex colors embedded in calculated field formulas
        hex_matches = re.findall(r'#([0-9a-fA-F]{6})', formula)
        for hm in hex_matches:
            norm = _normalize_hex(hm)
            if norm:
                result["conditional_colors"].append(norm)

    return result


def _extract_annotation_colors(ann_element, result):
    """Extract colors from an annotation element into result dict."""
    for fmt in ann_element.iter("format"):
        attr = fmt.get("attr", "")
        val = fmt.get("value", "")
        if attr in ("font-color", "border-color", "fill", "background-color") and val:
            norm = _normalize_hex(val)
            if norm:
                result["annotation_colors"].append(norm)


def build_unified_palette(extracted_colors, max_colors=12):
    """Build a deduplicated, frequency-ranked color palette.

    Filters out near-white/near-black noise. Returns hex colors with # prefix.

    Args:
        extracted_colors: dict from extract_all_colors().
        max_colors: maximum palette size.

    Returns:
        list of hex color strings (e.g. ["#4e79a7", "#f28e2b", ...]).
    """
    counter = Counter()

    # Weight sources by importance
    weights = {
        "palette_colors": 3,
        "mark_colors": 5,     # Mark colors are the most visible
        "gradient_stops": 2,
        "conditional_colors": 2,
        "reference_line_colors": 1,
        "annotation_colors": 1,
        "border_colors": 1,
        "tooltip_colors": 1,
    }

    for key, weight in weights.items():
        colors = extracted_colors.get(key, [])
        if isinstance(colors, list):
            for c in colors:
                if c and not _is_noise_color(c):
                    counter[c] += weight

    # Add dashboard and worksheet bg colors with low weight
    dash_bg = extracted_colors.get("dashboard_bg", "")
    if dash_bg and not _is_noise_color(dash_bg):
        counter[dash_bg] += 1

    for ws_name, bg in extracted_colors.get("worksheet_bgs", {}).items():
        if bg and not _is_noise_color(bg):
            counter[bg] += 1

    # Title colors
    for ws_name, tc in extracted_colors.get("title_colors", {}).items():
        if tc and not _is_noise_color(tc):
            counter[tc] += 2

    if not counter:
        return []

    # Sort by frequency (descending), take top N
    ranked = [c for c, _ in counter.most_common(max_colors)]
    return [f"#{c}" for c in ranked]


def extract_deep_palette(root, max_colors=12):
    """Deep palette extraction -- checks all XML elements for hex colors.

    Goes beyond mark_colors to find colors in:
    - <color-palette> definitions
    - <preferences> elements
    - <style-rule> formatting
    - <format> elements (font colors, background colors)
    - <encoding> color definitions
    - Any attribute containing a hex color pattern

    Returns list of hex color strings with # prefix, ranked by frequency.
    """
    import re
    from collections import Counter

    hex_pattern = re.compile(r'#([0-9a-fA-F]{6})\b')
    counter = Counter()

    # Walk the entire XML tree and extract hex colors from all attributes
    for elem in root.iter():
        # Check all attributes
        for attr_name, attr_val in elem.attrib.items():
            if not attr_val:
                continue
            matches = hex_pattern.findall(attr_val)
            for m in matches:
                c = m.lower()
                if not _is_noise_color(c):
                    counter[c] += 1

        # Check text content
        if elem.text:
            matches = hex_pattern.findall(elem.text)
            for m in matches:
                c = m.lower()
                if not _is_noise_color(c):
                    counter[c] += 1

    # Also run the standard extraction for weighted scoring
    try:
        extracted = extract_all_colors(root)
        for c in extracted.get("palette_colors", []):
            if c and not _is_noise_color(c):
                counter[c] += 5  # Palette definitions are highest priority
        for c in extracted.get("mark_colors", []):
            if c and not _is_noise_color(c):
                counter[c] += 3
    except Exception:
        pass

    if not counter:
        return []

    ranked = [c for c, _ in counter.most_common(max_colors)]
    return [f"#{c}" for c in ranked]


def compute_color_harmony(palette):
    """Analyze the color harmony type of a palette.

    Args:
        palette: list of hex color strings (with # prefix).

    Returns:
        dict with harmony_type (str) and confidence (float 0-1).
        harmony_type is one of: monochromatic, complementary, analogous,
        triadic, split_complementary, diverse.
    """
    if not palette or len(palette) < 2:
        return {"harmony_type": "monochromatic", "confidence": 1.0}

    hues = []
    for c in palette:
        hex6 = _normalize_hex(c)
        if hex6:
            h, s, l = _hex_to_hsl(hex6)
            if s > 0.1:  # Skip near-gray colors
                hues.append(h)

    if len(hues) < 2:
        return {"harmony_type": "monochromatic", "confidence": 0.9}

    # Calculate pairwise hue differences
    diffs = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            diff = abs(hues[i] - hues[j])
            if diff > 180:
                diff = 360 - diff
            diffs.append(diff)

    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    max_diff = max(diffs) if diffs else 0

    # Classify based on hue distribution
    if max_diff < 30:
        return {"harmony_type": "monochromatic", "confidence": 0.9 - (max_diff / 60)}

    if len(hues) == 2 and 150 < diffs[0] < 210:
        return {"harmony_type": "complementary", "confidence": 1.0 - abs(diffs[0] - 180) / 60}

    if max_diff < 60:
        return {"harmony_type": "analogous", "confidence": 0.9 - (max_diff / 120)}

    # Check for triadic (3 colors ~120 degrees apart)
    if len(hues) >= 3:
        sorted_hues = sorted(hues[:3])
        gaps = [
            sorted_hues[1] - sorted_hues[0],
            sorted_hues[2] - sorted_hues[1],
            360 - sorted_hues[2] + sorted_hues[0],
        ]
        triadic_score = 1.0 - (max(abs(g - 120) for g in gaps) / 120)
        if triadic_score > 0.5:
            return {"harmony_type": "triadic", "confidence": max(0.0, min(1.0, triadic_score))}

    # Check for split complementary
    if len(hues) >= 3:
        base = hues[0]
        comp = (base + 180) % 360
        split_diffs = [min(abs(h - comp), 360 - abs(h - comp)) for h in hues[1:]]
        if any(20 < d < 50 for d in split_diffs):
            return {"harmony_type": "split_complementary", "confidence": 0.6}

    return {"harmony_type": "diverse", "confidence": max(0.0, min(1.0, 1.0 - avg_diff / 180))}
