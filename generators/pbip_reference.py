"""
PBIP Reference Templates - Loaded from the reference_template project.

The reference_template was saved by PBI Desktop 2.150.2455.0 64-bit with
these preview features enabled:
  - PBI_gitIntegration
  - PBI_enhancedReportFormat
  - PBI_tmdlInDataset

All JSON templates and TMDL boilerplate are loaded from the reference files
at import time.  The generator only changes dynamic content (names, visuals,
columns, measures, data paths).  Structure and metadata stay identical.
"""

import json
from pathlib import Path

# ------------------------------------------------------------------ #
#  Reference template root (sibling of generators/)
# ------------------------------------------------------------------ #
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REFERENCE_REPORT_DIR = _PROJECT_ROOT / "reference_template.Report"
REFERENCE_MODEL_DIR = _PROJECT_ROOT / "reference_template.SemanticModel"


def _load_json(path, fallback):
    """Load JSON from a reference file, return fallback dict if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def _load_text(path, fallback):
    """Load text from a reference file, return fallback string if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return fallback


# ------------------------------------------------------------------ #
#  {name}.pbip  (root pointer - sibling to .Report and .SemanticModel)
# ------------------------------------------------------------------ #
PBIP_TEMPLATE = _load_json(
    _PROJECT_ROOT / "reference_template.pbip",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json",
        "version": "1.0",
        "artifacts": [{"report": {"path": None}}],
        "settings": {"enableAutoRecovery": True},
    },
)

# ------------------------------------------------------------------ #
#  .platform  (used for both .Report and .SemanticModel)
#  Loaded from .Report/.platform as the structural template
# ------------------------------------------------------------------ #
PLATFORM_TEMPLATE = _load_json(
    REFERENCE_REPORT_DIR / ".platform",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": None, "displayName": None},
        "config": {"version": "2.0", "logicalId": None},
    },
)

# ------------------------------------------------------------------ #
#  {name}.Report/definition.pbir  (report -> semantic model pointer)
# ------------------------------------------------------------------ #
PBIR_TEMPLATE = _load_json(
    REFERENCE_REPORT_DIR / "definition.pbir",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": None}, "byConnection": None},
    },
)

# ------------------------------------------------------------------ #
#  {name}.Report/definition/report.json  (theme + settings)
# ------------------------------------------------------------------ #
REPORT_JSON_TEMPLATE = _load_json(
    REFERENCE_REPORT_DIR / "definition" / "report.json",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.1.0/schema.json",
        "themeCollection": {
            "baseTheme": {
                "name": "CY25SU12",
                "reportVersionAtImport": {
                    "visual": "2.5.0", "report": "3.1.0", "page": "2.3.0",
                },
                "type": "SharedResources",
            }
        },
        "objects": {
            "section": [{
                "properties": {
                    "verticalAlignment": {
                        "expr": {"Literal": {"Value": "'Top'"}}
                    }
                }
            }]
        },
        "resourcePackages": [{
            "name": "SharedResources",
            "type": "SharedResources",
            "items": [{
                "name": "CY25SU12",
                "path": "BaseThemes/CY25SU12.json",
                "type": "BaseTheme",
            }],
        }],
        "settings": {
            "useStylableVisualContainerHeader": True,
            "exportDataMode": "AllowSummarized",
            "defaultDrillFilterOtherVisuals": True,
            "allowChangeFilterTypes": True,
            "useEnhancedTooltips": True,
            "useDefaultAggregateDisplayName": True,
        },
    },
)

# ------------------------------------------------------------------ #
#  {name}.Report/definition/version.json
# ------------------------------------------------------------------ #
VERSION_JSON_TEMPLATE = _load_json(
    REFERENCE_REPORT_DIR / "definition" / "version.json",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json",
        "version": "2.0.0",
    },
)

# ------------------------------------------------------------------ #
#  {name}.Report/definition/pages/pages.json
# ------------------------------------------------------------------ #
PAGES_JSON_TEMPLATE = {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json",
    "pageOrder": [],
    "activePageName": None,
}

# ------------------------------------------------------------------ #
#  {name}.Report/definition/pages/{pageId}/page.json
# ------------------------------------------------------------------ #
PAGE_JSON_TEMPLATE = {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.0.0/schema.json",
    "name": None,
    "displayName": None,
    "displayOption": "FitToPage",
    "height": 720,
    "width": 1280,
}

# ------------------------------------------------------------------ #
#  {name}.Report/definition/pages/{pageId}/visuals/{vizId}/visual.json
# ------------------------------------------------------------------ #
VISUAL_JSON_TEMPLATE = {
    "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.5.0/schema.json",
    "name": None,
    "position": {
        "x": 0, "y": 0, "z": 0,
        "height": 200, "width": 300, "tabOrder": 0,
    },
    "visual": {
        "visualType": None,
        "query": {"queryState": {}},
        "drillFilterOtherVisuals": True,
    },
}

# ------------------------------------------------------------------ #
#  {name}.SemanticModel/definition.pbism
# ------------------------------------------------------------------ #
PBISM_TEMPLATE = _load_json(
    REFERENCE_MODEL_DIR / "definition.pbism",
    {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
        "version": "4.2",
        "settings": {},
    },
)

# ------------------------------------------------------------------ #
#  {name}.SemanticModel/definition/database.tmdl
# ------------------------------------------------------------------ #
DATABASE_TMDL = _load_text(
    REFERENCE_MODEL_DIR / "definition" / "database.tmdl",
    "database\n\tcompatibilityLevel: 1600\n\n",
)

# ------------------------------------------------------------------ #
#  {name}.SemanticModel/definition/cultures/en-US.tmdl
# ------------------------------------------------------------------ #
CULTURE_TMDL = _load_text(
    REFERENCE_MODEL_DIR / "definition" / "cultures" / "en-US.tmdl",
    (
        "cultureInfo en-US\n"
        "\n"
        "\tlinguisticMetadata =\n"
        "\t\t\t{\n"
        '\t\t\t  "Version": "1.0.0",\n'
        '\t\t\t  "Language": "en-US"\n'
        "\t\t\t}\n"
        "\t\tcontentType: json\n"
        "\n"
    ),
)

# ------------------------------------------------------------------ #
#  CY25SU12.json theme - path relative to project root
# ------------------------------------------------------------------ #
THEME_SOURCE_PATH = "reference_template.Report/StaticResources/SharedResources/BaseThemes/CY25SU12.json"
