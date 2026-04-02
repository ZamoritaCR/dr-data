"""
Layout Assembler -- converts a RequirementsContract into PBI report_layout sections.

Produces pixel-precise positions on a 1280x720 canvas, obeying all layout
constraints from the contract. Pure geometry -- no LLM calls.

Output format matches what PBIPGenerator._write_pages() consumes:
  sections: [
    {
      "name": "page_id",
      "displayName": "Page Title",
      "width": 1280,
      "height": 720,
      "visualContainers": [
        {"x": int, "y": int, "width": int, "height": int,
         "config": {"visualType": str, "title": str, "dataRoles": dict}}
      ]
    }
  ]
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Dict

from core.requirements_contract import (
    RequirementsContract,
    VisualContract,
    VisualType,
    FilterContractSpec,
    NavigationElement,
    LayoutConstraint,
)


# ================================================================== #
#  Canvas constants                                                     #
# ================================================================== #

CANVAS_W = 1280
CANVAS_H = 720

# Spacing
MARGIN = 12           # Edge margin
GAP = 8               # Gap between visuals
SLICER_H = 52         # Height of a slicer row
NAV_H = 40            # Height of a nav/tab-selector row
HEADER_H = 36         # Height of a section header / title card
CARD_H = 80           # Height of a KPI card

# Minimum visual height
MIN_VISUAL_H = 100


# ================================================================== #
#  Layout result types                                                  #
# ================================================================== #

@dataclass
class PlacedVisual:
    """A visual with computed pixel position."""
    visual_id: str
    x: int
    y: int
    width: int
    height: int
    visual_type: str
    title: str = ""
    data_roles: dict = None
    source_worksheet: str = ""

    def __post_init__(self):
        if self.data_roles is None:
            self.data_roles = {}

    def to_container(self) -> dict:
        """Convert to the visualContainers format PBIPGenerator expects."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "config": {
                "visualType": self.visual_type,
                "title": self.title,
                "dataRoles": self.data_roles,
            },
        }


@dataclass
class AssembledPage:
    """A fully laid-out page."""
    name: str
    display_name: str
    width: int = CANVAS_W
    height: int = CANVAS_H
    visuals: List[PlacedVisual] = None

    def __post_init__(self):
        if self.visuals is None:
            self.visuals = []

    def to_section(self) -> dict:
        """Convert to the sections format PBIPGenerator expects."""
        return {
            "name": self.name,
            "displayName": self.display_name,
            "width": self.width,
            "height": self.height,
            "displayOption": 1,  # fit to page
            "visualContainers": [v.to_container() for v in self.visuals],
        }


@dataclass
class AssembledLayout:
    """Complete layout result."""
    pages: List[AssembledPage] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.pages is None:
            self.pages = []
        if self.warnings is None:
            self.warnings = []

    def to_report_layout(self) -> dict:
        """Convert to the report_layout dict PBIPGenerator expects."""
        return {
            "sections": [p.to_section() for p in self.pages],
        }


# ================================================================== #
#  Constraint readers                                                   #
# ================================================================== #

def _get_constraint(contract: RequirementsContract, ctype: str) -> Optional[str]:
    """Get the value of a layout constraint by type."""
    for lc in contract.layout_constraints:
        if lc.constraint_type == ctype:
            return lc.value
    return None


def _get_stacking(contract: RequirementsContract) -> str:
    """Get stacking direction from contract. Default: vertical."""
    return _get_constraint(contract, "stacking_direction") or "vertical"


def _is_single_page(contract: RequirementsContract) -> bool:
    """Check if single-page is enforced."""
    return _get_constraint(contract, "page_count_fixed") == "1"


# ================================================================== #
#  Data role builder                                                    #
# ================================================================== #

def _build_data_roles(vc: VisualContract) -> dict:
    """Build PBI data_roles dict from a VisualContract."""
    roles = {}

    if vc.category_axis and vc.category_axis.field:
        roles["category"] = [vc.category_axis.field]
    if vc.value_axis and vc.value_axis.field:
        roles["values"] = [vc.value_axis.field]
    if vc.secondary_axis and vc.secondary_axis.field:
        if "values" in roles:
            roles["values"].append(vc.secondary_axis.field)
        else:
            roles["values"] = [vc.secondary_axis.field]
    if vc.color_field:
        roles["series"] = [vc.color_field]

    return roles


def _slicer_data_roles(filt: FilterContractSpec) -> dict:
    """Build data roles for a slicer visual."""
    if filt.field:
        return {"category": [filt.field]}
    return {}


def _nav_data_roles(nav: NavigationElement) -> dict:
    """Build data roles for a navigation slicer."""
    if nav.selector_field:
        return {"category": [nav.selector_field]}
    return {}


# ================================================================== #
#  Main assembler                                                       #
# ================================================================== #

def assemble(contract: RequirementsContract) -> AssembledLayout:
    """Assemble a complete PBI layout from a RequirementsContract.

    This is the main entry point. Deterministic -- same contract always
    produces the same layout.
    """
    layout = AssembledLayout()

    if contract.page_count < 1:
        layout.warnings.append("Contract has page_count < 1, forcing 1 page")

    page_count = max(1, contract.page_count)

    # Group visuals, filters, nav by page
    for page_idx in range(page_count):
        page_name = (
            contract.page_names[page_idx]
            if page_idx < len(contract.page_names)
            else f"Page {page_idx + 1}"
        )

        # Collect items for this page
        page_visuals = [
            v for v in contract.visuals
            if v.page == page_name or (not v.page and page_idx == 0)
        ]
        page_filters = [
            f for f in contract.top_filters
            if f.page == page_name or not f.page
        ]
        page_nav = contract.navigation_elements  # Nav typically appears on all pages

        # Sort visuals by stacking_order
        page_visuals.sort(key=lambda v: v.stacking_order)

        # Assemble the page
        assembled_page = _assemble_page(
            page_name=page_name,
            visuals=page_visuals,
            filters=page_filters,
            nav_elements=page_nav,
            stacking=_get_stacking(contract),
            contract=contract,
        )
        layout.pages.append(assembled_page)

    return layout


def _assemble_page(
    page_name: str,
    visuals: List[VisualContract],
    filters: List[FilterContractSpec],
    nav_elements: List[NavigationElement],
    stacking: str,
    contract: RequirementsContract,
) -> AssembledPage:
    """Assemble a single page with precise positioning."""

    page = AssembledPage(
        name=page_name.replace(" ", ""),
        display_name=page_name,
    )

    usable_x = MARGIN
    usable_w = CANVAS_W - 2 * MARGIN
    cursor_y = MARGIN  # Tracks vertical position as we stack rows

    # -- Row 1: Title / header card + optional refresh text --
    refresh_constraint = next(
        (lc for lc in contract.layout_constraints if lc.constraint_type == "refresh_text"),
        None
    )
    if contract.dashboard_title:
        title_w = usable_w
        if refresh_constraint:
            title_w = usable_w - 200 - GAP  # Leave room for refresh text
        page.visuals.append(PlacedVisual(
            visual_id="title_card",
            x=usable_x,
            y=cursor_y,
            width=title_w,
            height=HEADER_H,
            visual_type="textbox",
            title=contract.dashboard_title,
        ))
        if refresh_constraint:
            page.visuals.append(PlacedVisual(
                visual_id="refresh_text",
                x=usable_x + title_w + GAP,
                y=cursor_y,
                width=200,
                height=HEADER_H,
                visual_type="textbox",
                title="Last Refresh",
            ))
        cursor_y += HEADER_H + GAP
    elif refresh_constraint:
        # No title but refresh text requested
        page.visuals.append(PlacedVisual(
            visual_id="refresh_text",
            x=CANVAS_W - MARGIN - 200,
            y=cursor_y,
            width=200,
            height=HEADER_H,
            visual_type="textbox",
            title="Last Refresh",
        ))
        cursor_y += HEADER_H + GAP

    # -- Row 2: Top-row slicers --
    top_filters = [f for f in filters if f.position == "top"]
    if top_filters:
        slicer_count = len(top_filters)
        slicer_w = (usable_w - (slicer_count - 1) * GAP) // slicer_count

        for idx, filt in enumerate(top_filters):
            sx = usable_x + idx * (slicer_w + GAP)
            page.visuals.append(PlacedVisual(
                visual_id=f"slicer_{filt.field or idx}",
                x=sx,
                y=cursor_y,
                width=slicer_w,
                height=SLICER_H,
                visual_type="slicer",
                title=filt.title or filt.field,
                data_roles=_slicer_data_roles(filt),
            ))

        cursor_y += SLICER_H + GAP

    # -- Row 3: Horizontal navigation / tab selector --
    tab_navs = [n for n in nav_elements if n.element_type == "tab_selector"]
    if tab_navs:
        for nav in tab_navs:
            page.visuals.append(PlacedVisual(
                visual_id=f"nav_{nav.selector_field or 'tab'}",
                x=usable_x,
                y=cursor_y,
                width=usable_w,
                height=NAV_H,
                visual_type="slicer",
                title=nav.selector_field or "Category",
                data_roles=_nav_data_roles(nav),
            ))
            cursor_y += NAV_H + GAP

    # -- Right-sidebar filters --
    right_filters = [f for f in filters if f.position == "right"]
    right_sidebar_w = 0
    if right_filters:
        right_sidebar_w = 200
        sidebar_x = CANVAS_W - MARGIN - right_sidebar_w
        sidebar_y = cursor_y

        for idx, filt in enumerate(right_filters):
            page.visuals.append(PlacedVisual(
                visual_id=f"slicer_right_{filt.field or idx}",
                x=sidebar_x,
                y=sidebar_y,
                width=right_sidebar_w,
                height=SLICER_H * 2,
                visual_type="slicer",
                title=filt.title or filt.field,
                data_roles=_slicer_data_roles(filt),
            ))
            sidebar_y += SLICER_H * 2 + GAP

    # -- Main content area --
    content_w = usable_w - right_sidebar_w - (GAP if right_sidebar_w else 0)
    remaining_h = CANVAS_H - cursor_y - MARGIN

    # Separate out special visuals from chart content
    kpi_visuals = [v for v in visuals
                   if v.visual_type in (VisualType.CARD, VisualType.KPI)]
    textbox_visuals = [v for v in visuals
                       if v.visual_type == VisualType.TEXT_BOX]
    content_visuals = [v for v in visuals
                       if v not in kpi_visuals and v not in textbox_visuals]

    # -- Section labels (text boxes that are not refresh/title) --
    section_labels = [v for v in textbox_visuals
                      if "section" in v.visual_id or "label" in v.visual_id.lower()]

    # -- KPI card row (if any KPI cards) --
    if kpi_visuals:
        kpi_count = len(kpi_visuals)
        kpi_w = (content_w - (kpi_count - 1) * GAP) // kpi_count

        for idx, vc in enumerate(kpi_visuals):
            kx = usable_x + idx * (kpi_w + GAP)
            page.visuals.append(PlacedVisual(
                visual_id=vc.visual_id,
                x=kx,
                y=cursor_y,
                width=kpi_w,
                height=CARD_H,
                visual_type=vc.visual_type.value if isinstance(vc.visual_type, VisualType) else str(vc.visual_type),
                title=vc.title,
                data_roles=_build_data_roles(vc),
                source_worksheet=vc.source_worksheet,
            ))

        cursor_y += CARD_H + GAP
        remaining_h = CANVAS_H - cursor_y - MARGIN

    # -- Section labels before content --
    for sl in section_labels:
        page.visuals.append(PlacedVisual(
            visual_id=sl.visual_id,
            x=usable_x,
            y=cursor_y,
            width=content_w,
            height=HEADER_H,
            visual_type="textbox",
            title=sl.title,
        ))
        cursor_y += HEADER_H + GAP
        remaining_h = CANVAS_H - cursor_y - MARGIN

    # -- Content visuals layout --
    if not content_visuals:
        return page

    if stacking == "vertical":
        _layout_vertical_stack(page, content_visuals, usable_x, cursor_y,
                               content_w, remaining_h)
    elif stacking == "horizontal":
        _layout_horizontal(page, content_visuals, usable_x, cursor_y,
                           content_w, remaining_h)
    elif stacking == "grid":
        _layout_grid(page, content_visuals, usable_x, cursor_y,
                     content_w, remaining_h)
    else:
        # Default: vertical
        _layout_vertical_stack(page, content_visuals, usable_x, cursor_y,
                               content_w, remaining_h)

    return page


# ================================================================== #
#  Stacking strategies                                                  #
# ================================================================== #

def _layout_vertical_stack(
    page: AssembledPage,
    visuals: List[VisualContract],
    x: int, start_y: int,
    width: int, available_h: int,
):
    """Stack visuals vertically, equal width, distributed height."""
    n = len(visuals)
    if n == 0:
        return

    total_gap = (n - 1) * GAP
    per_visual_h = max(MIN_VISUAL_H, (available_h - total_gap) // n)

    cursor_y = start_y
    for vc in visuals:
        # If contract specifies height_pct, use it
        if vc.height_pct > 0:
            h = max(MIN_VISUAL_H, int(CANVAS_H * vc.height_pct / 100))
        else:
            h = per_visual_h

        page.visuals.append(PlacedVisual(
            visual_id=vc.visual_id,
            x=x,
            y=cursor_y,
            width=width,
            height=h,
            visual_type=_vtype_str(vc),
            title=vc.title,
            data_roles=_build_data_roles(vc),
            source_worksheet=vc.source_worksheet,
        ))
        cursor_y += h + GAP


def _layout_horizontal(
    page: AssembledPage,
    visuals: List[VisualContract],
    x: int, start_y: int,
    width: int, available_h: int,
):
    """Place visuals side by side, equal height, distributed width."""
    n = len(visuals)
    if n == 0:
        return

    total_gap = (n - 1) * GAP
    per_visual_w = (width - total_gap) // n
    h = available_h

    cursor_x = x
    for vc in visuals:
        if vc.width_pct > 0 and vc.width_pct < 100:
            w = max(100, int(CANVAS_W * vc.width_pct / 100))
        else:
            w = per_visual_w

        page.visuals.append(PlacedVisual(
            visual_id=vc.visual_id,
            x=cursor_x,
            y=start_y,
            width=w,
            height=h,
            visual_type=_vtype_str(vc),
            title=vc.title,
            data_roles=_build_data_roles(vc),
            source_worksheet=vc.source_worksheet,
        ))
        cursor_x += w + GAP


def _layout_grid(
    page: AssembledPage,
    visuals: List[VisualContract],
    x: int, start_y: int,
    width: int, available_h: int,
):
    """Arrange visuals in a 2-column grid."""
    n = len(visuals)
    if n == 0:
        return

    cols = 2
    rows = (n + cols - 1) // cols
    cell_w = (width - (cols - 1) * GAP) // cols
    cell_h = max(MIN_VISUAL_H, (available_h - (rows - 1) * GAP) // rows)

    for idx, vc in enumerate(visuals):
        row = idx // cols
        col = idx % cols
        vx = x + col * (cell_w + GAP)
        vy = start_y + row * (cell_h + GAP)

        page.visuals.append(PlacedVisual(
            visual_id=vc.visual_id,
            x=vx,
            y=vy,
            width=cell_w,
            height=cell_h,
            visual_type=_vtype_str(vc),
            title=vc.title,
            data_roles=_build_data_roles(vc),
            source_worksheet=vc.source_worksheet,
        ))


# ================================================================== #
#  Helpers                                                              #
# ================================================================== #

def _vtype_str(vc: VisualContract) -> str:
    """Get visual type as string."""
    if isinstance(vc.visual_type, VisualType):
        return vc.visual_type.value
    return str(vc.visual_type)
