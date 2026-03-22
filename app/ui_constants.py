"""
Global UI constants and design system for Dr. Data.
Art of the Possible -- never-seen-before design.
"""

# Color palette
COLORS = {
    "bg": "#070A0F",
    "surface": "#0D1520",
    "surface2": "#141E2E",
    "border": "#1E2D42",
    "green": "#00D47E",
    "green_dim": "rgba(0,212,126,0.12)",
    "amber": "#F59E0B",
    "amber_dim": "rgba(245,158,11,0.12)",
    "red": "#EF4444",
    "blue": "#3B82F6",
    "blue_dim": "rgba(59,130,246,0.12)",
    "purple": "#A78BFA",
    "text": "#E2E8F0",
    "text_dim": "#94A3B8",
    "text_muted": "#4E6580",
}

# Tab identities -- each tab has a unique accent color
TAB_COLORS = {
    "migration": "#00D47E",
    "dq": "#3B82F6",
    "rationalization": "#F59E0B",
    "modeler": "#A78BFA",
}

# Shared CSS injected into every page
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, .stApp { background-color: #070A0F !important; }
.stApp { font-family: 'DM Sans', sans-serif !important; }

/* Tab navigation */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1520 !important;
    border-bottom: 1px solid #1E2D42 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #4E6580 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    color: #E2E8F0 !important;
    border-bottom: 2px solid #00D47E !important;
    background: transparent !important;
}

/* Chat message styling */
.stChatMessage { border-radius: 10px !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0D1520 !important;
    border: 1px solid #1E2D42 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* Dataframe / tables */
.stDataFrame { border: 1px solid #1E2D42 !important; border-radius: 8px !important; }

/* Buttons */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: #00D47E !important;
    color: #000 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00F090 !important;
    box-shadow: 0 0 12px rgba(0,212,126,0.35) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1520 !important;
    border-right: 1px solid #1E2D42 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #0D1520 !important;
    border: 1px solid #1E2D42 !important;
    border-radius: 8px !important;
}

/* Code blocks */
.stCode { background: #040609 !important; border: 1px solid #1E2D42 !important; }

/* Info / warning / error boxes */
.stAlert[data-baseweb="notification"] { border-radius: 8px !important; }

/* Tab indicator badges */
.tab-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 600;
}
</style>
"""


def tab_header(title, subtitle, accent="#00D47E", icon="//"):
    """Returns an HTML header block for a tab."""
    return (
        f"<div style='padding:20px 0 16px 0;border-bottom:1px solid #1E2D42;margin-bottom:20px;'>"
        f"<div style='display:flex;align-items:center;gap:12px;'>"
        f"<div style='width:36px;height:36px;border-radius:9px;"
        f"background:linear-gradient(135deg,{accent},{accent}88);"
        f"display:flex;align-items:center;justify-content:center;font-size:18px;"
        f"box-shadow:0 0 12px {accent}40;'>{icon}</div>"
        f"<div>"
        f"<div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;"
        f"letter-spacing:-.01em;color:#E2E8F0;'>{title}</div>"
        f"<div style='font-size:13px;color:#4E6580;margin-top:2px;'>{subtitle}</div>"
        f"</div></div></div>"
    )


def badge(text, color="green"):
    """Returns an HTML badge."""
    colors = {
        "green": ("rgba(0,212,126,0.12)", "#00D47E", "rgba(0,212,126,0.25)"),
        "amber": ("rgba(245,158,11,0.12)", "#F59E0B", "rgba(245,158,11,0.25)"),
        "blue": ("rgba(59,130,246,0.12)", "#3B82F6", "rgba(59,130,246,0.25)"),
        "red": ("rgba(239,68,68,0.1)", "#EF4444", "rgba(239,68,68,0.25)"),
        "purple": ("rgba(167,139,250,0.12)", "#A78BFA", "rgba(167,139,250,0.25)"),
    }
    bg, fg, border = colors.get(color, colors["green"])
    return (
        f"<span style='background:{bg};color:{fg};border:1px solid {border};"
        f"font-family:JetBrains Mono,monospace;font-size:10.5px;font-weight:600;"
        f"padding:3px 8px;border-radius:4px;'>{text}</span>"
    )
