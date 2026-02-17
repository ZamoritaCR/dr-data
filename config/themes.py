"""
Color themes for charts and dashboards.
"""

DARK_PROFESSIONAL = {
    "bg_primary": "#0d1117",
    "bg_surface": "#161b22",
    "bg_elevated": "#1c2128",
    "border": "#30363d",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "accent": "#1f6feb",
    "success": "#238636",
    "warning": "#d29922",
    "danger": "#da3633",
    "chart_colors": [
        "#1f6feb", "#238636", "#d29922", "#da3633",
        "#a371f7", "#3fb950", "#d2a8ff", "#79c0ff",
        "#56d364", "#e3b341", "#ff7b72", "#ffa657"
    ],
    "plotly_template": {
        "layout": {
            "paper_bgcolor": "#0d1117",
            "plot_bgcolor": "#161b22",
            "font": {"color": "#e6edf3", "family": "system-ui, sans-serif"},
            "title": {"font": {"size": 16}},
            "xaxis": {
                "gridcolor": "#30363d",
                "linecolor": "#30363d",
                "zerolinecolor": "#30363d"
            },
            "yaxis": {
                "gridcolor": "#30363d",
                "linecolor": "#30363d",
                "zerolinecolor": "#30363d"
            },
            "colorway": [
                "#1f6feb", "#238636", "#d29922", "#da3633",
                "#a371f7", "#3fb950", "#d2a8ff", "#79c0ff"
            ],
            "margin": {"l": 60, "r": 30, "t": 50, "b": 50}
        }
    }
}
