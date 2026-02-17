"""
Agent Session Bridge.

Connects MultiFileSession to Dr. Data's agent. Generates the context
string that gets injected into every message so the agent KNOWS what
files it has, what data is loaded, and what it can build.

Drop this in: core/agent_session.py

Usage in dr_data_agent.py:
    from core.agent_session import AgentSessionBridge
    self.session_bridge = AgentSessionBridge(self.multi_file_session)
    context = self.session_bridge.get_context()
    # Prepend context to user message before sending to Claude
"""


class AgentSessionBridge:

    def __init__(self, session):
        """
        Args:
            session: MultiFileSession instance
        """
        self.session = session

    def get_context(self):
        """
        Build the context string injected before every user message.
        This is what makes Dr. Data aware of all uploaded files.
        """
        s = self.session
        summary = s.get_summary()
        parts = []

        if summary["file_count"] == 0:
            return ""

        parts.append("[SYSTEM CONTEXT -- DO NOT REPEAT THIS TO THE USER]")

        # List all files
        file_lines = []
        for f in summary["files"]:
            line = f"{f['filename']} ({f['category']}, {f['size_kb']} KB)"
            if "rows" in f:
                line += f" -- {f['rows']:,} rows, {len(f.get('columns', []))} cols"
            if "structure_summary" in f:
                line += f" -- {f['structure_summary']}"
            file_lines.append(line)
        parts.append(f"Files loaded: {'; '.join(file_lines)}")

        # Data availability
        if summary["has_data"]:
            shape = summary.get("primary_df_shape", {})
            if shape:
                cols = shape.get("column_names", [])
                parts.append(
                    f"Primary dataset: {shape['rows']:,} rows, {shape['columns']} columns: "
                    f"{', '.join(cols[:15])}"
                    f"{'...' if len(cols) > 15 else ''}"
                )
            parts.append("You have the data. Do NOT ask the user for a file path.")
        else:
            parts.append("No data loaded yet.")

        # Structure file details
        if summary["has_structure"]:
            stype = summary["structure_type"]

            if stype in ("tableau_structure", "tableau_packaged") and "tableau" in summary:
                t = summary["tableau"]
                parts.append(
                    f"Tableau structure parsed: {t['datasource_count']} data sources, "
                    f"{t['worksheet_count']} worksheets, {t['dashboard_count']} dashboards, "
                    f"{t['calculated_fields']} calculated fields, {t['parameters']} parameters"
                )

                # Data source mapping status
                mapping = summary.get("data_source_mapping", {})
                unmapped = t.get("unmapped_sources", [])
                if mapping:
                    mapped_lines = []
                    for ds_name, info in mapping.items():
                        mapped_lines.append(f"{ds_name} -> {info['data_file']}")
                    parts.append(f"Data source mapping: {'; '.join(mapped_lines)}")

                if unmapped:
                    parts.append(
                        f"UNMAPPED data sources (need data files): {', '.join(unmapped)}. "
                        f"Ask the user to upload the data files for these sources."
                    )

            elif stype == "alteryx" and "alteryx" in summary:
                a = summary["alteryx"]
                parts.append(
                    f"Alteryx workflow parsed: {a['tool_count']} tools, "
                    f"{a['connection_count']} connections"
                )

        # What the agent should do next
        if summary["needs_data"]:
            unmapped = []
            if "tableau" in summary:
                unmapped = summary["tableau"].get("unmapped_sources", [])
            if unmapped:
                parts.append(
                    f"ACTION NEEDED: Ask the user to upload data files for: "
                    f"{', '.join(unmapped)}. Say something like: 'I have parsed "
                    f"your Tableau workbook. I can see the structure but I need "
                    f"the underlying data. Can you upload the data files "
                    f"({', '.join(unmapped)})?'"
                )
            else:
                parts.append(
                    "ACTION NEEDED: Structure file loaded but no data. Ask the "
                    "user to upload the data file(s)."
                )
        elif summary["ready_for_build"]:
            parts.append(
                "READY TO BUILD. You have structure + data. Ask the user what "
                "output they want: Interactive HTML Dashboard, Power BI Project, "
                "or both. Then build it."
            )

        parts.append("[END SYSTEM CONTEXT]")
        return "\n".join(parts)

    def get_upload_response_hint(self, filename):
        """
        After a file upload, suggest what Dr. Data should say.
        Returns a hint string the agent can use (not shown to user).
        """
        s = self.session
        summary = s.get_summary()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Tableau workbook uploaded
        if ext in ("twb", "twbx"):
            t = summary.get("tableau", {})
            if summary["needs_data"]:
                unmapped = t.get("unmapped_sources", [])
                return (
                    f"RESPOND: Tell user you parsed the Tableau workbook. Give stats: "
                    f"{t.get('worksheet_count',0)} worksheets, "
                    f"{t.get('dashboard_count',0)} dashboards, "
                    f"{t.get('calculated_fields',0)} calculated fields. "
                    f"Then ask them to upload the data file(s) for: {', '.join(unmapped)}. "
                    f"Be specific about what you need."
                )
            else:
                return (
                    f"RESPOND: Tell user you parsed the Tableau workbook AND have the data. "
                    f"Give stats with SPECIFIC NUMBERS from the data. "
                    f"Ask what they want: HTML dashboard, Power BI project, or both."
                )

        # Data file uploaded after a structure file
        if ext in ("csv", "xlsx", "xls", "tsv", "parquet", "json"):
            if summary["has_structure"] and summary["ready_for_build"]:
                mapping = summary.get("data_source_mapping", {})
                mapped_to = [f"{k} -> {v['data_file']}" for k, v in mapping.items()]
                shape = summary.get("primary_df_shape", {})
                return (
                    f"RESPOND: Data file received and mapped: {'; '.join(mapped_to)}. "
                    f"Dataset has {shape.get('rows', '?')} rows. "
                    f"Give the user SPECIFIC NUMBERS from the data (totals, ranges, etc). "
                    f"Ask what they want built: HTML dashboard, Power BI project, or both."
                )
            elif not summary["has_structure"]:
                shape = summary.get("primary_df_shape", {})
                return (
                    f"RESPOND: Data loaded -- {shape.get('rows', '?')} rows, "
                    f"{shape.get('columns', '?')} columns. "
                    f"Analyze the data and give SPECIFIC NUMBERS. "
                    f"Ask what they want built."
                )

        # Alteryx
        if ext in ("yxmd", "yxwz", "yxmc", "yxzp"):
            a = summary.get("alteryx", {})
            return (
                f"RESPOND: Alteryx workflow parsed -- {a.get('tool_count', '?')} tools. "
                f"Describe what the workflow does. Offer Dataiku migration report "
                f"or ask if they have the output data to build a dashboard."
            )

        return ""
