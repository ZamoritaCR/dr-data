"""
Data Lineage -- tracks where data comes from, what transformations
happen, and what downstream assets are affected.  A dependency graph
for data.
"""

import json
import os
from datetime import datetime, timezone
from collections import defaultdict, deque

_NODE_TYPES = {
    "source_system", "database", "schema", "table", "column",
    "etl_job", "transformation", "dashboard", "report",
    "api", "file", "model",
}
_RELATIONSHIPS = {
    "feeds", "derived_from", "transforms_to", "used_by",
    "references", "joins_with", "aggregates", "filters",
}
_FK_SUFFIXES = {
    "key", "id", "code", "custkey", "orderkey", "partkey",
    "suppkey", "nationkey", "regionkey",
}


class DataLineage:
    """Directed graph of data lineage nodes and edges."""

    def __init__(self, lineage_path="data_lineage.json"):
        self.lineage_path = lineage_path
        self.data = self._load()

    # ── Persistence ──

    def _load(self):
        try:
            if os.path.exists(self.lineage_path):
                with open(self.lineage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[LINEAGE] Loaded {len(data.get('nodes', {}))} nodes, "
                      f"{len(data.get('edges', []))} edges")
                return data
        except Exception as e:
            print(f"[LINEAGE] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "nodes": {},
            "edges": [],
        }

    def _save(self):
        try:
            self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.lineage_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"[LINEAGE] Save failed: {e}")

    # ── Graph Primitives ──

    def add_node(self, node_id, node_type, name, metadata=None):
        try:
            self.data["nodes"][node_id] = {
                "id": node_id, "type": node_type, "name": name,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"[LINEAGE] add_node failed: {e}")

    def _edge_exists(self, src, tgt, rel):
        return any(
            e.get("source") == src and e.get("target") == tgt
            and e.get("relationship") == rel
            for e in self.data["edges"]
        )

    def add_edge(self, source_id, target_id, relationship,
                 metadata=None):
        try:
            if self._edge_exists(source_id, target_id, relationship):
                return
            self.data["edges"].append({
                "source": source_id, "target": target_id,
                "relationship": relationship,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            print(f"[LINEAGE] add_edge failed: {e}")

    # ── Auto-Build ──

    def auto_build_lineage(self, tables_dict,
                           source_system="Snowflake",
                           database="SNOWFLAKE_SAMPLE_DATA",
                           schema="TPCH_SF1"):
        try:
            sys_id = f"sys_{source_system.lower()}"
            db_id = f"db_{database.lower()}"
            sch_id = f"schema_{schema.lower()}"

            self.add_node(sys_id, "source_system", source_system)
            self.add_node(db_id, "database", database)
            self.add_node(sch_id, "schema", schema)
            self.add_edge(sys_id, db_id, "feeds")
            self.add_edge(db_id, sch_id, "feeds")

            for tname, df in tables_dict.items():
                tid = f"table_{tname.lower()}"
                self.add_node(tid, "table", tname, {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                })
                self.add_edge(sch_id, tid, "feeds")
                for col in df.columns:
                    cid = f"col_{tname.lower()}_{col.lower()}"
                    self.add_node(cid, "column", f"{tname}.{col}", {
                        "dtype": str(df[col].dtype), "table": tname,
                    })
                    self.add_edge(tid, cid, "feeds")

            # Detect FK relationships
            tnames = list(tables_dict.keys())
            for i, t1 in enumerate(tnames):
                for t2 in tnames[i + 1:]:
                    for c1 in tables_dict[t1].columns:
                        c1s = c1.lower().split("_")[-1] if "_" in c1 else c1.lower()
                        if c1s not in _FK_SUFFIXES:
                            continue
                        for c2 in tables_dict[t2].columns:
                            c2s = c2.lower().split("_")[-1] if "_" in c2 else c2.lower()
                            if c1s == c2s:
                                cid1 = f"col_{t1.lower()}_{c1.lower()}"
                                cid2 = f"col_{t2.lower()}_{c2.lower()}"
                                self.add_edge(cid1, cid2, "references", {
                                    "relationship_type": "foreign_key",
                                    "confidence": "high",
                                })

            self._save()
            n = len(self.data["nodes"])
            e = len(self.data["edges"])
            print(f"[LINEAGE] Built lineage: {n} nodes, {e} edges")
            return {"nodes": n, "edges": e}
        except Exception as e:
            print(f"[LINEAGE] auto_build_lineage failed: {e}")
            return {"nodes": 0, "edges": 0}

    def add_deliverable_lineage(self, deliverable_name,
                                deliverable_type, source_tables,
                                columns_used=None):
        try:
            did = f"deliverable_{deliverable_name.lower().replace(' ', '_')}"
            self.add_node(did, deliverable_type, deliverable_name)
            for tbl in source_tables:
                self.add_edge(f"table_{tbl.lower()}", did, "used_by")
            if columns_used:
                for cref in columns_used:
                    cid = f"col_{cref.lower().replace('.', '_')}"
                    self.add_edge(cid, did, "used_by")
            self._save()
            print(f"[LINEAGE] Added deliverable: {deliverable_name}")
        except Exception as e:
            print(f"[LINEAGE] add_deliverable_lineage failed: {e}")

    # ── Traversal ──

    def _bfs(self, start, direction, depth):
        """BFS graph traversal. direction='up' or 'down'."""
        visited = set()
        queue = deque([(start, 0, [start])])
        results = []
        while queue:
            nid, dist, path = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            if nid != start:
                nd = self.data["nodes"].get(nid, {})
                results.append({
                    "node_id": nid,
                    "node_name": nd.get("name", nid),
                    "node_type": nd.get("type", "unknown"),
                    "distance": dist, "path": list(path),
                })
            if dist >= depth:
                continue
            for edge in self.data["edges"]:
                if direction == "up" and edge.get("target") == nid:
                    nxt = edge["source"]
                elif direction == "down" and edge.get("source") == nid:
                    nxt = edge["target"]
                else:
                    continue
                if nxt not in visited:
                    queue.append((nxt, dist + 1, path + [nxt]))
        return sorted(results, key=lambda x: x["distance"])

    def get_upstream(self, node_id, depth=3):
        try:
            return self._bfs(node_id, "up", depth)
        except Exception as e:
            print(f"[LINEAGE] get_upstream failed: {e}")
            return []

    def get_downstream(self, node_id, depth=3):
        try:
            return self._bfs(node_id, "down", depth)
        except Exception as e:
            print(f"[LINEAGE] get_downstream failed: {e}")
            return []

    # ── Impact Analysis ──

    def impact_analysis(self, node_id):
        try:
            downstream = self._bfs(node_id, "down", 50)
            by_type = defaultdict(int)
            affected = defaultdict(list)
            for item in downstream:
                nt = item["node_type"]
                by_type[nt] += 1
                if nt in ("dashboard", "report", "table", "column"):
                    affected[nt].append(item["node_name"])

            sev = "LOW"
            if affected.get("dashboard") or affected.get("report"):
                sev = "HIGH"
            elif affected.get("table"):
                sev = "MEDIUM"

            return {
                "impacted_node": node_id,
                "total_downstream": len(downstream),
                "by_type": dict(by_type),
                "affected_dashboards": affected.get("dashboard", []),
                "affected_reports": affected.get("report", []),
                "affected_tables": affected.get("table", []),
                "affected_columns": affected.get("column", []),
                "impact_severity": sev,
            }
        except Exception as e:
            print(f"[LINEAGE] impact_analysis failed: {e}")
            return {"impacted_node": node_id, "total_downstream": 0,
                    "impact_severity": "LOW"}

    # ── Stats ──

    def get_lineage_stats(self):
        try:
            nodes = self.data.get("nodes", {})
            edges = self.data.get("edges", [])
            by_type = defaultdict(int)
            for n in nodes.values():
                by_type[n.get("type", "unknown")] += 1
            by_rel = defaultdict(int)
            for e in edges:
                by_rel[e.get("relationship", "unknown")] += 1

            table_ids = {nid for nid, n in nodes.items()
                         if n.get("type") == "table"}
            targets = {e["target"] for e in edges}
            sources = {e["source"] for e in edges}
            has_up = table_ids & targets
            has_down = table_ids & sources
            both = has_up & has_down
            cov = (len(both) / len(table_ids) * 100
                   if table_ids else 0)
            orphans = table_ids - targets

            return {
                "total_nodes": len(nodes), "total_edges": len(edges),
                "by_node_type": dict(by_type),
                "by_relationship": dict(by_rel),
                "tables_with_lineage": len(table_ids),
                "orphan_tables": len(orphans),
                "coverage_pct": round(cov, 1),
            }
        except Exception as e:
            print(f"[LINEAGE] get_lineage_stats failed: {e}")
            return {"total_nodes": 0, "total_edges": 0}

    # ── Mermaid Diagram ──

    def generate_mermaid_diagram(self, center_node=None, depth=2):
        try:
            if center_node:
                up = self._bfs(center_node, "up", depth)
                down = self._bfs(center_node, "down", depth)
                relevant = {center_node}
                for item in up + down:
                    relevant.add(item["node_id"])
            else:
                relevant = set(self.data["nodes"].keys())

            if not relevant:
                return ""

            # Limit to 80 nodes for readability
            if len(relevant) > 80:
                relevant = set(list(relevant)[:80])

            _shapes = {
                "source_system": ("([", "])"),
                "database": ("[(", ")]"),
                "schema": ("([", "])"),
                "table": ("[", "]"),
                "column": ("(", ")"),
                "dashboard": ("{{", "}}"),
                "report": ("{{", "}}"),
            }
            _arrows = {
                "feeds": "-->", "references": "-.->",
                "used_by": "==>", "derived_from": "-->",
                "transforms_to": "-->", "joins_with": "<-->",
                "aggregates": "-->", "filters": "-->",
            }

            lines = ["graph LR"]
            for nid in relevant:
                nd = self.data["nodes"].get(nid, {})
                name = nd.get("name", nid).replace('"', "'")
                nt = nd.get("type", "table")
                l, r = _shapes.get(nt, ("[", "]"))
                safe = nid.replace("-", "_").replace("/", "_").replace(".", "_")
                lines.append(f'    {safe}{l}"{name}"{r}')

            for edge in self.data["edges"]:
                src, tgt = edge.get("source"), edge.get("target")
                if src in relevant and tgt in relevant:
                    arrow = _arrows.get(
                        edge.get("relationship", "feeds"), "-->")
                    ss = src.replace("-", "_").replace("/", "_").replace(".", "_")
                    ts = tgt.replace("-", "_").replace("/", "_").replace(".", "_")
                    lines.append(f"    {ss} {arrow} {ts}")

            return "\n".join(lines)
        except Exception as e:
            print(f"[LINEAGE] generate_mermaid_diagram failed: {e}")
            return ""

    # ── Graphviz DOT Diagram ──

    def generate_graphviz(self, center_node=None, depth=2):
        """Build a Graphviz DOT string for the lineage graph."""
        try:
            nodes = self.data.get("nodes", {})
            edges = self.data.get("edges", [])

            # Determine which nodes to include
            if center_node:
                up = self._bfs(center_node, "up", depth)
                down = self._bfs(center_node, "down", depth)
                relevant = {center_node}
                for item in up + down:
                    relevant.add(item["node_id"])
            else:
                relevant = set(nodes.keys())

            if not relevant:
                return ""

            # Collapse columns if graph is too large
            collapse_columns = len(relevant) > 80
            if collapse_columns:
                # Count columns per table for collapsed labels
                table_col_counts = {}
                for nid in list(relevant):
                    nd = nodes.get(nid, {})
                    if nd.get("type") == "column":
                        tbl = nd.get("metadata", {}).get("table", "")
                        if tbl:
                            table_col_counts[tbl] = table_col_counts.get(tbl, 0) + 1
                            relevant.discard(nid)

            def _safe_id(nid):
                return nid.replace(".", "_").replace(" ", "_").replace("-", "_")

            # Node style maps
            _node_styles = {
                "source_system": 'shape=cylinder fillcolor="#7C3AED"',
                "database":      'shape=cylinder fillcolor="#1E3A5F"',
                "schema":        'shape=folder fillcolor="#1E3A5F"',
                "table":         'shape=box style="filled,bold" fillcolor="#00D4FF" color="#00D4FF" fontcolor="#0B1120"',
                "column":        'shape=box fillcolor="#131B2E" fontsize=8',
                "dashboard":     'shape=hexagon fillcolor="#FFE600" fontcolor="#000000"',
                "report":        'shape=hexagon fillcolor="#FFE600" fontcolor="#000000"',
                "deliverable":   'shape=hexagon fillcolor="#F59E0B" fontcolor="#000000"',
            }

            # Edge style maps
            _edge_styles = {
                "feeds":      'color="#00D4FF"',
                "references": 'color="#F59E0B" style=dashed label="FK" fontcolor="#F59E0B" fontsize=8',
                "used_by":    'color="#10B981" style=bold',
            }
            _edge_default = 'color="#00D4FF"'

            # Build DOT
            lines = [
                "digraph lineage {",
                '  rankdir=LR;',
                '  bgcolor="#0B1120";',
                '  node [style=filled fontname=Arial fontsize=10 fontcolor=white];',
                '  edge [fontname=Arial];',
            ]

            # Emit nodes
            for nid in relevant:
                nd = nodes.get(nid, {})
                name = nd.get("name", nid).replace('"', '\\"')
                nt = nd.get("type", "table")
                style = _node_styles.get(nt, 'shape=box fillcolor="#131B2E"')
                sid = _safe_id(nid)
                lines.append(f'  {sid} [label="{name}" {style}];')

            # Emit collapsed column summary nodes
            if collapse_columns:
                for tbl, cnt in table_col_counts.items():
                    sid = _safe_id(f"cols_{tbl.lower()}")
                    lines.append(
                        f'  {sid} [label="{tbl} ({cnt} columns)" '
                        f'shape=box3d fillcolor="#131B2E" fontsize=8];'
                    )
                    # Edge from table to collapsed node
                    tid = _safe_id(f"table_{tbl.lower()}")
                    lines.append(f'  {tid} -> {sid} [color="#00D4FF"];')

            # Emit edges
            for edge in edges:
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if src in relevant and tgt in relevant:
                    rel = edge.get("relationship", "feeds")
                    style = _edge_styles.get(rel, _edge_default)
                    lines.append(f'  {_safe_id(src)} -> {_safe_id(tgt)} [{style}];')

            lines.append("}")
            return "\n".join(lines)
        except Exception as e:
            print(f"[LINEAGE] generate_graphviz failed: {e}")
            return ""

    # ── Export ──

    def export_lineage(self, fmt="json"):
        try:
            if fmt == "json":
                return json.dumps(self.data, indent=2, default=str)
            if fmt == "mermaid":
                return self.generate_mermaid_diagram()
            if fmt == "csv":
                header = "source,target,relationship"
                lines = [header]
                for e in self.data.get("edges", []):
                    lines.append(
                        f"{e.get('source', '')},"
                        f"{e.get('target', '')},"
                        f"{e.get('relationship', '')}")
                return "\n".join(lines)
            return json.dumps(self.data, indent=2, default=str)
        except Exception as e:
            print(f"[LINEAGE] export_lineage failed: {e}")
            return "[]"
