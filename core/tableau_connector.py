"""
Tableau Cloud Connector for Dr. Data.

Connects to Tableau Cloud via Personal Access Token (PAT) using the
tableauserverclient (TSC) library. Downloads workbooks, view images,
crosstab data, and metadata for the cloud bridge pipeline.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional

import tableauserverclient as TSC
from dotenv import load_dotenv

logger = logging.getLogger("drdata-tableau")

_ENV_PATH = Path.home() / ".env.drdata"


class TableauCloudConnector:
    """Interface to Tableau Cloud REST API via TSC."""

    def __init__(self):
        if _ENV_PATH.exists():
            load_dotenv(_ENV_PATH, override=True)

        self.pat_name = os.getenv("TABLEAU_PAT_NAME", "")
        self.pat_secret = os.getenv("TABLEAU_PAT_SECRET", "")
        self.server_url = (
            os.getenv("TABLEAU_CLOUD_SERVER", "")
            or os.getenv("TABLEAU_SERVER_URL", "")
        )
        self.site_name = (
            os.getenv("TABLEAU_CLOUD_SITE", "")
            or os.getenv("TABLEAU_SITE_ID", "")
        )

        if not all([self.pat_name, self.pat_secret, self.server_url]):
            missing = []
            if not self.pat_name:
                missing.append("TABLEAU_PAT_NAME")
            if not self.pat_secret:
                missing.append("TABLEAU_PAT_SECRET")
            if not self.server_url:
                missing.append("TABLEAU_CLOUD_SERVER")
            raise ValueError(
                f"Missing Tableau credentials: {', '.join(missing)}. "
                f"Set them in {_ENV_PATH}."
            )

        # Normalize server URL
        if not self.server_url.startswith("http"):
            self.server_url = f"https://{self.server_url}"

        self._auth = TSC.PersonalAccessTokenAuth(
            self.pat_name, self.pat_secret, site_id=self.site_name
        )
        self._server = TSC.Server(self.server_url, use_server_version=True)
        self._server.add_http_options({"verify": True})

    def _sign_in(self):
        """Sign in and return the server context manager."""
        return self._server.auth.sign_in(self._auth)

    def list_workbooks(self) -> List[Dict]:
        """List all workbooks accessible to the authenticated user.

        Returns list of dicts: [{id, name, project_name, updated_at, views}]
        """
        with self._server.auth.sign_in(self._auth):
            all_workbooks = []
            for wb in TSC.Pager(self._server.workbooks):
                # Populate views
                self._server.workbooks.populate_views(wb)
                views = [
                    {"id": v.id, "name": v.name}
                    for v in (wb.views or [])
                ]
                all_workbooks.append({
                    "id": wb.id,
                    "name": wb.name,
                    "project_name": wb.project_name or "",
                    "updated_at": str(wb.updated_at) if wb.updated_at else "",
                    "content_url": wb.content_url or "",
                    "views": views,
                })
            return all_workbooks

    def download_workbook(self, workbook_id: str, dest_dir: str = None) -> str:
        """Download a workbook as .twbx.

        Args:
            workbook_id: Tableau workbook ID
            dest_dir: Directory to save into (default: temp dir)

        Returns:
            Path to downloaded .twbx file.
        """
        if not dest_dir:
            dest_dir = tempfile.mkdtemp(prefix="drdata_tc_")
        Path(dest_dir).mkdir(parents=True, exist_ok=True)

        with self._server.auth.sign_in(self._auth):
            path = self._server.workbooks.download(
                workbook_id,
                filepath=dest_dir,
                include_extract=True,
            )
            logger.info(f"Downloaded workbook {workbook_id} -> {path}")
            return path

    def get_view_images(self, workbook_id: str, dest_dir: str) -> Dict[str, str]:
        """Download PNG screenshot for each view in a workbook.

        Args:
            workbook_id: Tableau workbook ID
            dest_dir: Directory to save images

        Returns:
            Dict mapping view_name -> image_path
        """
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        images = {}

        with self._server.auth.sign_in(self._auth):
            wb = self._server.workbooks.get_by_id(workbook_id)
            self._server.workbooks.populate_views(wb)

            for view in (wb.views or []):
                try:
                    self._server.views.populate_image(view)
                    if view.image:
                        safe_name = "".join(
                            c if c.isalnum() or c in "._- " else "_"
                            for c in view.name
                        )
                        img_path = os.path.join(dest_dir, f"{safe_name}.png")
                        with open(img_path, "wb") as f:
                            f.write(view.image)
                        images[view.name] = img_path
                        logger.info(f"View image: {view.name} -> {img_path}")
                except Exception as e:
                    logger.warning(f"Failed to get image for view '{view.name}': {e}")

        return images

    def get_view_data(self, workbook_id: str, dest_dir: str) -> Dict[str, Optional[str]]:
        """Download crosstab CSV data for each view.

        Args:
            workbook_id: Tableau workbook ID
            dest_dir: Directory to save data files

        Returns:
            Dict mapping view_name -> csv_path (None if crosstab not available)
        """
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        data_files = {}

        with self._server.auth.sign_in(self._auth):
            wb = self._server.workbooks.get_by_id(workbook_id)
            self._server.workbooks.populate_views(wb)

            for view in (wb.views or []):
                try:
                    self._server.views.populate_csv(view)
                    if view.csv:
                        safe_name = "".join(
                            c if c.isalnum() or c in "._- " else "_"
                            for c in view.name
                        )
                        csv_path = os.path.join(dest_dir, f"{safe_name}.csv")
                        with open(csv_path, "wb") as f:
                            # view.csv is an iterator of bytes
                            for chunk in view.csv:
                                f.write(chunk)
                        data_files[view.name] = csv_path
                        logger.info(f"View data: {view.name} -> {csv_path}")
                    else:
                        data_files[view.name] = None
                except Exception as e:
                    logger.warning(f"Crosstab not available for '{view.name}': {e}")
                    data_files[view.name] = None

        return data_files

    def get_metadata(self, workbook_name: str) -> Dict:
        """Query the Metadata API for calculated fields and field bindings.

        Uses GraphQL endpoint on Tableau Cloud.

        Args:
            workbook_name: Name of the workbook to query

        Returns:
            Dict with calculated_fields, fields, and sheet_bindings.
        """
        import requests

        with self._server.auth.sign_in(self._auth):
            token = self._server.auth_token
            metadata_url = f"{self.server_url}/api/metadata/graphql"

            query = """
            query WorkbookMetadata($name: String!) {
              workbooks(filter: {name: $name}) {
                name
                sheets {
                  name
                  sheetFieldInstances {
                    name
                    datasourceField {
                      name
                      dataType
                      isCalculated
                      formula
                    }
                  }
                }
                embeddedDatasources {
                  name
                  fields {
                    name
                    dataType
                    isCalculated
                    formula
                  }
                }
              }
            }
            """

            try:
                resp = requests.post(
                    metadata_url,
                    headers={
                        "X-Tableau-Auth": token,
                        "Content-Type": "application/json",
                    },
                    json={"query": query, "variables": {"name": workbook_name}},
                    timeout=30,
                )

                if resp.status_code != 200:
                    logger.warning(
                        f"Metadata API returned {resp.status_code}: {resp.text[:200]}"
                    )
                    return {"calculated_fields": [], "fields": [], "sheets": []}

                data = resp.json().get("data", {})
                workbooks = data.get("workbooks", [])
                if not workbooks:
                    return {"calculated_fields": [], "fields": [], "sheets": []}

                wb = workbooks[0]
                calc_fields = []
                all_fields = []

                for ds in wb.get("embeddedDatasources", []):
                    for field in ds.get("fields", []):
                        entry = {
                            "name": field.get("name", ""),
                            "data_type": field.get("dataType", ""),
                            "datasource": ds.get("name", ""),
                        }
                        all_fields.append(entry)
                        if field.get("isCalculated") and field.get("formula"):
                            calc_fields.append({
                                **entry,
                                "formula": field["formula"],
                            })

                sheets = []
                for sheet in wb.get("sheets", []):
                    field_names = [
                        fi.get("name", "")
                        for fi in sheet.get("sheetFieldInstances", [])
                    ]
                    sheets.append({
                        "name": sheet.get("name", ""),
                        "fields": field_names,
                    })

                return {
                    "calculated_fields": calc_fields,
                    "fields": all_fields,
                    "sheets": sheets,
                }

            except Exception as e:
                logger.warning(f"Metadata API error: {e}")
                return {"calculated_fields": [], "fields": [], "sheets": []}
