"""
Power BI REST API Publisher for Dr. Data.

Publishes PBIP project folders to Power BI workspaces via the Fabric REST APIs.
Uses MSAL client credentials flow (service principal) for authentication.

Auth: Service principal with client_credentials grant.
API:  Fabric REST APIs at api.fabric.microsoft.com/v1/.
Flow: 1) Auth -> 2) List workspaces -> 3) POST SemanticModel -> 4) POST Report.

Credentials loaded from ~/.env.drdata (chmod 600, never committed to git).
"""

import base64
import json
import logging
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msal
from dotenv import load_dotenv

logger = logging.getLogger("drdata-v2.powerbi_publisher")


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

FABRIC_API = "https://api.fabric.microsoft.com/v1"
PBI_API = "https://api.powerbi.com/v1.0/myorg"
FABRIC_SCOPE = ["https://api.fabric.microsoft.com/.default"]
PBI_SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
AUTHORITY_BASE = "https://login.microsoftonline.com"

# Default path to credentials file
_ENV_PATH = Path.home() / ".env.drdata"


# ------------------------------------------------------------------ #
#  Credential loading                                                  #
# ------------------------------------------------------------------ #

def _load_credentials() -> Dict[str, str]:
    """Load PBI credentials from ~/.env.drdata.

    Returns dict with tenant_id, client_id, client_secret.
    Raises ValueError if any are missing.
    """
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=True)

    tenant_id = os.getenv("PBI_TENANT_ID", "")
    client_id = os.getenv("PBI_CLIENT_ID", "")
    client_secret = os.getenv("PBI_CLIENT_SECRET", "")

    if not all([tenant_id, client_id, client_secret]):
        missing = []
        if not tenant_id:
            missing.append("PBI_TENANT_ID")
        if not client_id:
            missing.append("PBI_CLIENT_ID")
        if not client_secret:
            missing.append("PBI_CLIENT_SECRET")
        raise ValueError(
            f"Missing Power BI credentials: {', '.join(missing)}. "
            f"Set them in {_ENV_PATH} or as environment variables."
        )

    return {
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_secret": client_secret,
    }


# ------------------------------------------------------------------ #
#  Authentication                                                      #
# ------------------------------------------------------------------ #

def get_access_token(scope: List[str] = None) -> str:
    """Acquire an access token via service principal (client credentials).

    Args:
        scope: OAuth scopes. Defaults to Fabric API scope.

    Returns:
        Access token string.

    Raises:
        RuntimeError: If token acquisition fails.
    """
    creds = _load_credentials()
    scope = scope or FABRIC_SCOPE

    app = msal.ConfidentialClientApplication(
        creds["client_id"],
        authority=f"{AUTHORITY_BASE}/{creds['tenant_id']}",
        client_credential=creds["client_secret"],
    )

    result = app.acquire_token_for_client(scopes=scope)

    if "access_token" in result:
        return result["access_token"]

    error = result.get("error", "unknown_error")
    error_desc = result.get("error_description", "No description")
    raise RuntimeError(
        f"Token acquisition failed: {error} -- {error_desc}"
    )


def get_access_token_both() -> Tuple[Optional[str], Optional[str]]:
    """Try to acquire tokens for both Fabric and PBI API scopes.

    Returns (fabric_token, pbi_token). Either may be None if that scope fails.
    """
    fabric_token = None
    pbi_token = None

    try:
        fabric_token = get_access_token(FABRIC_SCOPE)
    except Exception as e:
        print(f"[PBI-AUTH] Fabric scope failed: {e}")

    try:
        pbi_token = get_access_token(PBI_SCOPE)
    except Exception as e:
        print(f"[PBI-AUTH] PBI scope failed: {e}")

    return fabric_token, pbi_token


# ------------------------------------------------------------------ #
#  Workspace discovery                                                 #
# ------------------------------------------------------------------ #

def list_workspaces(token: str) -> List[Dict]:
    """List all workspaces accessible to the authenticated principal.

    Tries Fabric API first, falls back to PBI API.

    Returns:
        List of workspace dicts: [{id, displayName/name, ...}]
    """
    # Try Fabric API first
    try:
        resp = requests.get(
            f"{FABRIC_API}/workspaces",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("value", [])
    except Exception:
        pass

    # Fallback: PBI API (uses "groups" endpoint)
    resp = requests.get(
        f"{PBI_API}/groups",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


# ------------------------------------------------------------------ #
#  PBIP Publishing (Fabric REST APIs)                                  #
# ------------------------------------------------------------------ #

def _encode_file(file_path: str) -> str:
    """Read a file and return base64-encoded content."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _collect_parts(root_dir: str,
                   file_overrides: Optional[Dict[str, bytes]] = None) -> List[Dict]:
    """Walk a directory and collect all files as Fabric API parts.

    Args:
        root_dir: Directory to walk.
        file_overrides: Optional dict of {relative_path: bytes_content} to override
            specific files with in-memory content instead of reading from disk.

    Returns:
        List of part dicts with path, payload (base64), payloadType.
    """
    overrides = file_overrides or {}
    parts = []
    root = Path(root_dir)
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            # Skip non-essential files
            if file_path.name in ("Open_Dashboard.bat", "setup.ps1",
                                   "README.txt", ".gitignore"):
                continue
            # Skip __pycache__ and similar
            if "__pycache__" in str(file_path):
                continue

            rel = str(file_path.relative_to(root)).replace("\\", "/")
            if rel in overrides:
                payload = base64.b64encode(overrides[rel]).decode("utf-8")
            else:
                payload = _encode_file(str(file_path))
            parts.append({
                "path": rel,
                "payload": payload,
                "payloadType": "InlineBase64",
            })
    return parts


def _find_item(token: str, workspace_id: str, display_name: str,
               item_type: str) -> Optional[Dict]:
    """Look up a workspace item by name and type after async creation."""
    try:
        resp = requests.get(
            f"{FABRIC_API}/workspaces/{workspace_id}/items",
            headers={"Authorization": f"Bearer {token}"},
            params={"type": item_type},
            timeout=15,
        )
        if resp.status_code == 200:
            for item in resp.json().get("value", []):
                if item.get("displayName") == display_name:
                    print(f"[PBI-PUBLISH] Found {item_type}: {item.get('id')}")
                    return item
    except Exception:
        pass
    return None


def _publish_item(
    token: str,
    workspace_id: str,
    display_name: str,
    item_type: str,
    item_dir: str,
    file_overrides: Optional[Dict[str, bytes]] = None,
) -> Dict:
    """Create a Fabric item (SemanticModel or Report) from a folder.

    Args:
        token: OAuth bearer token
        workspace_id: Target workspace UUID
        display_name: Display name in Power BI
        item_type: "SemanticModel" or "Report"
        item_dir: Path to the folder containing the item files

    Returns:
        API response dict or error dict.
    """
    parts = _collect_parts(item_dir, file_overrides=file_overrides)
    if not parts:
        return {"error": f"No files found in {item_dir}"}

    print(f"[PBI-PUBLISH] Creating {item_type} '{display_name}' "
          f"with {len(parts)} parts ({sum(len(p['payload']) for p in parts)} bytes base64)")

    resp = requests.post(
        f"{FABRIC_API}/workspaces/{workspace_id}/items",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "displayName": display_name,
            "type": item_type,
            "definition": {"parts": parts},
        },
        timeout=60,
    )

    if resp.status_code == 201:
        print(f"[PBI-PUBLISH] {item_type} created (201 sync)")
        return resp.json()
    elif resp.status_code == 202:
        location = resp.headers.get("Location", "")
        retry_after = resp.headers.get("Retry-After", "")
        print(f"[PBI-PUBLISH] {item_type} accepted (202 async), polling...")
        poll_result = poll_status(token, location)
        status = poll_result.get("status", "")
        if status in ("Succeeded", "Completed"):
            # Poll result doesn't contain the item -- look it up by name
            item = _find_item(token, workspace_id, display_name, item_type)
            if item:
                return item
            # Fallback: return poll result but mark success
            poll_result.pop("error", None)
            return poll_result
        return {"error": f"Async {item_type} failed", "detail": poll_result}
    else:
        detail = resp.text[:500]
        print(f"[PBI-PUBLISH] {item_type} failed: {resp.status_code} -- {detail}")
        return {"error": resp.status_code, "detail": detail}


def publish_pbip(
    token: str,
    workspace_id: str,
    pbip_folder_path: str,
    display_name: str,
    proof_output_dir: Optional[str] = None,
) -> Dict:
    """Publish a complete PBIP project to a Power BI workspace.

    Finds the .SemanticModel and .Report folders inside the project folder,
    publishes the semantic model FIRST (required), then the report.

    Args:
        token: OAuth bearer token (Fabric scope)
        workspace_id: Target workspace UUID
        pbip_folder_path: Path to the PBIP project folder
        display_name: Name for the published items

    Returns:
        Dict with semantic_model, report, report_url, and render proof fields.
        On error, returns dict with "error" key.
    """
    project = Path(pbip_folder_path)
    if not project.is_dir():
        return {"error": f"Not a directory: {pbip_folder_path}"}

    # Find .SemanticModel and .Report folders
    sm_dirs = list(project.glob("*.SemanticModel"))
    rpt_dirs = list(project.glob("*.Report"))

    if not sm_dirs:
        return {"error": f"No .SemanticModel folder found in {pbip_folder_path}"}
    if not rpt_dirs:
        return {"error": f"No .Report folder found in {pbip_folder_path}"}

    sm_dir = str(sm_dirs[0])
    rpt_dir = str(rpt_dirs[0])

    print(f"[PBI-PUBLISH] Publishing PBIP project: {display_name}")
    print(f"  SemanticModel: {sm_dir}")
    print(f"  Report: {rpt_dir}")

    # Step 1: Publish semantic model
    sm_result = _publish_item(token, workspace_id, display_name,
                               "SemanticModel", sm_dir)
    if sm_result.get("error"):
        return {"error": f"SemanticModel publish failed", "detail": sm_result}

    sm_id = sm_result.get("id", "")
    print(f"[PBI-PUBLISH] SemanticModel ID: {sm_id}")

    # Step 2: Rewrite definition.pbir to V4 byConnection format.
    # The Fabric Items API requires byConnection (byPath only works in
    # git-integrated workspaces). We use the XMLA connectionString format
    # which is the only format Fabric accepts for API-published reports.
    # Missing $schema or using the old pbiModelVirtualServerName format causes
    # Fabric to fail silently — pages publish but visuals show as 0.
    sm_name = sm_result.get("displayName", display_name)
    conn_str = (
        f"powerbi://api.powerbi.com/v1.0/myorg/{workspace_id}"
        f";initial catalog={sm_name}"
        f";semanticModelId={sm_id}"
    )
    pbir_override = json.dumps({
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {
            "byPath": None,
            "byConnection": {"connectionString": conn_str},
        },
    }, indent=2).encode("utf-8")
    rpt_overrides = {"definition.pbir": pbir_override}
    print(f"[PBI-PUBLISH] Rewrote definition.pbir to reference SM {sm_id} via byConnection (V4 XMLA)")

    # Step 3: Publish report
    rpt_result = _publish_item(token, workspace_id, display_name,
                                "Report", rpt_dir,
                                file_overrides=rpt_overrides)
    if rpt_result.get("error"):
        return {
            "error": "Report publish failed (SemanticModel succeeded)",
            "semantic_model": sm_result,
            "detail": rpt_result,
        }

    rpt_id = rpt_result.get("id", "")
    report_url = f"https://app.powerbi.com/groups/{workspace_id}/reports/{rpt_id}"
    embed_url = (
        f"https://app.powerbi.com/reportEmbed"
        f"?reportId={rpt_id}&groupId={workspace_id}"
    )

    print(f"[PBI-PUBLISH] Report ID: {rpt_id}")
    print(f"[PBI-PUBLISH] Report URL: {report_url}")

    # Step 4: Trigger dataset refresh so inline data is queryable
    if sm_id:
        _refresh_dataset(workspace_id, sm_id)

    render_result = {
        "render_proof": "",
        "render_proof_method": "",
        "render_proof_status": "not_attempted",
        "render_proof_error": "",
    }
    if rpt_id:
        render_result = _capture_render_proof(
            workspace_id=workspace_id,
            report_id=rpt_id,
            output_dir=proof_output_dir,
            embed_url=embed_url,
        )

    return {
        "semantic_model": sm_result,
        "report": rpt_result,
        "semantic_model_id": sm_id,
        "report_id": rpt_id,
        "report_url": report_url,
        "embed_url": embed_url,
        **render_result,
    }


def _capture_render_proof(
    workspace_id: str,
    report_id: str,
    output_dir: Optional[str] = None,
    embed_url: str = "",
) -> Dict[str, str]:
    """Capture a first-page PNG render proof after publish.

    Attempts Power BI ExportTo first. If the export call returns 403,
    falls back to a headless Playwright screenshot of the embed URL.
    """
    proof_dir = Path(output_dir or (Path.cwd() / "output" / report_id))
    proof_dir.mkdir(parents=True, exist_ok=True)
    png_path = proof_dir / "pbi_render_page1.png"

    try:
        pbi_token = get_access_token(PBI_SCOPE)
    except Exception as exc:
        logger.warning("PBI scope token failed for render proof: %s", exc)
        return {
            "render_proof": "",
            "render_proof_method": "",
            "render_proof_status": "failed",
            "render_proof_error": f"PBI token error: {str(exc)[:300]}",
        }

    export_result = _export_first_page_png(
        pbi_token=pbi_token,
        workspace_id=workspace_id,
        report_id=report_id,
        output_path=str(png_path),
    )
    if export_result.get("ok"):
        logger.info("Render proof captured via ExportTo: %s", png_path)
        return {
            "render_proof": str(png_path),
            "render_proof_method": "export_api",
            "render_proof_status": "succeeded",
            "render_proof_error": "",
        }

    export_error = export_result.get("error", "export_failed")
    if export_result.get("status_code") == 403 and embed_url:
        screenshot_result = _capture_playwright_embed_screenshot(
            embed_url=embed_url,
            output_path=str(png_path),
            access_token=pbi_token,
        )
        if screenshot_result.get("ok"):
            logger.info("Render proof captured via Playwright: %s", png_path)
            return {
                "render_proof": str(png_path),
                "render_proof_method": "playwright",
                "render_proof_status": "succeeded",
                "render_proof_error": "",
            }
        export_error = (
            f"ExportTo 403; Playwright fallback failed: "
            f"{screenshot_result.get('error', 'unknown_error')}"
        )

    logger.warning("Render proof capture failed: %s", export_error)
    return {
        "render_proof": "",
        "render_proof_method": "",
        "render_proof_status": "failed",
        "render_proof_error": str(export_error)[:500],
    }


def _export_first_page_png(
    pbi_token: str,
    workspace_id: str,
    report_id: str,
    output_path: str,
    max_wait_s: int = 180,
) -> Dict[str, object]:
    """Export the first report page to PNG via the Power BI Export API."""
    headers = {
        "Authorization": f"Bearer {pbi_token}",
        "Content-Type": "application/json",
    }

    page_name = ""
    try:
        pages = get_report_pages(pbi_token, workspace_id, report_id)
        if pages:
            page_name = pages[0].get("name", "")
    except Exception as exc:
        logger.warning("Could not resolve first page name for export: %s", exc)

    body = {"format": "PNG"}
    if page_name:
        body["powerBIReportConfiguration"] = {
            "pages": [{"pageName": page_name}]
        }

    export_url = (
        f"{PBI_API}/groups/{workspace_id}/reports/{report_id}/ExportTo"
    )
    try:
        resp = requests.post(export_url, headers=headers, json=body, timeout=60)
    except Exception as exc:
        return {"ok": False, "error": f"export request failed: {exc}"}

    if resp.status_code == 403:
        return {
            "ok": False,
            "error": f"export forbidden: {resp.text[:300]}",
            "status_code": 403,
        }
    if resp.status_code not in (200, 202):
        return {
            "ok": False,
            "error": f"export failed: {resp.status_code} {resp.text[:300]}",
            "status_code": resp.status_code,
        }

    if resp.status_code == 200:
        Path(output_path).write_bytes(resp.content)
        return {"ok": True, "path": output_path}

    export_id = resp.json().get("id", "")
    if not export_id:
        return {"ok": False, "error": "export accepted without export id"}

    status_url = (
        f"{PBI_API}/groups/{workspace_id}/reports/{report_id}/exports/{export_id}"
    )
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        time.sleep(5)
        try:
            status_resp = requests.get(status_url, headers=headers, timeout=30)
        except Exception as exc:
            return {"ok": False, "error": f"status polling failed: {exc}"}

        if status_resp.status_code not in (200, 202):
            return {
                "ok": False,
                "error": f"status failed: {status_resp.status_code} {status_resp.text[:300]}",
                "status_code": status_resp.status_code,
            }

        export_state = status_resp.json()
        status = export_state.get("status", "")
        if status == "Succeeded":
            file_url = f"{status_url}/file"
            file_resp = requests.get(file_url, headers=headers, timeout=60)
            if file_resp.status_code != 200:
                return {
                    "ok": False,
                    "error": f"file download failed: {file_resp.status_code} {file_resp.text[:300]}",
                    "status_code": file_resp.status_code,
                }
            Path(output_path).write_bytes(file_resp.content)
            return {"ok": True, "path": output_path}
        if status == "Failed":
            return {
                "ok": False,
                "error": f"export failed: {json.dumps(export_state)[:300]}",
            }

    return {"ok": False, "error": f"export timed out after {max_wait_s}s"}


def _capture_playwright_embed_screenshot(
    embed_url: str,
    output_path: str,
    access_token: str = "",
) -> Dict[str, object]:
    """Capture a screenshot of the Power BI embed URL."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        return {"ok": False, "error": f"Playwright unavailable: {exc}"}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1600, "height": 900},
                ignore_https_errors=True,
            )
            if access_token:
                context.set_extra_http_headers(
                    {"Authorization": f"Bearer {access_token}"}
                )
            page = context.new_page()
            page.goto(embed_url, wait_until="networkidle", timeout=120000)
            page.wait_for_timeout(8000)
            page.screenshot(path=output_path, full_page=False)
            browser.close()
        return {"ok": True, "path": output_path}
    except Exception as exc:
        return {"ok": False, "error": f"screenshot failed: {exc}"}


# ------------------------------------------------------------------ #
#  Dataset refresh (required for inline #table data)                   #
# ------------------------------------------------------------------ #

def _refresh_dataset(workspace_id: str, dataset_id: str) -> None:
    """Trigger a dataset refresh so inline data becomes queryable.

    Uses the PBI API scope (not Fabric) since the refresh endpoint lives
    on api.powerbi.com. Fire-and-forget: logs result but does not block.
    """
    try:
        pbi_token = get_access_token(PBI_SCOPE)
        resp = requests.post(
            f"{PBI_API}/groups/{workspace_id}/datasets/{dataset_id}/refreshes",
            headers={
                "Authorization": f"Bearer {pbi_token}",
                "Content-Type": "application/json",
            },
            json={"type": "Full"},
            timeout=30,
        )
        if resp.status_code == 202:
            print(f"[PBI-REFRESH] Dataset refresh triggered for {dataset_id}")
            # Poll briefly for completion (inline data refreshes are fast)
            for _ in range(10):
                time.sleep(3)
                r = requests.get(
                    f"{PBI_API}/groups/{workspace_id}/datasets/{dataset_id}/refreshes?$top=1",
                    headers={"Authorization": f"Bearer {pbi_token}"},
                    timeout=15,
                )
                if r.status_code == 200:
                    refs = r.json().get("value", [])
                    if refs and refs[0].get("status") in ("Completed", "Failed"):
                        status = refs[0]["status"]
                        print(f"[PBI-REFRESH] Refresh {status}")
                        if status == "Failed":
                            err = refs[0].get("serviceExceptionJson", "")
                            print(f"[PBI-REFRESH] Error: {err[:200]}")
                        return
        else:
            print(f"[PBI-REFRESH] Refresh request failed: {resp.status_code} -- {resp.text[:200]}")
    except Exception as e:
        print(f"[PBI-REFRESH] Non-fatal refresh error: {e}")


# ------------------------------------------------------------------ #
#  Long-running operation polling                                      #
# ------------------------------------------------------------------ #

def poll_status(token: str, location_url: str, max_wait: int = 120) -> Dict:
    """Poll a long-running Fabric operation until it reaches a terminal state.

    Args:
        token: OAuth bearer token
        location_url: The Location header URL from the 202 response
        max_wait: Maximum seconds to wait

    Returns:
        Operation result dict. Contains "status" key.
        "Succeeded"/"Completed" = success. "Failed" = error with details.
    """
    if not location_url:
        return {"error": "No Location URL provided for polling"}

    headers = {"Authorization": f"Bearer {token}"}
    elapsed = 0
    interval = 2

    while elapsed < max_wait:
        time.sleep(interval)
        elapsed += interval

        try:
            resp = requests.get(location_url, headers=headers, timeout=15)
        except Exception as e:
            return {"error": f"Poll request failed: {e}"}

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "")
            if status in ("Succeeded", "Completed"):
                print(f"[PBI-POLL] Operation succeeded after {elapsed}s")
                return data
            elif status == "Failed":
                print(f"[PBI-POLL] Operation failed after {elapsed}s")
                return {"error": "Operation failed", "detail": data}
            # Still running -- continue
        elif resp.status_code == 202:
            pass  # Still in progress
        else:
            return {"error": resp.status_code, "detail": resp.text[:500]}

        interval = min(interval * 1.5, 10)

    return {"error": f"Timeout after {max_wait}s waiting for operation"}


# ------------------------------------------------------------------ #
#  Convenience: full publish from path                                 #
# ------------------------------------------------------------------ #

def publish_from_output(pbip_folder_path: str, display_name: str,
                        workspace_id: str = None) -> Dict:
    """High-level: authenticate, discover workspace, publish.

    If workspace_id is not provided, uses the first available workspace.

    Returns:
        Full result dict with report_url on success, or error details.
    """
    try:
        token = get_access_token()
    except Exception as e:
        return {"error": f"Authentication failed: {e}"}

    if not workspace_id:
        try:
            workspaces = list_workspaces(token)
            if not workspaces:
                return {"error": "No workspaces found. "
                        "Ensure the service principal has been added to at least one workspace."}
            workspace_id = workspaces[0]["id"]
            ws_name = workspaces[0].get("displayName", workspaces[0].get("name", "?"))
            print(f"[PBI-PUBLISH] Auto-selected workspace: {ws_name} ({workspace_id})")
        except Exception as e:
            return {"error": f"Workspace discovery failed: {e}"}

    return publish_pbip(token, workspace_id, pbip_folder_path, display_name)

# ------------------------------------------------------------------ #
#  REST API Read-Back Functions                                        #
# ------------------------------------------------------------------ #

def get_report_pages(token: str, workspace_id: str, report_id: str) -> List[Dict]:
    """Get all pages from a published report.

    Returns list of page dicts: [{name, displayName, order}]
    """
    resp = requests.get(
        f"{PBI_API}/groups/{workspace_id}/reports/{report_id}/pages",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


def get_report_visuals(token: str, workspace_id: str, report_id: str,
                       page_name: str) -> List[Dict]:
    """Get all visuals on a specific report page.

    Returns list of visual dicts: [{name, title, type, layout}]
    """
    resp = requests.get(
        f"{PBI_API}/groups/{workspace_id}/reports/{report_id}/pages/{page_name}/visuals",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json().get("value", [])
    # Some report types don't support this endpoint
    return []


def get_dataset_id_from_report(token: str, workspace_id: str,
                               report_id: str) -> Optional[str]:
    """Get the dataset ID linked to a report.

    Returns dataset ID string or None.
    """
    resp = requests.get(
        f"{PBI_API}/groups/{workspace_id}/reports/{report_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json().get("datasetId")
    return None


def get_dataset_tables(token: str, workspace_id: str,
                       dataset_id: str) -> List[Dict]:
    """Get tables and columns from a dataset.

    Returns list of table dicts: [{name, columns: [{name, dataType}]}]
    """
    resp = requests.get(
        f"{PBI_API}/groups/{workspace_id}/datasets/{dataset_id}/tables",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 200:
        return resp.json().get("value", [])
    return []


def execute_dax_query(token: str, workspace_id: str, dataset_id: str,
                      dax: str) -> Dict:
    """Execute a DAX query against a published dataset.

    Args:
        dax: DAX query string, e.g. "EVALUATE ROW(\"count\", COUNTROWS(Data))"

    Returns dict with results or error.
    """
    resp = requests.post(
        f"{PBI_API}/groups/{workspace_id}/datasets/{dataset_id}/executeQueries",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "queries": [{"query": dax}],
            "serializerSettings": {"includeNulls": True},
        },
        timeout=30,
    )
    if resp.status_code == 200:
        data = resp.json()
        results = data.get("results", [])
        if results:
            return {
                "tables": results[0].get("tables", []),
                "rows": results[0].get("tables", [{}])[0].get("rows", []) if results[0].get("tables") else [],
            }
        return {"tables": [], "rows": []}
    return {"error": resp.status_code, "detail": resp.text[:500]}


def delete_item(token: str, workspace_id: str, item_id: str) -> bool:
    """Delete any Fabric item (report, semantic model, etc).

    Returns True on success.
    """
    resp = requests.delete(
        f"{FABRIC_API}/workspaces/{workspace_id}/items/{item_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code in (200, 204):
        print(f"[PBI-DELETE] Deleted item {item_id}")
        return True
    print(f"[PBI-DELETE] Failed to delete {item_id}: {resp.status_code} {resp.text[:200]}")
    return False


def delete_report(token: str, workspace_id: str, report_id: str) -> bool:
    """Delete a report."""
    return delete_item(token, workspace_id, report_id)


def delete_dataset(token: str, workspace_id: str, dataset_id: str) -> bool:
    """Delete a dataset/semantic model."""
    return delete_item(token, workspace_id, dataset_id)
