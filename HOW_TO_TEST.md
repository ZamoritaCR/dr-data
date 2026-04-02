# How to Test Dr. Data -- Step-by-Step

URL: https://drdata.theartofthepossible.io

## Before you start

Hard refresh the browser: **Ctrl+Shift+R** (clears session state).


## Test 1: Fresh Data File (CSV/Excel) -> Power BI

**What you need:** Any CSV or Excel file with data.

**Steps:**
1. Open drdata.theartofthepossible.io
2. In the sidebar, upload your file (drag or click)
3. Wait for the sidebar to show "loaded" (you'll see a data preview in the workspace)
4. In the chat, type:

```
build me a power bi
```

**What you should see:**
- The workspace panel shows a progress status box (expanded)
- Steps appear one by one:
  - "Profiling data: X rows x Y columns..."
  - "Data profiled: N dimensions, M measures..."
  - "Requirements contract built: 1 page, N visuals..."
  - "Claude Opus 4 is designing your dashboard..."
  - "GPT-4 is generating Power BI visual containers..."
  - "Writing Power BI project files..."
  - "PBIP project built: N files..."
  - "QA manifest generated: N checks..."
  - "Done -- 1 deliverable(s)"
- A ZIP file appears in the workspace downloads section
- Dr. Data narrates what it built in the chat

**How to verify the output:**
1. Download the ZIP
2. Extract it to a folder on a Windows machine with Power BI Desktop
3. Double-click `Open_Dashboard.bat`
4. Power BI opens with the generated dashboard


## Test 2: Tableau File (.twbx) with Embedded Data

**What you need:** The `US - Superstore.twbx` file (already on this machine at `~/projects/dr-data/`).

**Steps:**
1. Hard refresh the browser
2. Upload `US - Superstore.twbx` AND `(US) Superstore.xlsx` together
3. In the chat, type:

```
convert this to a 3 page power bi report with slicers for Region and Category
```

**What you should see:**
- Progress shows Tableau structure mapping (21 worksheets, 6 dashboards)
- Migration intent built
- Claude Opus translates Tableau to PBI
- ZIP generated

**Verification:** The generated PBI should have 3 pages with Region and Category slicers.


## Test 3: Tableau File (.twbx) WITHOUT Data (Synthetic)

**What you need:** The `#WOW2025 W49.twbx` file (or any .twbx where the data is .hyper).

**Steps:**
1. Hard refresh
2. Upload ONLY the .twbx file (no data file)
3. In the chat, type:

```
build me a power bi
```

**What you should see:**
- Progress shows: "No extractable data found in the Tableau file (uses .hyper format). Generating synthetic data..."
- "Reading Tableau structure: N worksheets, M calculated fields"
- "Synthetic data generated: 2000 rows x N columns"
- Then the normal PBI generation flow continues
- ZIP is produced

**Verification:** Open the ZIP -- it contains a PBIP project with synthetic data. The visuals will have data, but the numbers are synthetic (placeholder).


## Test 4: Specific Layout Instructions

**Steps:**
1. Hard refresh
2. Upload any CSV/Excel file
3. In the chat, type exactly:

```
Build a single page Power BI report called "Dash Overall Transactions" with top-right last refresh text, 3 slicers in the top row, a horizontal tab-like selector for category, a section label "Total Monthly", 3 monthly visuals stacked vertically: 1) TV MTD as stacked column with MonthYear on X-axis and legend by transaction subtype, 2) TC MTD as stacked column with MonthYear on X-axis and legend by transaction subtype, 3) UU MTD as clustered column with MonthYear on X-axis. Enable cross-filtering. Use consistent legend colors across the stacked charts.
```

**What you should see:**
- Contract built with 1 page, 5+ visuals, 3 slicers
- All elements appear in the progress updates
- The generated PBI has exactly 1 page named "Dash Overall Transactions"

**Verification:**
- Download the QA audit report (qa_report_*.md) -- it lists every visual mapping, DAX measure, and a checklist of things to verify


## Test 5: HTML Dashboard (no Power BI needed)

**Steps:**
1. Upload a data file
2. In the chat, type:

```
build me an interactive dashboard
```

**What you should see:**
- An HTML file is generated
- Download it and open in any browser
- Interactive charts with filters, KPIs, drill-down


## Test 6: Multi-Format Export

**Steps:**
1. Upload a data file
2. Type:

```
build me a power bi, interactive dashboard, and pdf report
```

**What you should see:**
- 3+ files generated (ZIP + HTML + PDF)
- QA manifest included


## Test 7: Ask About the Data

**Steps:**
1. Upload any data file
2. Instead of asking for a build, just talk to Dr. Data:

```
what do you see in this data?
```

**What you should see:**
- Dr. Data analyzes the data proactively
- Finds insights, patterns, outliers
- Suggests what to build


## Checking the QA Audit Report

Every PBI generation creates two audit files in the output directory:
- `qa_manifest_*.json` -- machine-readable
- `qa_report_*.md` -- human-readable

The Markdown report contains:
- Source assets used (files, row counts)
- Visual mapping decisions (Tableau type -> PBI type, with rationale)
- Generated DAX measures (with code blocks)
- Mapping assumptions
- Unsupported features
- QA checklist with severity levels

**Key things to check in the checklist:**
- Items marked `must_verify` / `FAIL` need attention
- Items marked `should_verify` / `WARN` are worth checking
- Items marked `PASS` were auto-validated


## Running the Automated Tests

From the ROG machine terminal:

```bash
source ~/taop-agents-env/bin/activate
cd ~/taop-repos/dr-data
python -m pytest tests/ -v
```

Expected: 275 passed, 0 failed.


## If Something Goes Wrong

**Site not loading:**
```bash
cd ~/taop-repos/dr-data && source ~/taop-agents-env/bin/activate
nohup streamlit run app/streamlit_app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true > /tmp/drdata_streamlit.log 2>&1 &
```

**Rolling back:**
```bash
cp ~/taop-repos/dr-data/_backup_20260402_183505/* back to original locations
# Then restart Streamlit
```

**Checking logs:**
```bash
tail -50 /tmp/drdata_streamlit.log
```
