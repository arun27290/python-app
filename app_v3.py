"""
ITIS Incident Management Analyzer  v3
======================================
Run:  python app.py
Open: http://localhost:5050

New in v3:
- MTTR uses SubmitDate when available (LastResolvedDate - SubmitDate), else ReportedDate
- Overview: MTTR>200h alerts, HOP count>5 alerts, Incident Aging>30days alerts
- Overview: removed Open/Active, Total Hours, Avg Transfers, Status Breakdown
- Team tab: grouped by AssignedGroup, each group shows its assignees as rows
- CI tab: CI>3 table now includes AssignedGroup breakdown per CI
- KB Article analysis tab (optional - only shown when Request_Type01 col exists)
"""

import io, json, warnings, logging, re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, jsonify

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large — maximum 50 MB."}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    log.exception("Unhandled exception")
    return jsonify({"error": f"Server error: {str(e)}"}), 500

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIASES
# ─────────────────────────────────────────────────────────────────────────────
ALIASES = {
    "Incident_Number": "Incident_Number", "IncidentNumber": "Incident_Number",
    "Incident_Nember": "Incident_Number", "IncidentID": "Incident_Number",
    "Incident ID": "Incident_Number", "Incident Number": "Incident_Number",

    "ReportedDate": "ReportedDate", "Reported Date": "ReportedDate",
    "LastResolvedDate": "LastResolvedDate", "Last Resolved Date": "LastResolvedDate",
    "ResolvedDate": "LastResolvedDate",
    "SubmitDate": "SubmitDate", "Submit Date": "SubmitDate",

    "Summary": "Summary",
    "Service_Type": "Service_Type", "ServiceType": "Service_Type",
    "HPD_CI": "HPD_CI", "HPDCI": "HPD_CI", "CI": "HPD_CI",
    "SLAStatus": "SLAStatus", "SLA_Status": "SLAStatus", "SLA Status": "SLAStatus",
    "Priority": "Priority",
    "AssignedGroup": "AssignedGroup", "Assigned Group": "AssignedGroup",
    "Assignee": "Assignee",
    "Assigned_Support_Organisation": "Assigned_Support_Organisation",
    "AssignedSupportOrganisation": "Assigned_Support_Organisation",
    "Organisation": "Assigned_Support_Organisation",
    "Organization": "Assigned_Support_Organisation",
    "Assigned_Support_Company": "Assigned_Support_Company",
    "Company": "Assigned_Support_Company",
    "Status": "Status",
    "Group_Transfers": "Group_Transfers", "GroupTransfers": "Group_Transfers",
    "Group Transfers": "Group_Transfers",
    # KB columns
    "Request_Type01": "Request_Type01", "RequestType01": "Request_Type01",
    "Request Type01": "Request_Type01",
    "Request_Type_Description": "Request_Type_Description",
    "RequestTypeDescription": "Request_Type_Description",
    "Request Type Description": "Request_Type_Description",
}

EXCEL_EPOCH = datetime(1899, 12, 30)

def excel_serial_to_dt(val):
    try:
        f = float(val)
        if f > 0:
            return EXCEL_EPOCH + timedelta(days=f)
    except (TypeError, ValueError):
        pass
    return pd.NaT

def parse_date_col(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.apply(excel_serial_to_dt)
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.5:
        parsed = series.apply(lambda v: excel_serial_to_dt(v) if pd.notna(v) else pd.NaT)
    return parsed

def normalise_columns(df):
    df.columns = [c.strip() for c in df.columns]
    rename_map = {col: ALIASES[col] for col in df.columns if col in ALIASES}
    return df.rename(columns=rename_map)

def priority_sort_key(p):
    return {"P1 - Critical": 0, "Critical": 0,
            "P2 - High": 1, "High": 1,
            "P3 - Medium": 2, "Medium": 2,
            "P4 - Low": 3, "Low": 3}.get(p, 99)

def safe_str(v):
    """Return string or em-dash for None/NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return str(v)

def extract_kb(text):
    """Extract KB article ID like KB0001234 from a description string."""
    if pd.isna(text):
        return None
    m = re.search(r'\bKB\d{4,10}\b', str(text), re.IGNORECASE)
    return m.group(0).upper() if m else None

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def analyse(df_raw):
    df = normalise_columns(df_raw.copy())

    # ── Parse dates ───────────────────────────────────────────────────────────
    for col in ["ReportedDate", "LastResolvedDate", "SubmitDate"]:
        if col in df.columns:
            df[col] = parse_date_col(df[col])

    has_reported  = "ReportedDate"     in df.columns
    has_resolved  = "LastResolvedDate" in df.columns
    has_submitted = "SubmitDate"       in df.columns
    has_status    = "Status"           in df.columns
    has_sla       = "SLAStatus"        in df.columns
    has_priority  = "Priority"         in df.columns
    has_group     = "AssignedGroup"    in df.columns
    has_assignee  = "Assignee"         in df.columns
    has_org       = "Assigned_Support_Organisation" in df.columns
    has_ci        = "HPD_CI"           in df.columns
    has_svc       = "Service_Type"     in df.columns
    has_xfer      = "Group_Transfers"  in df.columns
    has_kb_type   = "Request_Type01"   in df.columns
    has_kb_desc   = "Request_Type_Description" in df.columns
    inc_col       = "Incident_Number"  if "Incident_Number" in df.columns else df.columns[0]

    # ── Time-derived fields ───────────────────────────────────────────────────
    if has_reported:
        df["Month"]     = df["ReportedDate"].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df["ReportedDate"].dt.day_name()

    # MTTR: prefer SubmitDate as start, fallback to ReportedDate
    if has_resolved:
        start_col = "SubmitDate" if has_submitted else ("ReportedDate" if has_reported else None)
        if start_col:
            df["MTTR_Hours"] = (
                (df["LastResolvedDate"] - df[start_col])
                .dt.total_seconds() / 3600
            ).round(2)
            df.loc[df["MTTR_Hours"] < 0, "MTTR_Hours"] = np.nan
            df["MTTR_source"] = f"LastResolvedDate − {start_col}"
        else:
            df["MTTR_Hours"] = np.nan
            df["MTTR_source"] = "N/A"
    else:
        df["MTTR_Hours"] = np.nan

    # Incident age in days: LastResolvedDate - SubmitDate (or ReportedDate)
    if has_resolved:
        age_start = "SubmitDate" if has_submitted else ("ReportedDate" if has_reported else None)
        if age_start:
            df["AgeDays"] = (
                (df["LastResolvedDate"] - df[age_start])
                .dt.total_seconds() / 86400
            ).round(1)
            df.loc[df["AgeDays"] < 0, "AgeDays"] = np.nan

    if has_xfer:
        df["Group_Transfers"] = pd.to_numeric(df["Group_Transfers"], errors="coerce")

    # ── Resolved subset ───────────────────────────────────────────────────────
    open_statuses   = ["open","in progress","pending","assigned","work in progress","wip"]
    closed_statuses = ["resolved","closed","completed"]

    if has_status:
        resolved = df[df["Status"].str.strip().str.lower().isin(closed_statuses)].copy()
        open_ct  = int(df[df["Status"].str.strip().str.lower().isin(open_statuses)].shape[0])
        closed_ct = int(resolved.shape[0])
    elif has_resolved:
        resolved  = df[df["LastResolvedDate"].notna()].copy()
        open_ct   = int(df[df["LastResolvedDate"].isna()].shape[0])
        closed_ct = len(resolved)
    else:
        resolved  = df.copy()
        open_ct   = 0
        closed_ct = len(df)

    total = len(df)

    # ── Date range ────────────────────────────────────────────────────────────
    if has_reported and df["ReportedDate"].notna().any():
        date_min = df["ReportedDate"].min().strftime("%d-%b-%Y")
        date_max = df["ReportedDate"].max().strftime("%d-%b-%Y")
    else:
        date_min = date_max = "N/A"

    # ── SLA ───────────────────────────────────────────────────────────────────
    met_values = ["met","within sla","sla met","ok","yes"]
    if has_sla:
        sla_met_ct = int(df["SLAStatus"].str.strip().str.lower().isin(met_values).sum())
        sla_pct    = round(sla_met_ct / total * 100, 1) if total else 0
    else:
        sla_met_ct = sla_pct = 0

    # ── MTTR KPI ──────────────────────────────────────────────────────────────
    mttr_source = df["MTTR_source"].iloc[0] if "MTTR_source" in df.columns and len(df) else "N/A"
    if "MTTR_Hours" in df.columns and df["MTTR_Hours"].notna().any():
        mttr = round(float(df["MTTR_Hours"].mean()), 1)
    else:
        mttr = 0

    # P1 count
    if has_priority:
        p1_ct = int(df[df["Priority"].str.lower().str.contains("critical|p1", na=False)].shape[0])
    else:
        p1_ct = 0

    # ── ALERT LISTS ───────────────────────────────────────────────────────────
    # 1. MTTR > 200 hrs
    alert_cols = [inc_col, "AssignedGroup", "MTTR_Hours"]
    if "MTTR_Hours" in df.columns:
        mttr_alert = df[df["MTTR_Hours"] > 200][
            [c for c in alert_cols if c in df.columns]
        ].copy()
        if has_priority and "Priority" in df.columns:
            mttr_alert["Priority"] = df.loc[mttr_alert.index, "Priority"]
        mttr_alert = mttr_alert.sort_values("MTTR_Hours", ascending=False)
        mttr_alert["MTTR_Hours"] = mttr_alert["MTTR_Hours"].round(1)
        mttr_alert_rows = mttr_alert.fillna("—").to_dict("records")
    else:
        mttr_alert_rows = []

    # 2. HOP count (Group_Transfers) > 5
    if has_xfer:
        hop_cols = [inc_col, "AssignedGroup", "Group_Transfers"]
        if has_priority: hop_cols.append("Priority")
        hop_alert = df[df["Group_Transfers"] > 5][
            [c for c in hop_cols if c in df.columns]
        ].copy().sort_values("Group_Transfers", ascending=False)
        hop_alert_rows = hop_alert.fillna("—").to_dict("records")
    else:
        hop_alert_rows = []

    # 3. Incident aging > 30 days
    if "AgeDays" in df.columns:
        age_cols = [inc_col, "AssignedGroup", "AgeDays"]
        if has_priority: age_cols.append("Priority")
        if has_status:   age_cols.append("Status")
        age_alert = df[df["AgeDays"] > 30][
            [c for c in age_cols if c in df.columns]
        ].copy().sort_values("AgeDays", ascending=False)
        age_alert["AgeDays"] = age_alert["AgeDays"].round(1)
        age_alert_rows = age_alert.fillna("—").to_dict("records")
    else:
        age_alert_rows = []

    # ── Charts ────────────────────────────────────────────────────────────────
    # Monthly volume
    if has_reported and "Month" in df.columns:
        vol = df.groupby("Month").size().reset_index(name="count").sort_values("Month")
        vol_labels, vol_data = vol["Month"].tolist(), vol["count"].tolist()
    else:
        vol_labels = vol_data = []

    # Priority
    if has_priority:
        pri = df["Priority"].value_counts().reset_index()
        pri.columns = ["Priority","count"]
        pri["s"] = pri["Priority"].apply(priority_sort_key)
        pri = pri.sort_values("s").drop("s", axis=1)
        pri_labels, pri_data = pri["Priority"].tolist(), pri["count"].tolist()
    else:
        pri_labels = pri_data = []

    # SLA breakdown
    if has_sla:
        sla_s = df["SLAStatus"].value_counts().reset_index()
        sla_s.columns = ["SLAStatus","count"]
        sla_labels, sla_data = sla_s["SLAStatus"].tolist(), sla_s["count"].tolist()
    else:
        sla_labels = sla_data = []

    # SLA % by priority
    if has_priority and has_sla:
        def _sla_pct(x):
            return round(x["SLAStatus"].str.strip().str.lower().isin(met_values).sum() / len(x) * 100, 1)
        sp = df.groupby("Priority").apply(_sla_pct).reset_index(name="pct")
        sp["s"] = sp["Priority"].apply(priority_sort_key)
        sp = sp.sort_values("s").drop("s", axis=1)
        sla_pri_labels, sla_pri_vals = sp["Priority"].tolist(), sp["pct"].tolist()
    else:
        sla_pri_labels = sla_pri_vals = []

    # SLA trend
    if has_sla and has_reported and "Month" in df.columns:
        def _sla_mo(x):
            return round(x["SLAStatus"].str.strip().str.lower().isin(met_values).sum() / len(x) * 100, 1)
        st = df.groupby("Month").apply(_sla_mo).reset_index(name="pct").sort_values("Month")
        sla_tr_labels, sla_tr_vals = st["Month"].tolist(), st["pct"].tolist()
    else:
        sla_tr_labels = sla_tr_vals = []

    # Group bar
    if has_group:
        grp = df.groupby("AssignedGroup").size().reset_index(name="count").sort_values("count", ascending=False)
        grp_labels, grp_data = grp["AssignedGroup"].tolist(), grp["count"].tolist()
    else:
        grp_labels = grp_data = []

    # MTTR by group
    if has_group and "MTTR_Hours" in df.columns:
        mg = df.groupby("AssignedGroup")["MTTR_Hours"].mean().round(1).reset_index()
        mg.columns = ["group","mttr"]
        mg = mg.sort_values("mttr")
        mttr_grp_labels, mttr_grp_vals = mg["group"].tolist(), mg["mttr"].tolist()
    else:
        mttr_grp_labels = mttr_grp_vals = []

    # Day of week
    if has_reported and "DayOfWeek" in df.columns:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["DayOfWeek"].value_counts().reindex(days, fill_value=0)
        dow_labels, dow_data = dow.index.tolist(), dow.values.tolist()
    else:
        dow_labels = dow_data = []

    # Group transfers distribution
    if has_xfer:
        xf = df["Group_Transfers"].value_counts().sort_index().reset_index()
        xf.columns = ["transfers","count"]
        xf_labels = [str(int(v)) for v in xf["transfers"].tolist()]
        xf_data   = xf["count"].tolist()
    else:
        xf_labels = xf_data = []

    # Service type
    if has_svc:
        svc = df["Service_Type"].value_counts().reset_index()
        svc.columns = ["type","count"]
        svc_labels, svc_data = svc["type"].tolist(), svc["count"].tolist()
    else:
        svc_labels = svc_data = []

    # ── CI breakdown ──────────────────────────────────────────────────────────
    if has_ci:
        ci_all = df.groupby("HPD_CI").size().reset_index(name="count").sort_values("count", ascending=False)
        ci_labels = ci_all.head(15)["HPD_CI"].tolist()
        ci_data   = ci_all.head(15)["count"].tolist()

        # CI > 3 with AssignedGroup breakdown
        ci_gt3_list = []
        ci_gt3_base = ci_all[ci_all["count"] > 3].copy()
        for _, row in ci_gt3_base.iterrows():
            ci_name   = row["HPD_CI"]
            ci_inc_ct = int(row["count"])
            sub = df[df["HPD_CI"] == ci_name]
            if has_group:
                grp_breakdown = (
                    sub.groupby("AssignedGroup").size()
                    .reset_index(name="cnt")
                    .sort_values("cnt", ascending=False)
                    .apply(lambda r: f"{r['AssignedGroup']} ({r['cnt']})", axis=1)
                    .tolist()
                )
            else:
                grp_breakdown = []
            ci_gt3_list.append({
                "HPD_CI":    ci_name,
                "count":     ci_inc_ct,
                "groups":    grp_breakdown,
                "groups_str": " · ".join(grp_breakdown) if grp_breakdown else "—",
            })
    else:
        ci_labels = ci_data = []
        ci_gt3_list = []

    # ── Team: per-group tables with assignee rows ─────────────────────────────
    group_tables = []
    if has_group and has_assignee:
        for grp_name, grp_df in df.groupby("AssignedGroup"):
            rows = []
            for assignee, a_df in grp_df.groupby("Assignee"):
                r = {"Assignee": assignee, "Total": len(a_df)}
                if has_status:
                    r["Resolved"] = int(a_df["Status"].str.strip().str.lower().isin(closed_statuses).sum())
                if "MTTR_Hours" in df.columns:
                    r["Avg_MTTR_hrs"] = round(float(a_df["MTTR_Hours"].mean()), 1) if a_df["MTTR_Hours"].notna().any() else "—"
                if has_sla:
                    met = a_df["SLAStatus"].str.strip().str.lower().isin(met_values).sum()
                    r["SLA_Pct"] = round(met / len(a_df) * 100, 1)
                if has_xfer:
                    r["Avg_Transfers"] = round(float(a_df["Group_Transfers"].mean()), 1) if a_df["Group_Transfers"].notna().any() else "—"
                rows.append(r)
            rows.sort(key=lambda x: x["Total"], reverse=True)

            # Group-level totals
            g_total = len(grp_df)
            g_resolved = int(grp_df["Status"].str.strip().str.lower().isin(closed_statuses).sum()) if has_status else g_total
            g_mttr = round(float(grp_df["MTTR_Hours"].mean()), 1) if "MTTR_Hours" in df.columns and grp_df["MTTR_Hours"].notna().any() else "—"
            g_sla  = round(grp_df["SLAStatus"].str.strip().str.lower().isin(met_values).sum() / g_total * 100, 1) if has_sla and g_total else "—"
            g_xfer = round(float(grp_df["Group_Transfers"].mean()), 1) if has_xfer and grp_df["Group_Transfers"].notna().any() else "—"

            group_tables.append({
                "group": grp_name,
                "total": g_total,
                "resolved": g_resolved,
                "mttr": g_mttr,
                "sla_pct": g_sla,
                "avg_transfers": g_xfer,
                "assignees": rows,
            })
        group_tables.sort(key=lambda x: x["total"], reverse=True)

    # ── Org summary ───────────────────────────────────────────────────────────
    org_rows = []
    if has_org:
        og = df.groupby("Assigned_Support_Organisation")
        org_agg = og.size().reset_index(name="total")
        if has_status:
            org_agg = org_agg.merge(
                og["Status"].apply(lambda x: x.str.strip().str.lower().isin(open_statuses).sum()).reset_index(name="open"),
                on="Assigned_Support_Organisation")
        if has_sla:
            org_agg = org_agg.merge(
                og["SLAStatus"].apply(lambda x: round(x.str.strip().str.lower().isin(met_values).sum() / len(x) * 100, 1)).reset_index(name="sla_pct"),
                on="Assigned_Support_Organisation")
        if "MTTR_Hours" in df.columns:
            org_agg = org_agg.merge(
                og["MTTR_Hours"].mean().round(1).reset_index(name="mttr"),
                on="Assigned_Support_Organisation")
        org_rows = org_agg.fillna("—").to_dict("records")

    # ── KB Article Analysis ───────────────────────────────────────────────────
    kb_available = has_kb_type and has_kb_desc
    kb_group_rows   = []
    kb_no_kb_rows   = []
    kb_summary      = {}

    if kb_available:
        # Extract KB article ID from description
        df["_kb_id"] = df["Request_Type_Description"].apply(extract_kb)
        # RKM Solution rows
        rkm_mask = df["Request_Type01"].str.strip().str.lower().isin(
            ["rkm solution", "rkmsolution", "rkm_solution", "kb solution", "knowledge"]
        )
        rkm_df   = df[rkm_mask].copy()
        no_kb_df = df[~rkm_mask].copy()

        total_rkm   = len(rkm_df)
        total_no_kb = len(no_kb_df)

        kb_summary = {
            "total_rkm":    total_rkm,
            "total_no_kb":  total_no_kb,
            "unique_kb_ids": int(rkm_df["_kb_id"].nunique()),
        }

        # Per-group KB stats
        if has_group:
            for grp_name, grp_df in df.groupby("AssignedGroup"):
                grp_rkm    = grp_df[grp_df["Request_Type01"].str.strip().str.lower().isin(
                    ["rkm solution","rkmsolution","rkm_solution","kb solution","knowledge"]
                )]
                grp_no_kb  = grp_df[~grp_df.index.isin(grp_rkm.index)]
                kb_ids     = grp_rkm["_kb_id"].dropna().value_counts().head(5).reset_index()
                kb_ids.columns = ["KB_ID","count"]
                kb_group_rows.append({
                    "group":       grp_name,
                    "total":       len(grp_df),
                    "with_kb":     len(grp_rkm),
                    "without_kb":  len(grp_no_kb),
                    "top_kbs":     kb_ids.to_dict("records"),
                    "kb_pct":      round(len(grp_rkm) / len(grp_df) * 100, 1) if len(grp_df) else 0,
                })
            kb_group_rows.sort(key=lambda x: x["with_kb"], reverse=True)

        # Incidents with no KB (no_kb_df sample)
        no_kb_cols = [c for c in [inc_col, "AssignedGroup", "Priority", "Status"] if c in df.columns]
        kb_no_kb_rows = no_kb_df[no_kb_cols].head(200).fillna("—").to_dict("records")

    return {
        # meta
        "total": total, "closed_ct": closed_ct,
        "date_min": date_min, "date_max": date_max,
        "sla_met_ct": sla_met_ct, "sla_pct": sla_pct,
        "mttr": mttr, "mttr_source": mttr_source,
        "p1_ct": p1_ct,
        "detected_cols": [c for c in df.columns if not c.startswith("_")],
        "has_kb": kb_available,

        # alert tables (new)
        "mttr_alert_rows": mttr_alert_rows,
        "hop_alert_rows":  hop_alert_rows,
        "age_alert_rows":  age_alert_rows,

        # overview charts
        "vol_labels": vol_labels, "vol_data": vol_data,
        "pri_labels": pri_labels, "pri_data": pri_data,
        "sla_labels": sla_labels, "sla_data": sla_data,
        "sla_pri_labels": sla_pri_labels, "sla_pri_vals": sla_pri_vals,
        "sla_tr_labels": sla_tr_labels, "sla_tr_vals": sla_tr_vals,
        "grp_labels": grp_labels, "grp_data": grp_data,
        "mttr_grp_labels": mttr_grp_labels, "mttr_grp_vals": mttr_grp_vals,
        "dow_labels": dow_labels, "dow_data": dow_data,
        "xf_labels": xf_labels, "xf_data": xf_data,

        # CI & service
        "ci_labels": ci_labels, "ci_data": ci_data,
        "ci_gt3_rows": ci_gt3_list,
        "svc_labels": svc_labels, "svc_data": svc_data,

        # tables
        "org_rows":      org_rows,
        "group_tables":  group_tables,

        # KB
        "kb_summary":      kb_summary,
        "kb_group_rows":   kb_group_rows,
        "kb_no_kb_rows":   kb_no_kb_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    log.info("Upload from %s", request.remote_addr)

    if "file" not in request.files:
        return jsonify({"error": "No file received. Please select a file and try again."}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    fname = f.filename.lower().strip()
    log.info("Filename: %s", f.filename)

    try:
        file_bytes = io.BytesIO(f.read())
    except Exception as e:
        return jsonify({"error": f"Could not receive file: {e}"}), 400

    df = None
    try:
        if fname.endswith(".xlsx"):
            df = pd.read_excel(file_bytes, engine="openpyxl")
        elif fname.endswith(".xls"):
            try:
                import xlrd  # noqa
                df = pd.read_excel(file_bytes, engine="xlrd")
            except ImportError:
                try:
                    file_bytes.seek(0)
                    df = pd.read_excel(file_bytes, engine="openpyxl")
                except Exception:
                    return jsonify({"error": ".xls needs xlrd: run  pip install xlrd  OR resave as .xlsx"}), 400
        elif fname.endswith(".csv"):
            df = pd.read_csv(file_bytes)
        else:
            return jsonify({"error": f"Unsupported file type '{f.filename}'. Use .xlsx, .xls, or .csv"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400

    if df is None or df.empty:
        return jsonify({"error": "File is empty or has no readable data."}), 400

    log.info("Read OK — %d rows, %d cols: %s", len(df), len(df.columns), list(df.columns)[:8])

    try:
        result = analyse(df)
        log.info("Analysis done — total: %d, MTTR alerts: %d, HOP alerts: %d, Age alerts: %d",
                 result["total"], len(result["mttr_alert_rows"]),
                 len(result["hop_alert_rows"]), len(result["age_alert_rows"]))
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ITIS Incident Analyzer v3</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#07090f;--surf:#0d111e;--card:#111827;--card2:#161e30;
  --border:#1e2a40;--border2:#263047;
  --blue:#4a8cff;--red:#ff4f6a;--green:#30d988;--yellow:#ffc240;
  --purple:#a78bfa;--cyan:#22d3ee;--orange:#fb923c;--pink:#f472b6;
  --text:#edf2ff;--muted:#7b8db0;--dim:#3a4b6b;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh}
/* HEADER */
.hdr{position:sticky;top:0;z-index:500;background:rgba(7,9,15,.94);backdrop-filter:blur(14px);
  border-bottom:1px solid var(--border);padding:13px 28px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:11px}
.brand-icon{width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:16px}
.brand-name{font-size:.95rem;font-weight:800;letter-spacing:-.02em}
.brand-name span{color:var(--blue)}
.hdr-right{display:flex;align-items:center;gap:8px}
.pill{background:var(--card);border:1px solid var(--border);padding:4px 11px;border-radius:20px;
  font-family:'DM Mono',monospace;font-size:.67rem;color:var(--muted)}
.pill.live{border-color:var(--green);color:var(--green)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);
  display:inline-block;margin-right:5px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* UPLOAD */
#upload-section{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:calc(100vh - 62px);padding:40px 20px;
  background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(74,140,255,.07),transparent 70%)}
.ucard{width:100%;max-width:600px;background:var(--card);border:1px solid var(--border);
  border-radius:18px;padding:44px 38px;text-align:center;position:relative;overflow:hidden}
.ucard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--purple),var(--cyan))}
.u-title{font-size:1.6rem;font-weight:800;letter-spacing:-.03em;margin-bottom:6px}
.u-title span{color:var(--blue)}
.u-sub{color:var(--muted);font-size:.85rem;margin-bottom:28px;line-height:1.65}
.drop{border:2px dashed var(--border2);border-radius:12px;padding:36px 20px;cursor:pointer;
  transition:all .2s;position:relative;background:var(--card2)}
.drop:hover,.drop.drag{border-color:var(--blue);background:rgba(74,140,255,.06)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:2.2rem;margin-bottom:10px}
.drop-text{font-size:.88rem;color:var(--muted);line-height:1.6}
.drop-text strong{color:var(--text)}
.fmts{margin-top:8px;display:flex;gap:6px;justify-content:center}
.fmt{background:var(--surf);border:1px solid var(--border);padding:2px 9px;border-radius:5px;
  font-size:.66rem;font-family:'DM Mono',monospace;color:var(--muted)}
.ubtn{margin-top:20px;background:linear-gradient(135deg,var(--blue),var(--purple));
  border:none;color:#fff;padding:13px 36px;border-radius:11px;font-size:.92rem;font-weight:700;
  cursor:pointer;font-family:'Syne',sans-serif;transition:opacity .2s,transform .15s;width:100%}
.ubtn:hover{opacity:.9;transform:translateY(-1px)}
.ubtn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.fname{margin-top:10px;font-size:.78rem;color:var(--green);font-family:'DM Mono',monospace}
.alert-box{background:rgba(255,79,106,.1);border:1px solid rgba(255,79,106,.3);
  border-radius:9px;padding:12px 14px;color:var(--red);font-size:.83rem;margin-top:10px}
/* SPINNER */
#spin{display:none;position:fixed;inset:0;background:rgba(7,9,15,.85);z-index:900;
  align-items:center;justify-content:center;flex-direction:column;gap:14px}
#spin.show{display:flex}
.spinner{width:44px;height:44px;border:3px solid var(--border2);border-top-color:var(--blue);
  border-radius:50%;animation:rot .8s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
/* DASHBOARD */
#dash{display:none}
#dash.show{display:block}
.new-btn{margin:0 28px 16px;display:inline-flex;align-items:center;gap:8px;
  background:var(--card);border:1px solid var(--border);color:var(--muted);
  padding:9px 18px;border-radius:9px;cursor:pointer;font-family:'Syne',sans-serif;
  font-size:.8rem;font-weight:600;transition:all .2s;margin-top:16px}
.new-btn:hover{border-color:var(--blue);color:var(--blue)}
/* TABS */
.tabs{background:var(--surf);border-bottom:1px solid var(--border);
  padding:0 28px;display:flex;gap:2px;overflow-x:auto;position:sticky;top:62px;z-index:400}
.tab{padding:12px 17px;cursor:pointer;font-size:.8rem;font-weight:600;color:var(--muted);
  border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;user-select:none}
.tab:hover{color:var(--text)}
.tab.on{color:var(--blue);border-bottom-color:var(--blue)}
.tab.kb-tab{display:none}
.tab.kb-tab.show{display:block}
/* PAGES */
.page{display:none;padding:22px 28px}
.page.on{display:block}
/* KPIs */
.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:13px;margin-bottom:18px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:17px 15px;position:relative;overflow:hidden;transition:transform .15s,border-color .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi-bar{position:absolute;top:0;left:0;right:0;height:2px}
.kpi-lbl{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:8px}
.kpi-val{font-size:1.75rem;font-weight:800;font-family:'DM Mono',monospace;line-height:1;letter-spacing:-.03em}
.kpi-sub{font-size:.62rem;color:var(--dim);margin-top:4px}
.kpi-icon{position:absolute;right:13px;top:13px;font-size:1.2rem;opacity:.2}
/* SECTION */
.sec{font-size:.63rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin:20px 0 11px;display:flex;align-items:center;gap:9px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
/* CHART CARD */
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}
.cc-hd{display:flex;align-items:center;gap:7px;margin-bottom:14px}
.cc-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.cc-title{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.cc-badge{margin-left:auto;background:var(--surf);border:1px solid var(--border);
  padding:2px 8px;border-radius:7px;font-size:.63rem;color:var(--dim);font-family:'DM Mono',monospace}
/* GRIDS */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin-bottom:15px}
.g3-2{display:grid;grid-template-columns:3fr 2fr;gap:15px;margin-bottom:15px}
.full{margin-bottom:15px}
/* ALERT CARDS */
.alert-card{background:var(--card);border-radius:12px;border:1px solid var(--border);
  margin-bottom:15px;overflow:hidden}
.alert-hdr{padding:13px 18px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--border)}
.alert-hdr .icon{font-size:1.1rem}
.alert-hdr .title{font-size:.8rem;font-weight:700;letter-spacing:-.01em}
.alert-hdr .badge{margin-left:auto;padding:3px 10px;border-radius:8px;font-size:.68rem;font-weight:700}
.alert-none{padding:16px 18px;color:var(--dim);font-size:.82rem}
/* TABLES */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.77rem}
thead th{padding:9px 11px;background:var(--surf);color:var(--muted);font-weight:700;
  font-size:.64rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;
  border-bottom:1px solid var(--border);white-space:nowrap}
tbody td{padding:9px 11px;border-bottom:1px solid rgba(30,42,64,.6);vertical-align:middle}
tbody tr:hover td{background:rgba(74,140,255,.04)}
tbody tr:last-child td{border-bottom:none}
/* GROUP PANEL */
.grp-panel{background:var(--card);border:1px solid var(--border);border-radius:12px;
  margin-bottom:14px;overflow:hidden}
.grp-panel-hdr{padding:13px 18px;background:var(--card2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;cursor:pointer;user-select:none}
.grp-panel-hdr .grp-name{font-weight:700;font-size:.88rem;letter-spacing:-.01em}
.grp-panel-hdr .grp-stats{display:flex;gap:14px;margin-left:auto;flex-wrap:wrap}
.grp-stat{font-size:.72rem;color:var(--muted);font-family:'DM Mono',monospace}
.grp-stat strong{color:var(--text)}
.grp-expand{font-size:.8rem;color:var(--dim);margin-left:8px;transition:transform .2s}
.grp-panel-hdr.open .grp-expand{transform:rotate(180deg)}
.grp-body{display:none;padding:0}
.grp-body.open{display:block}
/* CHIPS */
.chip{display:inline-block;padding:2px 8px;border-radius:7px;font-size:.67rem;font-weight:700;white-space:nowrap}
.c-met{background:rgba(48,217,136,.15);color:var(--green)}
.c-breach{background:rgba(255,79,106,.15);color:var(--red)}
.c-pend{background:rgba(255,194,64,.15);color:var(--yellow)}
.c-blue{background:rgba(74,140,255,.12);color:var(--blue)}
.c-purple{background:rgba(167,139,250,.12);color:var(--purple)}
.c-orange{background:rgba(251,146,60,.12);color:var(--orange)}
.c-cyan{background:rgba(34,211,238,.12);color:var(--cyan)}
/* MINI BAR */
.mbar{display:flex;align-items:center;gap:6px;min-width:90px}
.mbar-bg{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.mbar-fill{height:100%;border-radius:3px}
/* MTTR source note */
.mttr-note{font-size:.68rem;color:var(--dim);font-family:'DM Mono',monospace;
  margin-top:4px;display:flex;align-items:center;gap:5px}
footer{text-align:center;padding:20px;color:var(--dim);font-size:.68rem;
  border-top:1px solid var(--border);margin-top:6px}
@media(max-width:1100px){.kpi-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.g2,.g3,.g3-2{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}}
@media(max-width:500px){.kpi-row{grid-template-columns:1fr}.page{padding:14px 12px}.tabs{padding:0 12px}}
</style>
</head>
<body>

<header class="hdr">
  <div class="brand">
    <div class="brand-icon">⚡</div>
    <div class="brand-name">ITIS <span>Incident</span> Analyzer <span style="font-size:.7rem;color:var(--dim);margin-left:4px">v3</span></div>
  </div>
  <div class="hdr-right">
    <span class="pill live"><span class="dot"></span>Live</span>
    <span class="pill" id="file-pill">No file</span>
    <span class="pill" id="mttr-pill" style="display:none"></span>
  </div>
</header>

<div id="spin"><div class="spinner"></div><div style="color:var(--muted);font-size:.85rem">Analysing incident data…</div></div>

<!-- UPLOAD -->
<section id="upload-section">
  <div class="ucard">
    <div class="u-title">ITIS <span>Incident</span> Dashboard</div>
    <div class="u-sub">Upload any ITIS Excel / CSV export.<br>Columns auto-detected. Extra columns preserved.</div>
    <div class="drop" id="drop">
      <input type="file" id="file-input" accept=".xlsx,.xls,.csv"/>
      <div class="drop-icon">📂</div>
      <div class="drop-text"><strong>Click to browse</strong> or drag &amp; drop</div>
      <div class="fmts"><span class="fmt">.xlsx</span><span class="fmt">.xls</span><span class="fmt">.csv</span></div>
    </div>
    <div class="fname" id="fname"></div>
    <div id="uerr"></div>
    <button class="ubtn" id="abtn" disabled onclick="doUpload()">Analyse File →</button>
  </div>
</section>

<!-- DASHBOARD -->
<div id="dash">
  <div style="padding:16px 28px 0">
    <button class="new-btn" onclick="reset()">← Upload New File</button>
  </div>
  <nav class="tabs">
    <div class="tab on"  onclick="go('overview',this)">📊 Overview</div>
    <div class="tab"     onclick="go('trends',this)">📈 Trends</div>
    <div class="tab"     onclick="go('sla',this)">🎯 SLA Analysis</div>
    <div class="tab"     onclick="go('team',this)">👥 Team Performance</div>
    <div class="tab"     onclick="go('ci',this)">🖥 CI &amp; Service</div>
    <div class="tab kb-tab" id="kb-tab" onclick="go('kb',this)">📚 KB Articles</div>
    <div class="tab"     onclick="go('info',this)">🔍 Data Info</div>
  </nav>

  <!-- ══ OVERVIEW ══ -->
  <section class="page on" id="pg-overview">
    <div class="kpi-row" id="kpi-row"></div>

    <div class="g3">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Priority Distribution</span></div><canvas id="c-pri" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Status</span></div><canvas id="c-sla-donut" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Incidents by AssignedGroup</span><span class="cc-badge" id="grp-badge"></span></div><canvas id="c-grp" height="210"></canvas></div>
    </div>

    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Organisation Summary</span></div>
      <div class="tbl-wrap"><table><thead><tr><th>Organisation</th><th>Total</th><th>Open</th><th>SLA %</th><th>MTTR (hrs)</th><th>SLA Bar</th></tr></thead><tbody id="org-tbody"></tbody></table></div>
    </div>

    <div class="sec">🚨 Alerts &amp; Exceptions</div>

    <!-- MTTR > 200h -->
    <div class="alert-card">
      <div class="alert-hdr" style="background:rgba(255,79,106,.07);border-left:3px solid var(--red)">
        <span class="icon">⏱</span>
        <div>
          <div class="title">MTTR &gt; 200 Hours</div>
          <div style="font-size:.68rem;color:var(--muted);margin-top:2px">Incidents with extremely long resolution time</div>
        </div>
        <span class="badge" id="mttr-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
      </div>
      <div id="mttr-alert-body">
        <div class="alert-none">No incidents exceed 200 hours MTTR</div>
      </div>
    </div>

    <!-- HOP > 5 -->
    <div class="alert-card">
      <div class="alert-hdr" style="background:rgba(251,146,60,.07);border-left:3px solid var(--orange)">
        <span class="icon">🔄</span>
        <div>
          <div class="title">High HOP Count — Group Transfers &gt; 5</div>
          <div style="font-size:.68rem;color:var(--muted);margin-top:2px">Incidents bounced across more than 5 groups</div>
        </div>
        <span class="badge" id="hop-badge" style="background:rgba(251,146,60,.15);color:var(--orange)">0</span>
      </div>
      <div id="hop-alert-body">
        <div class="alert-none">No incidents have Group Transfers &gt; 5</div>
      </div>
    </div>

    <!-- Aging > 30d -->
    <div class="alert-card">
      <div class="alert-hdr" style="background:rgba(255,194,64,.07);border-left:3px solid var(--yellow)">
        <span class="icon">📅</span>
        <div>
          <div class="title">Incident Aging &gt; 30 Days</div>
          <div style="font-size:.68rem;color:var(--muted);margin-top:2px">Resolved date minus Submit date exceeds 30 days</div>
        </div>
        <span class="badge" id="age-badge" style="background:rgba(255,194,64,.15);color:var(--yellow)">0</span>
      </div>
      <div id="age-alert-body">
        <div class="alert-none">No incidents exceed 30 days aging</div>
      </div>
    </div>
  </section>

  <!-- ══ TRENDS ══ -->
  <section class="page" id="pg-trends">
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Monthly Incident Volume</span></div>
      <canvas id="c-vol" height="110"></canvas>
    </div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Incidents by Day of Week</span></div><canvas id="c-dow" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Group Transfers Distribution</span></div><canvas id="c-xfer" height="210"></canvas></div>
    </div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance Trend — Monthly %</span></div>
      <canvas id="c-sla-trend" height="110"></canvas>
    </div>
  </section>

  <!-- ══ SLA ══ -->
  <section class="page" id="pg-sla">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance % by Priority</span></div><canvas id="c-sla-pri" height="230"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">MTTR by AssignedGroup (hrs)</span></div><canvas id="c-mttr-grp" height="230"></canvas></div>
    </div>
  </section>

  <!-- ══ TEAM ══ -->
  <section class="page" id="pg-team">
    <div class="sec">Per-Group Performance — click a group to expand assignees</div>
    <div id="team-panels"></div>
  </section>

  <!-- ══ CI & SERVICE ══ -->
  <section class="page" id="pg-ci">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--cyan)"></span><span class="cc-title">Top HPD_CI by Count</span></div><canvas id="c-ci" height="280"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Service Type</span></div><canvas id="c-svc" height="280"></canvas></div>
    </div>
    <div class="full alert-card">
      <div class="alert-hdr" style="border-left:3px solid var(--cyan)">
        <span class="icon">🖥</span>
        <div>
          <div class="title">HPD_CI with Incident Count &gt; 3  — with AssignedGroup breakdown</div>
          <div style="font-size:.68rem;color:var(--muted);margin-top:2px">Repeat-offender assets / servers</div>
        </div>
        <span class="badge" id="ci3-badge" style="background:rgba(34,211,238,.12);color:var(--cyan)">0</span>
      </div>
      <div class="tbl-wrap" style="padding:0 4px">
        <table><thead><tr><th>#</th><th>HPD_CI</th><th>Total Incidents</th><th>AssignedGroup Breakdown</th><th>Volume</th></tr></thead>
        <tbody id="ci3-tbody"></tbody></table>
      </div>
    </div>
  </section>

  <!-- ══ KB ARTICLES ══ -->
  <section class="page" id="pg-kb">
    <div class="kpi-row" style="grid-template-columns:repeat(3,1fr)" id="kb-kpi-row"></div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span>
        <span class="cc-title">KB Article Usage by AssignedGroup</span>
        <span style="font-size:.7rem;color:var(--dim);margin-left:8px">Request_Type01 = RKM Solution</span>
      </div>
      <div class="tbl-wrap">
        <table><thead><tr><th>AssignedGroup</th><th>Total</th><th>With KB Article</th><th>Without KB</th><th>KB %</th><th>Top KB IDs</th><th>Coverage</th></tr></thead>
        <tbody id="kb-tbody"></tbody></table>
      </div>
    </div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--red)"></span>
        <span class="cc-title">Incidents Without KB Article (sample — first 200)</span>
      </div>
      <div class="tbl-wrap">
        <table><thead id="kb-no-thead"></thead><tbody id="kb-no-tbody"></tbody></table>
      </div>
    </div>
  </section>

  <!-- ══ DATA INFO ══ -->
  <section class="page" id="pg-info">
    <div class="cc full">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Detected Columns</span></div>
      <div id="col-chips" style="display:flex;flex-wrap:wrap;gap:7px;margin-top:4px"></div>
    </div>
    <div class="g2">
      <div class="cc">
        <div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">MTTR Calculation Logic</span></div>
        <div style="font-size:.82rem;color:var(--muted);line-height:1.8" id="mttr-logic-panel"></div>
      </div>
      <div class="cc">
        <div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Date Column Format</span></div>
        <div style="font-size:.82rem;color:var(--muted);line-height:1.8">
          Dates stored as <strong style="color:var(--yellow)">Excel serial numbers</strong>.<br><br>
          Conversion: <span style="color:var(--blue);font-family:'DM Mono',monospace">epoch + serial days → datetime</span><br><br>
          Used for: MTTR · Incident Aging · Monthly trends · Date range
        </div>
      </div>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script>
const $ = id => document.getElementById(id);
let charts = {}, D = null, selectedFile = null;

Chart.defaults.color = '#7b8db0';
Chart.defaults.borderColor = '#1e2a40';
Chart.defaults.font.family = "'Syne',sans-serif";
Chart.defaults.font.size = 11;

const PAL = ['#4a8cff','#ff4f6a','#30d988','#ffc240','#a78bfa','#22d3ee','#fb923c','#f472b6','#34d399','#f87171'];
const PRI_C = {'P1 - Critical':'#ff4f6a','Critical':'#ff4f6a','P2 - High':'#ffc240','High':'#ffc240','P3 - Medium':'#4a8cff','Medium':'#4a8cff','P4 - Low':'#30d988','Low':'#30d988'};
const SLA_C = {'Met':'#30d988','Breached':'#ff4f6a','Pending':'#ffc240','Exempt':'#7b8db0','Within SLA':'#30d988','SLA Met':'#30d988','OK':'#30d988','Yes':'#30d988'};

// ── file input ─────────────────────────────────────────────────────────────
const fi = $('file-input'), drop = $('drop');
fi.onchange = () => {
  selectedFile = fi.files[0];
  if (selectedFile) { $('fname').textContent = '📄 '+selectedFile.name; $('abtn').disabled = false; $('uerr').innerHTML=''; }
};
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('drag');
  if (e.dataTransfer.files.length) { selectedFile = e.dataTransfer.files[0]; $('fname').textContent='📄 '+selectedFile.name; $('abtn').disabled=false; }
});

function doUpload() {
  if (!selectedFile) return;
  const fd = new FormData(); fd.append('file', selectedFile);
  $('spin').classList.add('show'); $('abtn').disabled = true; $('uerr').innerHTML='';
  fetch('/upload', { method:'POST', body:fd })
    .then(r => r.text().then(t => {
      try { return { ok:r.ok, data:JSON.parse(t) }; }
      catch(e) { return { ok:false, data:{error:'Server error (HTTP '+r.status+'). Check terminal.'} }; }
    }))
    .then(res => {
      $('spin').classList.remove('show'); $('abtn').disabled=false;
      if (!res.ok || res.data.error) { $('uerr').innerHTML='<div class="alert-box">⚠ '+(res.data.error||'Unknown error')+'</div>'; return; }
      D = res.data; build(D);
    })
    .catch(err => {
      $('spin').classList.remove('show'); $('abtn').disabled=false;
      $('uerr').innerHTML='<div class="alert-box">⚠ Connection error: '+err.message+'</div>';
    });
}

function reset() {
  Object.values(charts).forEach(c => { try{c.destroy();}catch(_){} }); charts={}; D=null;
  $('dash').classList.remove('show'); $('upload-section').style.display='';
  $('abtn').disabled=true; fi.value=''; $('fname').textContent=''; $('file-pill').textContent='No file';
  $('mttr-pill').style.display='none';
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('on',i===0));
  document.querySelectorAll('.page').forEach((p,i) => p.classList.toggle('on',i===0));
}

function go(name, el) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  $('pg-'+name).classList.add('on'); el.classList.add('on');
}

function mk(id, type, labels, datasets, extra={}) {
  if (charts[id]) charts[id].destroy();
  const ctx = $(id); if (!ctx) return;
  charts[id] = new Chart(ctx, { type, data:{labels,datasets},
    options:{ responsive:true, maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#7b8db0',boxWidth:10,padding:12}}}, ...extra }});
}

// ── MAIN BUILD ─────────────────────────────────────────────────────────────
function build(d) {
  $('upload-section').style.display='none'; $('dash').classList.add('show');
  $('file-pill').textContent = selectedFile ? selectedFile.name : 'Loaded';
  $('mttr-pill').textContent = 'MTTR: '+d.mttr_source; $('mttr-pill').style.display='';
  $('footer').textContent = 'ITIS Incident Analyzer v3 · '+d.detected_cols.length+' columns · '+d.total+' records';

  if (d.has_kb) { $('kb-tab').classList.add('show'); }

  buildKPIs(d);
  buildAlerts(d);
  buildCharts(d);
  buildOrgTable(d);
  buildTeamPanels(d);
  buildCI(d);
  if (d.has_kb) buildKB(d);
  buildInfo(d);
}

// ── KPIs ───────────────────────────────────────────────────────────────────
function buildKPIs(d) {
  const kpis = [
    { l:'Total Incidents',   v:d.total,      c:'var(--blue)',   i:'📋', s:'All records in file' },
    { l:'Closed / Resolved', v:d.closed_ct,  c:'var(--green)',  i:'✅', s:'Resolved & Closed' },
    { l:'SLA Compliance',    v:d.sla_pct+'%',c:'var(--green)',  i:'🎯', s:d.sla_met_ct+' met SLA' },
    { l:'MTTR (avg)',        v:d.mttr+'h',   c:'var(--yellow)', i:'⏱', s:'Mean Time To Resolve' },
    { l:'P1 Critical',       v:d.p1_ct,      c:'var(--red)',    i:'🚨', s:'Highest priority' },
  ];
  $('kpi-row').innerHTML = kpis.map(k=>`
    <div class="kpi">
      <div class="kpi-bar" style="background:${k.c}"></div>
      <div class="kpi-icon">${k.i}</div>
      <div class="kpi-lbl">${k.l}</div>
      <div class="kpi-val" style="color:${k.c}">${k.v}</div>
      <div class="kpi-sub">${k.s}</div>
    </div>`).join('')
  + `<div style="grid-column:1/-1;background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:11px 16px;display:flex;align-items:center;gap:14px;font-size:.78rem">
      <span style="color:var(--muted)">📅 Date Range:</span>
      <span style="color:var(--blue);font-family:'DM Mono',monospace">${d.date_min} → ${d.date_max}</span>
    </div>`;
}

// ── ALERTS ─────────────────────────────────────────────────────────────────
function alertTable(rows, cols) {
  if (!rows.length) return '<div class="alert-none">None found ✓</div>';
  const ths = cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('');
  const trs = rows.map(r => '<tr>'+cols.map(c => {
    const v = r[c] !== undefined ? r[c] : '—';
    if (c==='MTTR_Hours' || c==='AgeDays') return `<td style="font-family:'DM Mono',monospace;color:var(--red);font-weight:700">${v}</td>`;
    if (c==='Group_Transfers') return `<td><span class="chip c-orange">${v}</span></td>`;
    if (c==='Priority') return `<td><span class="chip" style="background:${(PRI_C[v]||'#64748b')}22;color:${PRI_C[v]||'#64748b'}">${v}</span></td>`;
    return `<td>${v}</td>`;
  }).join('')+'</tr>').join('');
  return `<div class="tbl-wrap"><table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table></div>`;
}

function buildAlerts(d) {
  // MTTR > 200
  $('mttr-badge').textContent = d.mttr_alert_rows.length;
  const mttrCols = d.mttr_alert_rows.length
    ? Object.keys(d.mttr_alert_rows[0])
    : [];
  $('mttr-alert-body').innerHTML = alertTable(d.mttr_alert_rows, mttrCols);

  // HOP > 5
  $('hop-badge').textContent = d.hop_alert_rows.length;
  const hopCols = d.hop_alert_rows.length ? Object.keys(d.hop_alert_rows[0]) : [];
  $('hop-alert-body').innerHTML = alertTable(d.hop_alert_rows, hopCols);

  // Aging > 30d
  $('age-badge').textContent = d.age_alert_rows.length;
  const ageCols = d.age_alert_rows.length ? Object.keys(d.age_alert_rows[0]) : [];
  $('age-alert-body').innerHTML = alertTable(d.age_alert_rows, ageCols);
}

// ── CHARTS ─────────────────────────────────────────────────────────────────
function buildCharts(d) {
  const yg = {y:{grid:{color:'#1e2a40'}},x:{grid:{display:false}}};
  const ypct = {y:{grid:{color:'#1e2a40'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}};

  if (d.pri_labels.length) mk('c-pri','doughnut',d.pri_labels,[{
    data:d.pri_data, backgroundColor:d.pri_labels.map(l=>PRI_C[l]||PAL[4]),
    borderWidth:2, borderColor:'#111827'}],{cutout:'62%'});

  if (d.sla_labels.length) mk('c-sla-donut','doughnut',d.sla_labels,[{
    data:d.sla_data, backgroundColor:d.sla_labels.map(l=>SLA_C[l]||PAL[4]),
    borderWidth:2, borderColor:'#111827'}],{cutout:'60%'});

  if (d.grp_labels.length) {
    $('grp-badge').textContent = d.grp_labels.length+' groups';
    mk('c-grp','bar',d.grp_labels,[{label:'Incidents',data:d.grp_data,
      backgroundColor:'rgba(74,140,255,.75)',borderRadius:5,borderSkipped:false}],
      {indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{grid:{color:'#1e2a40'}},y:{grid:{display:false}}}});
  }

  if (d.vol_labels.length) mk('c-vol','line',d.vol_labels,[{label:'Incidents',data:d.vol_data,
    borderColor:'#4a8cff',backgroundColor:'rgba(74,140,255,.1)',fill:true,tension:.4,
    pointRadius:5,pointBackgroundColor:'#4a8cff',pointBorderColor:'#07090f',pointBorderWidth:2}],{scales:yg});

  if (d.dow_labels.length) mk('c-dow','bar',d.dow_labels,[{label:'Incidents',data:d.dow_data,
    backgroundColor:d.dow_labels.map((_,i)=>PAL[i%PAL.length]),borderRadius:6}],
    {plugins:{legend:{display:false}},scales:yg});

  if (d.xf_labels.length) mk('c-xfer','bar',d.xf_labels,[{label:'Count',data:d.xf_data,
    backgroundColor:'rgba(167,139,250,.75)',borderRadius:5}],
    {plugins:{legend:{display:false}},scales:yg});

  if (d.sla_tr_labels.length) mk('c-sla-trend','line',d.sla_tr_labels,[{label:'SLA %',data:d.sla_tr_vals,
    borderColor:'#30d988',backgroundColor:'rgba(48,217,136,.1)',fill:true,tension:.4,pointRadius:4,
    pointBackgroundColor:'#30d988'}],{scales:ypct});

  if (d.sla_pri_labels.length) mk('c-sla-pri','bar',d.sla_pri_labels,[{label:'SLA %',data:d.sla_pri_vals,
    backgroundColor:d.sla_pri_vals.map(v=>v>=70?'#30d988':v>=50?'#ffc240':'#ff4f6a'),
    borderRadius:7,borderSkipped:false}],{plugins:{legend:{display:false}},scales:ypct});

  if (d.mttr_grp_labels.length) mk('c-mttr-grp','bar',d.mttr_grp_labels,[{label:'Avg hrs',data:d.mttr_grp_vals,
    backgroundColor:'rgba(255,194,64,.75)',borderRadius:6,borderSkipped:false}],
    {indexAxis:'y',plugins:{legend:{display:false}},
     scales:{x:{grid:{color:'#1e2a40'},ticks:{callback:v=>v+'h'}},y:{grid:{display:false}}}});

  if (d.ci_labels.length) mk('c-ci','bar',d.ci_labels,[{label:'Incidents',data:d.ci_data,
    backgroundColor:'rgba(34,211,238,.7)',borderRadius:5,borderSkipped:false}],
    {indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{grid:{color:'#1e2a40'}},y:{grid:{display:false}}}});

  if (d.svc_labels.length) mk('c-svc','doughnut',d.svc_labels,[{
    data:d.svc_data,backgroundColor:PAL,borderWidth:2,borderColor:'#111827'}],{cutout:'55%'});
}

// ── ORG TABLE ─────────────────────────────────────────────────────────────
function buildOrgTable(d) {
  $('org-tbody').innerHTML = d.org_rows.map(r=>{
    const sla = parseFloat(r.sla_pct)||0;
    const cls = sla>=70?'c-met':sla>=50?'c-pend':'c-breach';
    return `<tr>
      <td style="font-weight:700">${r.Assigned_Support_Organisation||'—'}</td>
      <td style="font-family:'DM Mono',monospace;font-weight:700">${r.total}</td>
      <td><span class="chip c-breach">${r.open||0}</span></td>
      <td><span class="chip ${cls}">${sla}%</span></td>
      <td style="font-family:'DM Mono',monospace">${r.mttr||'—'}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${Math.min(sla,100)}%;background:${sla>=70?'#30d988':sla>=50?'#ffc240':'#ff4f6a'}"></div></div></div></td>
    </tr>`;
  }).join('');
}

// ── TEAM PANELS (grouped by AssignedGroup) ──────────────────────────────
function buildTeamPanels(d) {
  const container = $('team-panels');
  if (!d.group_tables.length) {
    container.innerHTML = '<div class="cc"><div style="color:var(--muted);font-size:.85rem">No AssignedGroup / Assignee data found in this file.</div></div>';
    return;
  }
  container.innerHTML = d.group_tables.map((g,gi) => {
    const sla_cls = typeof g.sla_pct==='number' ? (g.sla_pct>=70?'c-met':g.sla_pct>=50?'c-pend':'c-breach') : '';
    const sla_disp = typeof g.sla_pct==='number' ? g.sla_pct+'%' : g.sla_pct;
    const maxT = g.assignees.length ? Math.max(...g.assignees.map(a=>a.Total)) : 1;
    const aCols = g.assignees.length ? Object.keys(g.assignees[0]) : [];
    const aRows = g.assignees.map(a => {
      const asla = parseFloat(a.SLA_Pct||0);
      const aCls = asla>=70?'c-met':asla>=50?'c-pend':'c-breach';
      const pct  = Math.round(a.Total/maxT*100);
      return '<tr>'+aCols.map(k=>{
        const v = a[k];
        if (k==='SLA_Pct') return `<td><span class="chip ${aCls}">${v}%</span></td>`;
        if (k==='Assignee') return `<td style="font-weight:600">${v}</td>`;
        if (k==='Total') return `<td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--blue)"></div></div><span style="font-family:'DM Mono',monospace;font-size:.74rem;color:var(--muted)">${v}</span></div></td>`;
        return `<td style="font-family:'DM Mono',monospace;font-size:.76rem">${v}</td>`;
      }).join('')+'</tr>';
    }).join('');
    const thead = aCols.map(k=>`<th>${k.replace(/_/g,' ')}</th>`).join('');
    return `
    <div class="grp-panel">
      <div class="grp-panel-hdr" onclick="toggleGrp(this)" id="gh-${gi}">
        <span style="font-size:1rem">👥</span>
        <span class="grp-name">${g.group}</span>
        <div class="grp-stats">
          <span class="grp-stat">Total <strong>${g.total}</strong></span>
          <span class="grp-stat">Resolved <strong>${g.resolved}</strong></span>
          <span class="grp-stat">MTTR <strong>${g.mttr}h</strong></span>
          <span class="grp-stat">SLA <strong><span class="chip ${sla_cls}" style="font-size:.7rem">${sla_disp}</span></strong></span>
          <span class="grp-stat">Avg Xfers <strong>${g.avg_transfers}</strong></span>
        </div>
        <span class="grp-expand">▼</span>
      </div>
      <div class="grp-body" id="gb-${gi}">
        <div class="tbl-wrap"><table><thead><tr>${thead}</tr></thead><tbody>${aRows}</tbody></table></div>
      </div>
    </div>`;
  }).join('');
}

function toggleGrp(hdr) {
  hdr.classList.toggle('open');
  const id = hdr.id.replace('gh-','gb-');
  $(id).classList.toggle('open');
}

// ── CI TABLE ──────────────────────────────────────────────────────────────
function buildCI(d) {
  $('ci3-badge').textContent = d.ci_gt3_rows.length+' CIs';
  const maxC = d.ci_gt3_rows.length ? Math.max(...d.ci_gt3_rows.map(r=>r.count)) : 1;
  $('ci3-tbody').innerHTML = d.ci_gt3_rows.map((r,i)=>{
    const pct = Math.round(r.count/maxC*100);
    return `<tr>
      <td style="color:var(--dim);font-family:'DM Mono',monospace">${i+1}</td>
      <td style="font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace">${r.HPD_CI}</td>
      <td><span class="chip c-orange" style="font-family:'DM Mono',monospace">${r.count}</span></td>
      <td style="font-size:.75rem;color:var(--muted)">${r.groups_str}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--orange)"></div></div></div></td>
    </tr>`;
  }).join('');
}

// ── KB ARTICLES ───────────────────────────────────────────────────────────
function buildKB(d) {
  const s = d.kb_summary;
  $('kb-kpi-row').innerHTML = [
    {l:'RKM Solution Incidents', v:s.total_rkm,   c:'var(--purple)', i:'📚', sub:'Request_Type01 = RKM Solution'},
    {l:'Without KB',             v:s.total_no_kb, c:'var(--red)',    i:'❌', sub:'No KB article tagged'},
    {l:'Unique KB IDs',          v:s.unique_kb_ids,c:'var(--green)', i:'🔑', sub:'Distinct KB articles used'},
  ].map(k=>`<div class="kpi"><div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-icon">${k.i}</div><div class="kpi-lbl">${k.l}</div>
    <div class="kpi-val" style="color:${k.c}">${k.v}</div><div class="kpi-sub">${k.sub}</div></div>`).join('');

  $('kb-tbody').innerHTML = d.kb_group_rows.map(r=>{
    const cls = r.kb_pct>=60?'c-met':r.kb_pct>=30?'c-pend':'c-breach';
    const tops = r.top_kbs.map(k=>`<span class="chip c-purple" style="margin:1px;font-size:.65rem">${k.KB_ID} (${k.count})</span>`).join(' ') || '—';
    return `<tr>
      <td style="font-weight:700">${r.group}</td>
      <td style="font-family:'DM Mono',monospace">${r.total}</td>
      <td style="font-family:'DM Mono',monospace;color:var(--purple)">${r.with_kb}</td>
      <td style="font-family:'DM Mono',monospace;color:var(--red)">${r.without_kb}</td>
      <td><span class="chip ${cls}">${r.kb_pct}%</span></td>
      <td style="max-width:260px">${tops}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${r.kb_pct}%;background:var(--purple)"></div></div></div></td>
    </tr>`;
  }).join('');

  if (d.kb_no_kb_rows.length) {
    const cols = Object.keys(d.kb_no_kb_rows[0]);
    $('kb-no-thead').innerHTML = '<tr>'+cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')+'</tr>';
    $('kb-no-tbody').innerHTML = d.kb_no_kb_rows.map(r=>
      '<tr>'+cols.map(c=>`<td style="font-size:.76rem">${r[c]||'—'}</td>`).join('')+'</tr>'
    ).join('');
  }
}

// ── DATA INFO ─────────────────────────────────────────────────────────────
function buildInfo(d) {
  $('col-chips').innerHTML = d.detected_cols.map(c=>`
    <span style="background:var(--surf);border:1px solid var(--border);padding:3px 10px;border-radius:6px;
      font-size:.68rem;font-family:'DM Mono',monospace;color:var(--muted)">${c}</span>`).join('');
  $('mttr-logic-panel').innerHTML = `
    <strong style="color:var(--text)">Formula used:</strong><br>
    <span style="color:var(--blue);font-family:'DM Mono',monospace">${d.mttr_source}</span><br><br>
    <strong style="color:var(--text)">Priority:</strong><br>
    1. If <span style="color:var(--green)">SubmitDate</span> exists → <code style="color:var(--cyan)">LastResolvedDate − SubmitDate</code><br>
    2. Else → <code style="color:var(--cyan)">LastResolvedDate − ReportedDate</code><br><br>
    Negative values (data errors) are excluded.<br>
    Result unit: <strong style="color:var(--yellow)">Hours</strong>`;
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "YOUR_LINUX_IP"

    port = 5050
    print("\n" + "=" * 62)
    print("  ITIS Incident Analyzer  v3")
    print("=" * 62)
    print(f"  Local   :  http://localhost:{port}")
    print(f"  Remote  :  http://{lan_ip}:{port}   ← open this from Windows")
    print("=" * 62)
    print("  Formats : .xlsx  .xls  .csv")
    print("  New     : MTTR via SubmitDate · Alert tables · KB analysis")
    print("  Press Ctrl+C to stop")
    print("=" * 62 + "\n")
    app.run(debug=False, port=port, host="0.0.0.0", threaded=True)
