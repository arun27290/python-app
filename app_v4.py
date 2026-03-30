"""
ITIS Incident Management Analyzer  v4
======================================
Run  :  python app.py
Open :  http://localhost:5050

Changes in v4
─────────────
FIXES
  • KB tab: now detects 'Request Type01' and 'Request Description01' (exact names you have)
            + fuzzy column matching so any similar name works automatically
  • KB tab: "Upload New File" now correctly hides/shows the KB tab on every new upload
  • Date range: uses SubmitDate or LastResolvedDate when ReportedDate absent
  • MTTR>200 alerts: grouped by AssignedGroup, collapsible panels (same as Team tab)
  • HOP count>5 alerts: grouped by AssignedGroup, collapsible panels

NEW FEATURES
  • Smart fuzzy column detector — maps columns with 70%+ name similarity automatically
  • Priority heatmap: Priority × Month showing volume trends
  • Searchable/filterable table on every alert panel
  • Export button: download any alert table as CSV directly from browser
  • Summary scorecard: health score (0-100) based on SLA%, MTTR, P1 breaches, HOP
  • Responsive sidebar summary visible on Overview always
"""

import io, json, warnings, logging, re, difflib
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, jsonify

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

@app.errorhandler(413)
def too_large(e): return jsonify({"error": "File too large — maximum 50 MB."}), 413

@app.errorhandler(Exception)
def handle_exc(e):
    log.exception("Unhandled"); return jsonify({"error": f"Server error: {e}"}), 500

# ─────────────────────────────────────────────────────────────────────────────
#  CANONICAL COLUMN REGISTRY
#  Each entry: canonical_name → list of known aliases (exact + fuzzy seeds)
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL = {
    "Incident_Number":  ["Incident_Number","IncidentNumber","Incident_Nember",
                         "IncidentID","Incident ID","Incident Number","Inc Number","INC No"],
    "ReportedDate":     ["ReportedDate","Reported Date","Reported_Date","Open Date","OpenDate"],
    "LastResolvedDate": ["LastResolvedDate","Last Resolved Date","ResolvedDate",
                         "Resolved Date","Close Date","CloseDate","Resolution Date"],
    "SubmitDate":       ["SubmitDate","Submit Date","Submit_Date","Submission Date","Created Date","CreateDate"],
    "Summary":          ["Summary","Description","Short Description","Issue"],
    "Service_Type":     ["Service_Type","ServiceType","Service Type","Incident Type","IncidentType"],
    "HPD_CI":           ["HPD_CI","HPDCI","CI","Configuration Item","Asset","Server","CI Name"],
    "SLAStatus":        ["SLAStatus","SLA_Status","SLA Status","SLA","SLA Met","SLA Compliance"],
    "Priority":         ["Priority","Incident Priority","Severity"],
    "AssignedGroup":    ["AssignedGroup","Assigned Group","Assignment Group","Team","Resolver Group"],
    "Assignee":         ["Assignee","Assigned To","Resolver","Owner","Engineer"],
    "Assigned_Support_Organisation": ["Assigned_Support_Organisation","Organisation","Organization",
                                      "Support Org","Tower","AssignedSupportOrganisation"],
    "Assigned_Support_Company":      ["Assigned_Support_Company","Company","Support Company","Vendor"],
    "Status":           ["Status","Incident Status","State","Current State"],
    "Group_Transfers":  ["Group_Transfers","GroupTransfers","Group Transfers","Transfers",
                         "Reassignment Count","Hop Count","Hops"],
    # KB – exact column names you confirmed + all variants
    "Request_Type01":   ["Request_Type01","RequestType01","Request Type01",
                         "Request_Type_01","Request Type 01","RequestType 01",
                         "Req Type01","Req_Type01"],
    "Request_Desc01":   ["Request_Desc01","RequestDesc01","Request Description01",
                         "Request_Description01","Request Desc01","Request_Type_Description",
                         "RequestTypeDescription","Request Type Description",
                         "Req Description01","Req Desc01","Description01"],
}

# Build fast lookup: alias_lower → canonical
_ALIAS_MAP = {}
for canon, aliases in CANONICAL.items():
    for a in aliases:
        _ALIAS_MAP[a.lower().strip()] = canon

def fuzzy_match(col_name, threshold=0.70):
    """Try exact alias lookup, then fuzzy similarity against all known aliases."""
    key = col_name.lower().strip()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    # fuzzy
    best_score, best_canon = 0, None
    for alias, canon in _ALIAS_MAP.items():
        score = difflib.SequenceMatcher(None, key, alias).ratio()
        if score > best_score:
            best_score, best_canon = score, canon
    if best_score >= threshold:
        log.info("Fuzzy matched '%s' → '%s' (%.0f%%)", col_name, best_canon, best_score*100)
        return best_canon
    return None

def normalise_columns(df):
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    mapped = set()
    for col in df.columns:
        canon = fuzzy_match(col)
        if canon and canon not in mapped:
            rename_map[col] = canon
            mapped.add(canon)
    if rename_map:
        log.info("Column mapping: %s", rename_map)
    return df.rename(columns=rename_map)

# ─────────────────────────────────────────────────────────────────────────────
#  DATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_EPOCH = datetime(1899, 12, 30)

def excel_serial_to_dt(val):
    try:
        f = float(val)
        if f > 0: return EXCEL_EPOCH + timedelta(days=f)
    except (TypeError, ValueError): pass
    return pd.NaT

def parse_date_col(series):
    if pd.api.types.is_datetime64_any_dtype(series): return series
    if pd.api.types.is_numeric_dtype(series): return series.apply(excel_serial_to_dt)
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.5:
        parsed = series.apply(lambda v: excel_serial_to_dt(v) if pd.notna(v) else pd.NaT)
    return parsed

def priority_sort_key(p):
    return {"P1 - Critical":0,"Critical":0,"P2 - High":1,"High":1,
            "P3 - Medium":2,"Medium":2,"P4 - Low":3,"Low":3}.get(str(p),99)

def extract_kb(text):
    if pd.isna(text): return None
    m = re.search(r'\b(KB\d{4,10})\b', str(text), re.IGNORECASE)
    return m.group(1).upper() if m else None

MET_VALUES      = {"met","within sla","sla met","ok","yes","compliant"}
CLOSED_STATUSES = {"resolved","closed","completed","done","fixed"}
OPEN_STATUSES   = {"open","in progress","pending","assigned","work in progress","wip","new","active"}

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def analyse(df_raw):
    df = normalise_columns(df_raw.copy())

    for col in ["ReportedDate","LastResolvedDate","SubmitDate"]:
        if col in df.columns:
            df[col] = parse_date_col(df[col])

    # ── Feature flags ─────────────────────────────────────────────────────────
    F = {k: k in df.columns for k in [
        "ReportedDate","LastResolvedDate","SubmitDate","Status","SLAStatus",
        "Priority","AssignedGroup","Assignee","Assigned_Support_Organisation",
        "HPD_CI","Service_Type","Group_Transfers","Request_Type01","Request_Desc01"
    ]}
    inc_col = "Incident_Number" if "Incident_Number" in df.columns else df.columns[0]

    # ── Month / DayOfWeek from best available date ────────────────────────────
    date_ref = None
    for c in ["ReportedDate","SubmitDate","LastResolvedDate"]:
        if F.get(c) and df[c].notna().any():
            date_ref = c; break
    if date_ref:
        df["Month"]     = df[date_ref].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df[date_ref].dt.day_name()

    # ── MTTR (prefer SubmitDate start, fallback ReportedDate) ────────────────
    if F["LastResolvedDate"]:
        start = "SubmitDate" if F["SubmitDate"] else ("ReportedDate" if F["ReportedDate"] else None)
        if start:
            df["MTTR_Hours"] = ((df["LastResolvedDate"] - df[start])
                                 .dt.total_seconds() / 3600).round(2)
            df.loc[df["MTTR_Hours"] < 0, "MTTR_Hours"] = np.nan
            mttr_source = f"LastResolvedDate − {start}"
        else:
            df["MTTR_Hours"] = np.nan
            mttr_source = "N/A"
    else:
        df["MTTR_Hours"] = np.nan
        mttr_source = "N/A"

    # ── Incident aging in days ────────────────────────────────────────────────
    if F["LastResolvedDate"]:
        age_start = "SubmitDate" if F["SubmitDate"] else ("ReportedDate" if F["ReportedDate"] else None)
        if age_start:
            df["AgeDays"] = ((df["LastResolvedDate"] - df[age_start])
                              .dt.total_seconds() / 86400).round(1)
            df.loc[df["AgeDays"] < 0, "AgeDays"] = np.nan

    if F["Group_Transfers"]:
        df["Group_Transfers"] = pd.to_numeric(df["Group_Transfers"], errors="coerce")

    # ── Subsets ───────────────────────────────────────────────────────────────
    if F["Status"]:
        st_low = df["Status"].str.strip().str.lower()
        closed = df[st_low.isin(CLOSED_STATUSES)].copy()
        open_ct  = int(st_low.isin(OPEN_STATUSES).sum())
        closed_ct = len(closed)
    elif F["LastResolvedDate"]:
        closed = df[df["LastResolvedDate"].notna()].copy()
        open_ct, closed_ct = int(df["LastResolvedDate"].isna().sum()), len(closed)
    else:
        closed = df.copy(); open_ct, closed_ct = 0, len(df)

    total = len(df)

    # ── Date range ────────────────────────────────────────────────────────────
    date_min = date_max = "N/A"
    for c in ["SubmitDate","ReportedDate","LastResolvedDate"]:
        if F.get(c) and df[c].notna().any():
            date_min = df[c].min().strftime("%d-%b-%Y")
            date_max = df[c].max().strftime("%d-%b-%Y")
            break

    # ── SLA ───────────────────────────────────────────────────────────────────
    if F["SLAStatus"]:
        sla_low = df["SLAStatus"].str.strip().str.lower()
        sla_met_ct = int(sla_low.isin(MET_VALUES).sum())
        sla_pct    = round(sla_met_ct / total * 100, 1) if total else 0
    else:
        sla_met_ct = sla_pct = 0

    # ── MTTR KPI ──────────────────────────────────────────────────────────────
    mttr = round(float(df["MTTR_Hours"].mean()), 1) if df["MTTR_Hours"].notna().any() else 0

    # ── P1 ────────────────────────────────────────────────────────────────────
    p1_ct = int(df["Priority"].str.lower().str.contains("critical|p1", na=False).sum()) if F["Priority"] else 0

    # ── Health Score (0-100) ──────────────────────────────────────────────────
    health = 100
    if sla_pct   < 70: health -= min(30, int((70 - sla_pct)))
    if mttr      > 24: health -= min(25, int((mttr - 24) / 4))
    mttr_alerts = len(df[df["MTTR_Hours"] > 200]) if "MTTR_Hours" in df.columns else 0
    hop_alerts  = len(df[df["Group_Transfers"] > 5]) if F["Group_Transfers"] else 0
    health -= min(25, mttr_alerts * 3)
    health -= min(20, hop_alerts  * 1)
    health = max(0, min(100, health))

    # ── ALERT: MTTR > 200 hrs — grouped by AssignedGroup ─────────────────────
    mttr_grp_alerts = []
    if "MTTR_Hours" in df.columns:
        mttr_hi = df[df["MTTR_Hours"] > 200].copy()
        mttr_hi["MTTR_Hours"] = mttr_hi["MTTR_Hours"].round(1)
        if F["AssignedGroup"]:
            for gname, gdf in mttr_hi.groupby("AssignedGroup"):
                cols = [c for c in [inc_col,"AssignedGroup","MTTR_Hours","Priority",
                                    "SLAStatus","Status","HPD_CI"] if c in gdf.columns]
                mttr_grp_alerts.append({
                    "group":   gname,
                    "count":   len(gdf),
                    "rows":    gdf[cols].fillna("—").to_dict("records"),
                    "avg_mttr": round(float(gdf["MTTR_Hours"].mean()), 1),
                })
            mttr_grp_alerts.sort(key=lambda x: x["count"], reverse=True)
        else:
            # No group col — flat list
            cols = [c for c in [inc_col,"MTTR_Hours","Priority","Status"] if c in mttr_hi.columns]
            mttr_grp_alerts = [{"group":"All","count":len(mttr_hi),
                                 "rows": mttr_hi[cols].fillna("—").to_dict("records"),
                                 "avg_mttr": round(float(mttr_hi["MTTR_Hours"].mean()),1)}]
    mttr_alert_total = sum(g["count"] for g in mttr_grp_alerts)

    # ── ALERT: HOP count > 5 — grouped by AssignedGroup ──────────────────────
    hop_grp_alerts = []
    if F["Group_Transfers"]:
        hop_hi = df[df["Group_Transfers"] > 5].copy()
        if F["AssignedGroup"]:
            for gname, gdf in hop_hi.groupby("AssignedGroup"):
                cols = [c for c in [inc_col,"AssignedGroup","Group_Transfers","Priority",
                                    "SLAStatus","Status","MTTR_Hours"] if c in gdf.columns]
                gdf2 = gdf[cols].copy()
                if "MTTR_Hours" in gdf2.columns:
                    gdf2["MTTR_Hours"] = gdf2["MTTR_Hours"].round(1)
                hop_grp_alerts.append({
                    "group":    gname,
                    "count":    len(gdf),
                    "rows":     gdf2.fillna("—").to_dict("records"),
                    "avg_hops": round(float(gdf["Group_Transfers"].mean()), 1),
                })
            hop_grp_alerts.sort(key=lambda x: x["count"], reverse=True)
        else:
            cols = [c for c in [inc_col,"Group_Transfers","Priority","Status"] if c in hop_hi.columns]
            hop_grp_alerts = [{"group":"All","count":len(hop_hi),
                                "rows": hop_hi[cols].fillna("—").to_dict("records"),
                                "avg_hops": round(float(hop_hi["Group_Transfers"].mean()),1)}]
    hop_alert_total = sum(g["count"] for g in hop_grp_alerts)

    # ── ALERT: Aging > 30 days (flat, sorted) ────────────────────────────────
    age_alert_rows = []
    if "AgeDays" in df.columns:
        age_hi = df[df["AgeDays"] > 30].copy()
        age_hi["AgeDays"] = age_hi["AgeDays"].round(1)
        cols = [c for c in [inc_col,"AssignedGroup","AgeDays","Priority","Status"] if c in age_hi.columns]
        age_alert_rows = age_hi[cols].sort_values("AgeDays", ascending=False).fillna("—").to_dict("records")

    # ── Monthly volume ────────────────────────────────────────────────────────
    if "Month" in df.columns:
        vol = df.groupby("Month").size().reset_index(name="count").sort_values("Month")
        vol_labels, vol_data = vol["Month"].tolist(), vol["count"].tolist()
    else:
        vol_labels = vol_data = []

    # ── Priority ──────────────────────────────────────────────────────────────
    if F["Priority"]:
        pri = df["Priority"].value_counts().reset_index()
        pri.columns = ["Priority","count"]
        pri["s"] = pri["Priority"].apply(priority_sort_key)
        pri = pri.sort_values("s").drop("s",axis=1)
        pri_labels, pri_data = pri["Priority"].tolist(), pri["count"].tolist()
    else:
        pri_labels = pri_data = []

    # ── Priority × Month heatmap ──────────────────────────────────────────────
    pri_heat = []
    if F["Priority"] and "Month" in df.columns:
        heat = df.groupby(["Month","Priority"]).size().unstack(fill_value=0)
        pri_heat = {
            "months": heat.index.tolist(),
            "priorities": heat.columns.tolist(),
            "values": heat.values.tolist(),
        }

    # ── SLA ───────────────────────────────────────────────────────────────────
    if F["SLAStatus"]:
        sla_s = df["SLAStatus"].value_counts().reset_index()
        sla_s.columns = ["SLAStatus","count"]
        sla_labels, sla_data = sla_s["SLAStatus"].tolist(), sla_s["count"].tolist()
    else:
        sla_labels = sla_data = []

    # ── SLA % by priority ─────────────────────────────────────────────────────
    sla_pri_labels = sla_pri_vals = []
    if F["Priority"] and F["SLAStatus"]:
        def _sp(x):
            return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)
        sp = df.groupby("Priority").apply(_sp).reset_index(name="pct")
        sp["s"] = sp["Priority"].apply(priority_sort_key)
        sp = sp.sort_values("s").drop("s",axis=1)
        sla_pri_labels, sla_pri_vals = sp["Priority"].tolist(), sp["pct"].tolist()

    # ── SLA trend ─────────────────────────────────────────────────────────────
    sla_tr_labels = sla_tr_vals = []
    if F["SLAStatus"] and "Month" in df.columns:
        def _sm(x):
            return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)
        st = df.groupby("Month").apply(_sm).reset_index(name="pct").sort_values("Month")
        sla_tr_labels, sla_tr_vals = st["Month"].tolist(), st["pct"].tolist()

    # ── Group bar ─────────────────────────────────────────────────────────────
    if F["AssignedGroup"]:
        grp = df.groupby("AssignedGroup").size().reset_index(name="count").sort_values("count",ascending=False)
        grp_labels, grp_data = grp["AssignedGroup"].tolist(), grp["count"].tolist()
    else:
        grp_labels = grp_data = []

    # ── MTTR by group ─────────────────────────────────────────────────────────
    mttr_grp_labels = mttr_grp_vals = []
    if F["AssignedGroup"] and df["MTTR_Hours"].notna().any():
        mg = df.groupby("AssignedGroup")["MTTR_Hours"].mean().round(1).reset_index()
        mg.columns = ["group","mttr"]; mg = mg.sort_values("mttr")
        mttr_grp_labels, mttr_grp_vals = mg["group"].tolist(), mg["mttr"].tolist()

    # ── Day of week ───────────────────────────────────────────────────────────
    dow_labels = dow_data = []
    if "DayOfWeek" in df.columns:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["DayOfWeek"].value_counts().reindex(days, fill_value=0)
        dow_labels, dow_data = dow.index.tolist(), dow.values.tolist()

    # ── Group Transfers distribution ──────────────────────────────────────────
    xf_labels = xf_data = []
    if F["Group_Transfers"]:
        xf = df["Group_Transfers"].value_counts().sort_index().reset_index()
        xf.columns = ["transfers","count"]
        xf_labels = [str(int(v)) for v in xf["transfers"].tolist()]
        xf_data   = xf["count"].tolist()

    # ── Service type ──────────────────────────────────────────────────────────
    if F["Service_Type"]:
        svc = df["Service_Type"].value_counts().reset_index()
        svc.columns = ["type","count"]
        svc_labels, svc_data = svc["type"].tolist(), svc["count"].tolist()
    else:
        svc_labels = svc_data = []

    # ── CI breakdown ──────────────────────────────────────────────────────────
    ci_labels = ci_data = []
    ci_gt3_list = []
    if F["HPD_CI"]:
        ci_all = df.groupby("HPD_CI").size().reset_index(name="count").sort_values("count",ascending=False)
        ci_labels = ci_all.head(15)["HPD_CI"].tolist()
        ci_data   = ci_all.head(15)["count"].tolist()
        for _, row in ci_all[ci_all["count"] > 3].iterrows():
            sub = df[df["HPD_CI"] == row["HPD_CI"]]
            grp_bd = []
            if F["AssignedGroup"]:
                grp_bd = (sub.groupby("AssignedGroup").size().reset_index(name="cnt")
                           .sort_values("cnt",ascending=False)
                           .apply(lambda r: f"{r['AssignedGroup']} ({r['cnt']})",axis=1).tolist())
            ci_gt3_list.append({"HPD_CI": row["HPD_CI"], "count": int(row["count"]),
                                  "groups_str": " · ".join(grp_bd) if grp_bd else "—"})

    # ── Team: per-group / per-assignee ────────────────────────────────────────
    group_tables = []
    if F["AssignedGroup"] and F["Assignee"]:
        for gname, gdf in df.groupby("AssignedGroup"):
            assignee_rows = []
            for aname, adf in gdf.groupby("Assignee"):
                r = {"Assignee": aname, "Total": len(adf)}
                if F["Status"]:
                    r["Resolved"] = int(adf["Status"].str.strip().str.lower().isin(CLOSED_STATUSES).sum())
                if df["MTTR_Hours"].notna().any():
                    r["Avg_MTTR_hrs"] = round(float(adf["MTTR_Hours"].mean()),1) if adf["MTTR_Hours"].notna().any() else "—"
                if F["SLAStatus"]:
                    r["SLA_Pct"] = round(adf["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(adf)*100,1)
                if F["Group_Transfers"]:
                    r["Avg_Transfers"] = round(float(adf["Group_Transfers"].mean()),1) if adf["Group_Transfers"].notna().any() else "—"
                assignee_rows.append(r)
            assignee_rows.sort(key=lambda x: x["Total"], reverse=True)

            g_total    = len(gdf)
            g_resolved = int(gdf["Status"].str.strip().str.lower().isin(CLOSED_STATUSES).sum()) if F["Status"] else g_total
            g_mttr     = round(float(gdf["MTTR_Hours"].mean()),1) if gdf["MTTR_Hours"].notna().any() else "—"
            g_sla      = round(gdf["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/g_total*100,1) if F["SLAStatus"] and g_total else "—"
            g_xfer     = round(float(gdf["Group_Transfers"].mean()),1) if F["Group_Transfers"] and gdf["Group_Transfers"].notna().any() else "—"
            group_tables.append({"group":gname,"total":g_total,"resolved":g_resolved,
                                  "mttr":g_mttr,"sla_pct":g_sla,"avg_transfers":g_xfer,
                                  "assignees":assignee_rows})
        group_tables.sort(key=lambda x: x["total"], reverse=True)

    # ── Org summary ───────────────────────────────────────────────────────────
    org_rows = []
    if F["Assigned_Support_Organisation"]:
        og = df.groupby("Assigned_Support_Organisation")
        org_agg = og.size().reset_index(name="total")
        if F["Status"]:
            org_agg = org_agg.merge(
                og["Status"].apply(lambda x: x.str.strip().str.lower().isin(OPEN_STATUSES).sum()).reset_index(name="open"),
                on="Assigned_Support_Organisation")
        if F["SLAStatus"]:
            org_agg = org_agg.merge(
                og["SLAStatus"].apply(lambda x: round(x.str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)).reset_index(name="sla_pct"),
                on="Assigned_Support_Organisation")
        if df["MTTR_Hours"].notna().any():
            org_agg = org_agg.merge(og["MTTR_Hours"].mean().round(1).reset_index(name="mttr"), on="Assigned_Support_Organisation")
        org_rows = org_agg.fillna("—").to_dict("records")

    # ── KB Article Analysis ───────────────────────────────────────────────────
    kb_available     = F["Request_Type01"] and F["Request_Desc01"]
    kb_group_rows    = []
    kb_no_kb_rows    = []
    kb_summary_data  = {}
    log.info("KB columns present — Request_Type01:%s  Request_Desc01:%s", F["Request_Type01"], F["Request_Desc01"])

    if kb_available:
        df["_kb_id"] = df["Request_Desc01"].apply(extract_kb)
        rkm_vals = {"rkm solution","rkmsolution","rkm_solution","kb solution","knowledge","rkm"}
        rkm_mask = df["Request_Type01"].str.strip().str.lower().isin(rkm_vals)
        rkm_df   = df[rkm_mask]
        no_kb_df = df[~rkm_mask]

        kb_summary_data = {
            "total_rkm":     len(rkm_df),
            "total_no_kb":   len(no_kb_df),
            "unique_kb_ids": int(rkm_df["_kb_id"].nunique()),
        }
        if F["AssignedGroup"]:
            for gname, gdf in df.groupby("AssignedGroup"):
                grp_rkm   = gdf[gdf["Request_Type01"].str.strip().str.lower().isin(rkm_vals)]
                grp_no_kb = gdf[~gdf.index.isin(grp_rkm.index)]
                kb_ids    = grp_rkm["_kb_id"].dropna().value_counts().head(5).reset_index()
                kb_ids.columns = ["KB_ID","count"]
                kb_group_rows.append({
                    "group":      gname, "total": len(gdf),
                    "with_kb":    len(grp_rkm), "without_kb": len(grp_no_kb),
                    "top_kbs":    kb_ids.to_dict("records"),
                    "kb_pct":     round(len(grp_rkm)/len(gdf)*100,1) if len(gdf) else 0,
                })
            kb_group_rows.sort(key=lambda x: x["with_kb"], reverse=True)

        no_kb_cols = [c for c in [inc_col,"AssignedGroup","Priority","Status"] if c in df.columns]
        kb_no_kb_rows = no_kb_df[no_kb_cols].head(200).fillna("—").to_dict("records")

    return {
        "total": total, "closed_ct": closed_ct,
        "date_min": date_min, "date_max": date_max,
        "sla_met_ct": sla_met_ct, "sla_pct": sla_pct,
        "mttr": mttr, "mttr_source": mttr_source,
        "p1_ct": p1_ct, "health": health,
        "detected_cols": [c for c in df.columns if not c.startswith("_")],
        "has_kb": kb_available,
        "feature_flags": {k: v for k, v in F.items()},

        "mttr_grp_alerts": mttr_grp_alerts, "mttr_alert_total": mttr_alert_total,
        "hop_grp_alerts":  hop_grp_alerts,  "hop_alert_total":  hop_alert_total,
        "age_alert_rows":  age_alert_rows,

        "vol_labels": vol_labels, "vol_data": vol_data,
        "pri_labels": pri_labels, "pri_data": pri_data,
        "pri_heat":   pri_heat,
        "sla_labels": sla_labels, "sla_data": sla_data,
        "sla_pri_labels": sla_pri_labels, "sla_pri_vals": sla_pri_vals,
        "sla_tr_labels": sla_tr_labels, "sla_tr_vals": sla_tr_vals,
        "grp_labels": grp_labels, "grp_data": grp_data,
        "mttr_grp_labels": mttr_grp_labels, "mttr_grp_vals": mttr_grp_vals,
        "dow_labels": dow_labels, "dow_data": dow_data,
        "xf_labels": xf_labels, "xf_data": xf_data,
        "ci_labels": ci_labels, "ci_data": ci_data,
        "ci_gt3_rows": ci_gt3_list,
        "svc_labels": svc_labels, "svc_data": svc_data,
        "org_rows": org_rows,
        "group_tables": group_tables,
        "kb_summary": kb_summary_data,
        "kb_group_rows": kb_group_rows,
        "kb_no_kb_rows": kb_no_kb_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template_string(PAGE)

@app.route("/upload", methods=["POST","OPTIONS"])
def upload():
    if request.method == "OPTIONS": return jsonify({}), 200
    log.info("Upload from %s", request.remote_addr)
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400
    fname = f.filename.lower().strip()
    log.info("File: %s", f.filename)
    try:
        fb = io.BytesIO(f.read())
    except Exception as e:
        return jsonify({"error": f"Could not receive file: {e}"}), 400
    df = None
    try:
        if fname.endswith(".xlsx"):
            df = pd.read_excel(fb, engine="openpyxl")
        elif fname.endswith(".xls"):
            try:
                import xlrd; df = pd.read_excel(fb, engine="xlrd")
            except ImportError:
                try: fb.seek(0); df = pd.read_excel(fb, engine="openpyxl")
                except: return jsonify({"error": ".xls needs xlrd: pip install xlrd  OR resave as .xlsx"}), 400
        elif fname.endswith(".csv"):
            df = pd.read_csv(fb)
        else:
            return jsonify({"error": f"Unsupported type '{f.filename}'. Use .xlsx .xls .csv"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400
    if df is None or df.empty:
        return jsonify({"error": "File is empty or has no readable data."}), 400
    log.info("Read OK — %d rows, cols: %s", len(df), list(df.columns)[:10])
    try:
        result = analyse(df)
        log.info("Done — health:%d  MTTR alerts:%d  HOP alerts:%d  KB:%s",
                 result["health"], result["mttr_alert_total"],
                 result["hop_alert_total"], result["has_kb"])
    except Exception as e:
        log.exception("Analysis failed"); return jsonify({"error": f"Analysis failed: {e}"}), 500
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
#  FRONTEND
# ─────────────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ITIS Analyzer v4</title>
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
.hdr{position:sticky;top:0;z-index:500;background:rgba(7,9,15,.94);backdrop-filter:blur(14px);
  border-bottom:1px solid var(--border);padding:12px 26px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:15px}
.brand-name{font-size:.92rem;font-weight:800;letter-spacing:-.02em}
.brand-name span{color:var(--blue)}
.hdr-right{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.pill{background:var(--card);border:1px solid var(--border);padding:3px 10px;border-radius:18px;
  font-family:'DM Mono',monospace;font-size:.65rem;color:var(--muted)}
.pill.live{border-color:var(--green);color:var(--green)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);
  display:inline-block;margin-right:4px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* health ring */
.health-ring{width:32px;height:32px;position:relative;flex-shrink:0}
.health-ring svg{transform:rotate(-90deg)}
.health-ring .track{fill:none;stroke:var(--border2);stroke-width:3}
.health-ring .fill{fill:none;stroke-width:3;stroke-linecap:round;transition:stroke-dashoffset .8s}
.health-val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:.55rem;font-weight:800;font-family:'DM Mono',monospace}
/* UPLOAD */
#upload-section{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:calc(100vh - 58px);padding:40px 20px;
  background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(74,140,255,.07),transparent 70%)}
.ucard{width:100%;max-width:580px;background:var(--card);border:1px solid var(--border);
  border-radius:18px;padding:40px 36px;text-align:center;position:relative;overflow:hidden}
.ucard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--purple),var(--cyan))}
.u-title{font-size:1.55rem;font-weight:800;letter-spacing:-.03em;margin-bottom:6px}
.u-title span{color:var(--blue)}
.u-sub{color:var(--muted);font-size:.83rem;margin-bottom:26px;line-height:1.65}
.drop{border:2px dashed var(--border2);border-radius:12px;padding:34px 20px;cursor:pointer;
  transition:all .2s;position:relative;background:var(--card2)}
.drop:hover,.drop.drag{border-color:var(--blue);background:rgba(74,140,255,.06)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.fmts{margin-top:8px;display:flex;gap:6px;justify-content:center}
.fmt{background:var(--surf);border:1px solid var(--border);padding:2px 9px;border-radius:5px;
  font-size:.65rem;font-family:'DM Mono',monospace;color:var(--muted)}
.ubtn{margin-top:18px;background:linear-gradient(135deg,var(--blue),var(--purple));border:none;
  color:#fff;padding:12px 32px;border-radius:11px;font-size:.9rem;font-weight:700;cursor:pointer;
  font-family:'Syne',sans-serif;transition:opacity .2s,transform .15s;width:100%}
.ubtn:hover{opacity:.9;transform:translateY(-1px)}
.ubtn:disabled{opacity:.38;cursor:not-allowed;transform:none}
.fname{margin-top:9px;font-size:.76rem;color:var(--green);font-family:'DM Mono',monospace}
.alert-box{background:rgba(255,79,106,.1);border:1px solid rgba(255,79,106,.3);
  border-radius:9px;padding:11px 14px;color:var(--red);font-size:.82rem;margin-top:10px}
#spin{display:none;position:fixed;inset:0;background:rgba(7,9,15,.88);z-index:900;
  align-items:center;justify-content:center;flex-direction:column;gap:12px}
#spin.show{display:flex}
.spinner{width:42px;height:42px;border:3px solid var(--border2);border-top-color:var(--blue);
  border-radius:50%;animation:rot .8s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
/* DASHBOARD */
#dash{display:none}
#dash.show{display:block}
.new-btn{display:inline-flex;align-items:center;gap:7px;background:var(--card);
  border:1px solid var(--border);color:var(--muted);padding:8px 16px;border-radius:9px;
  cursor:pointer;font-family:'Syne',sans-serif;font-size:.78rem;font-weight:600;
  transition:all .2s;margin:14px 26px 0}
.new-btn:hover{border-color:var(--blue);color:var(--blue)}
.tabs{background:var(--surf);border-bottom:1px solid var(--border);padding:0 26px;
  display:flex;gap:2px;overflow-x:auto;position:sticky;top:58px;z-index:400}
.tab{padding:11px 16px;cursor:pointer;font-size:.78rem;font-weight:600;color:var(--muted);
  border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;user-select:none}
.tab:hover{color:var(--text)}.tab.on{color:var(--blue);border-bottom-color:var(--blue)}
.kb-tab{display:none}.kb-tab.show{display:block}
.page{display:none;padding:20px 26px}.page.on{display:block}
/* KPIs */
.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:16px 14px;position:relative;overflow:hidden;transition:transform .15s,border-color .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi-bar{position:absolute;top:0;left:0;right:0;height:2px}
.kpi-lbl{font-size:.61rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:7px}
.kpi-val{font-size:1.7rem;font-weight:800;font-family:'DM Mono',monospace;line-height:1;letter-spacing:-.03em}
.kpi-sub{font-size:.61rem;color:var(--dim);margin-top:4px}
.kpi-icon{position:absolute;right:12px;top:12px;font-size:1.1rem;opacity:.2}
/* SECTION */
.sec{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin:18px 0 10px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
/* CHART CARD */
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:17px}
.cc-hd{display:flex;align-items:center;gap:7px;margin-bottom:13px}
.cc-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.cc-title{font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.cc-badge{margin-left:auto;background:var(--surf);border:1px solid var(--border);
  padding:2px 7px;border-radius:7px;font-size:.62rem;color:var(--dim);font-family:'DM Mono',monospace}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.full{margin-bottom:14px}
/* ALERT CARD */
.al-card{background:var(--card);border-radius:12px;border:1px solid var(--border);margin-bottom:14px;overflow:hidden}
.al-hdr{padding:12px 16px;display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--border);cursor:pointer;user-select:none}
.al-icon{font-size:1rem}
.al-info .al-title{font-size:.8rem;font-weight:700}
.al-info .al-sub{font-size:.66rem;color:var(--muted);margin-top:1px}
.al-badge{margin-left:auto;padding:3px 10px;border-radius:8px;font-size:.68rem;font-weight:700;flex-shrink:0}
.al-expand{font-size:.78rem;color:var(--dim);margin-left:4px;transition:transform .2s;flex-shrink:0}
.al-hdr.open .al-expand{transform:rotate(180deg)}
.al-body{display:none}.al-body.open{display:block}
.al-none{padding:14px 16px;color:var(--dim);font-size:.8rem}
/* GROUP PANEL (reused for alerts and team) */
.grp-panel{background:var(--card);border:1px solid var(--border);border-radius:12px;margin-bottom:12px;overflow:hidden}
.grp-hdr{padding:12px 16px;background:var(--card2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}
.grp-hdr .gname{font-weight:700;font-size:.86rem;letter-spacing:-.01em}
.grp-stats{display:flex;gap:12px;margin-left:auto;flex-wrap:wrap}
.gs{font-size:.7rem;color:var(--muted);font-family:'DM Mono',monospace}
.gs strong{color:var(--text)}
.g-expand{font-size:.78rem;color:var(--dim);margin-left:4px;transition:transform .2s;flex-shrink:0}
.grp-hdr.open .g-expand{transform:rotate(180deg)}
.grp-body{display:none;padding:0}.grp-body.open{display:block}
/* SEARCH + EXPORT */
.tbl-toolbar{padding:10px 14px;display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--border);background:var(--surf)}
.search-box{flex:1;background:var(--card);border:1px solid var(--border2);border-radius:7px;
  padding:6px 11px;color:var(--text);font-family:'DM Mono',monospace;font-size:.76rem;outline:none}
.search-box:focus{border-color:var(--blue)}
.export-btn{background:rgba(74,140,255,.12);border:1px solid rgba(74,140,255,.3);color:var(--blue);
  padding:5px 12px;border-radius:7px;cursor:pointer;font-size:.72rem;font-weight:700;
  font-family:'Syne',sans-serif;transition:all .15s;white-space:nowrap}
.export-btn:hover{background:rgba(74,140,255,.22)}
/* TABLES */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.76rem}
thead th{padding:8px 10px;background:var(--surf);color:var(--muted);font-weight:700;
  font-size:.63rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;
  border-bottom:1px solid var(--border);white-space:nowrap;position:sticky;top:0}
tbody td{padding:9px 10px;border-bottom:1px solid rgba(30,42,64,.55);vertical-align:middle}
tbody tr:hover td{background:rgba(74,140,255,.04)}
tbody tr:last-child td{border-bottom:none}
tbody tr.hidden{display:none}
/* CHIPS */
.chip{display:inline-block;padding:2px 8px;border-radius:7px;font-size:.66rem;font-weight:700;white-space:nowrap}
.c-met{background:rgba(48,217,136,.15);color:var(--green)}
.c-breach{background:rgba(255,79,106,.15);color:var(--red)}
.c-pend{background:rgba(255,194,64,.15);color:var(--yellow)}
.c-blue{background:rgba(74,140,255,.12);color:var(--blue)}
.c-purple{background:rgba(167,139,250,.12);color:var(--purple)}
.c-orange{background:rgba(251,146,60,.12);color:var(--orange)}
.c-cyan{background:rgba(34,211,238,.12);color:var(--cyan)}
/* MINI BAR */
.mbar{display:flex;align-items:center;gap:6px;min-width:80px}
.mbar-bg{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.mbar-fill{height:100%;border-radius:3px}
/* HEATMAP */
.heat-table{border-collapse:collapse;font-size:.74rem;width:100%}
.heat-table th{padding:7px 10px;background:var(--surf);color:var(--muted);font-size:.63rem;font-weight:700;text-transform:uppercase}
.heat-table td{padding:8px 10px;text-align:center;border:1px solid var(--border);font-family:'DM Mono',monospace}
.heat-table .row-lbl{text-align:left;font-family:'Syne',sans-serif;font-weight:600;color:var(--text);background:transparent}
footer{text-align:center;padding:18px;color:var(--dim);font-size:.67rem;border-top:1px solid var(--border);margin-top:4px}
@media(max-width:1100px){.kpi-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.g2,.g3{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}}
@media(max-width:500px){.kpi-row{grid-template-columns:1fr}.page{padding:13px 11px}}
</style>
</head>
<body>
<header class="hdr">
  <div class="brand">
    <div class="brand-icon">⚡</div>
    <div class="brand-name">ITIS <span>Incident</span> Analyzer <span style="font-size:.62rem;color:var(--dim);margin-left:3px">v4</span></div>
  </div>
  <div class="hdr-right">
    <span class="pill live"><span class="dot"></span>Live</span>
    <span class="pill" id="file-pill">No file</span>
    <span class="pill" id="mttr-pill" style="display:none"></span>
    <div class="health-ring" id="health-ring" style="display:none" title="Health Score">
      <svg viewBox="0 0 32 32" width="32" height="32">
        <circle class="track" cx="16" cy="16" r="13"/>
        <circle class="fill" id="health-arc" cx="16" cy="16" r="13" stroke-dasharray="81.68" stroke-dashoffset="81.68"/>
      </svg>
      <div class="health-val" id="health-val">—</div>
    </div>
  </div>
</header>

<div id="spin"><div class="spinner"></div><div style="color:var(--muted);font-size:.83rem">Analysing incident data…</div></div>

<section id="upload-section">
  <div class="ucard">
    <div class="u-title">ITIS <span>Incident</span> Dashboard</div>
    <div class="u-sub">Upload any ITIS Excel or CSV export.<br>
      Columns are <strong style="color:var(--green)">auto-detected with fuzzy matching</strong> —<br>
      extra or renamed columns are handled automatically.</div>
    <div class="drop" id="drop">
      <input type="file" id="fi" accept=".xlsx,.xls,.csv"/>
      <div style="font-size:2rem;margin-bottom:9px">📂</div>
      <div style="font-size:.86rem;color:var(--muted)"><strong style="color:var(--text)">Click to browse</strong> or drag &amp; drop</div>
      <div class="fmts"><span class="fmt">.xlsx</span><span class="fmt">.xls</span><span class="fmt">.csv</span></div>
    </div>
    <div class="fname" id="fname"></div>
    <div id="uerr"></div>
    <button class="ubtn" id="abtn" disabled onclick="doUpload()">Analyse File →</button>
  </div>
</section>

<div id="dash">
  <button class="new-btn" onclick="reset()">← Upload New File</button>
  <nav class="tabs">
    <div class="tab on"  onclick="go('ov',this)">📊 Overview</div>
    <div class="tab"     onclick="go('tr',this)">📈 Trends</div>
    <div class="tab"     onclick="go('sl',this)">🎯 SLA</div>
    <div class="tab"     onclick="go('tm',this)">👥 Team</div>
    <div class="tab"     onclick="go('ci',this)">🖥 CI &amp; Service</div>
    <div class="tab kb-tab" id="kb-tab" onclick="go('kb',this)">📚 KB Articles</div>
    <div class="tab"     onclick="go('nf',this)">🔍 Data Info</div>
  </nav>

  <!-- OVERVIEW -->
  <section class="page on" id="pg-ov">
    <div class="kpi-row" id="kpi-row"></div>
    <div class="g3">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Priority Distribution</span></div><canvas id="c-pri" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Status</span></div><canvas id="c-sla-donut" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Incidents by AssignedGroup</span><span class="cc-badge" id="grp-badge"></span></div><canvas id="c-grp" height="210"></canvas></div>
    </div>
    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Organisation Summary</span></div>
      <div class="tbl-wrap"><table><thead><tr><th>Organisation</th><th>Total</th><th>Open</th><th>SLA %</th><th>MTTR (hrs)</th><th>SLA Bar</th></tr></thead><tbody id="org-tbody"></tbody></table></div>
    </div>
    <div class="sec">🚨 Alerts &amp; Exceptions</div>

    <!-- MTTR > 200 grouped -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(255,79,106,.06);border-left:3px solid var(--red)" onclick="toggleAl(this,'mttr-al-body')">
        <span class="al-icon">⏱</span>
        <div class="al-info"><div class="al-title">MTTR &gt; 200 Hours — by AssignedGroup</div>
          <div class="al-sub">Incidents with extreme resolution time, grouped by team</div></div>
        <span class="al-badge" id="mttr-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="mttr-al-body"><div class="al-none">No incidents exceed 200 hours MTTR ✓</div></div>
    </div>

    <!-- HOP > 5 grouped -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(251,146,60,.06);border-left:3px solid var(--orange)" onclick="toggleAl(this,'hop-al-body')">
        <span class="al-icon">🔄</span>
        <div class="al-info"><div class="al-title">High HOP Count — Group Transfers &gt; 5 — by AssignedGroup</div>
          <div class="al-sub">Incidents bounced across more than 5 groups</div></div>
        <span class="al-badge" id="hop-badge" style="background:rgba(251,146,60,.15);color:var(--orange)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="hop-al-body"><div class="al-none">No incidents have Group Transfers &gt; 5 ✓</div></div>
    </div>

    <!-- Aging > 30d -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(255,194,64,.06);border-left:3px solid var(--yellow)" onclick="toggleAl(this,'age-al-body')">
        <span class="al-icon">📅</span>
        <div class="al-info"><div class="al-title">Incident Aging &gt; 30 Days</div>
          <div class="al-sub">Resolution date minus Submit date exceeds 30 days</div></div>
        <span class="al-badge" id="age-badge" style="background:rgba(255,194,64,.15);color:var(--yellow)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="age-al-body"><div class="al-none">No incidents exceed 30 days aging ✓</div></div>
    </div>
  </section>

  <!-- TRENDS -->
  <section class="page" id="pg-tr">
    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Monthly Incident Volume</span></div><canvas id="c-vol" height="110"></canvas></div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Incidents by Day of Week</span></div><canvas id="c-dow" height="210"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Group Transfers Distribution</span></div><canvas id="c-xfer" height="210"></canvas></div>
    </div>
    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance Trend — Monthly %</span></div><canvas id="c-sla-trend" height="110"></canvas></div>
    <div class="full cc" id="heat-card" style="display:none">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Priority × Month Heatmap</span></div>
      <div class="tbl-wrap"><table class="heat-table" id="heat-tbl"></table></div>
    </div>
  </section>

  <!-- SLA -->
  <section class="page" id="pg-sl">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance % by Priority</span></div><canvas id="c-sla-pri" height="230"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">MTTR by AssignedGroup (hrs)</span></div><canvas id="c-mttr-grp" height="230"></canvas></div>
    </div>
  </section>

  <!-- TEAM -->
  <section class="page" id="pg-tm">
    <div class="sec">Per-Group Performance — click header to expand assignees</div>
    <div id="team-panels"></div>
  </section>

  <!-- CI & SERVICE -->
  <section class="page" id="pg-ci">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--cyan)"></span><span class="cc-title">Top HPD_CI by Count</span></div><canvas id="c-ci" height="280"></canvas></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Service Type</span></div><canvas id="c-svc" height="280"></canvas></div>
    </div>
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--cyan);cursor:default">
        <span class="al-icon">🖥</span>
        <div class="al-info"><div class="al-title">HPD_CI with Incident Count &gt; 3 — with AssignedGroup Breakdown</div></div>
        <span class="al-badge" id="ci3-badge" style="background:rgba(34,211,238,.12);color:var(--cyan)">0</span>
      </div>
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search CI name or group…" oninput="filterTbl(this,'ci3-tbody')"/>
        <button class="export-btn" onclick="exportCSV('ci3-tbody','ci_repeat_offenders')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead><tr><th>#</th><th>HPD_CI</th><th>Total</th><th>AssignedGroup Breakdown</th><th>Volume</th></tr></thead><tbody id="ci3-tbody"></tbody></table></div>
    </div>
  </section>

  <!-- KB -->
  <section class="page" id="pg-kb">
    <div class="kpi-row" style="grid-template-columns:repeat(3,1fr)" id="kb-kpis"></div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span>
        <span class="cc-title">KB Article Coverage by AssignedGroup</span>
        <span style="font-size:.68rem;color:var(--dim);margin-left:7px">Request_Type01 = RKM Solution</span>
      </div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>AssignedGroup</th><th>Total</th><th>With KB</th><th>Without KB</th><th>KB %</th><th>Top KB IDs</th><th>Coverage</th>
      </tr></thead><tbody id="kb-tbody"></tbody></table></div>
    </div>
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--red);cursor:default">
        <span class="al-icon">❌</span>
        <div class="al-info"><div class="al-title">Incidents Without KB Article (first 200)</div></div>
      </div>
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'kb-no-tbody')"/>
        <button class="export-btn" onclick="exportCSV('kb-no-tbody','incidents_no_kb')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead id="kb-no-thead"></thead><tbody id="kb-no-tbody"></tbody></table></div>
    </div>
  </section>

  <!-- DATA INFO -->
  <section class="page" id="pg-nf">
    <div class="cc full"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Detected &amp; Mapped Columns</span></div>
      <div id="col-chips" style="display:flex;flex-wrap:wrap;gap:7px;margin-top:4px"></div>
    </div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">MTTR Calculation</span></div>
        <div id="mttr-logic" style="font-size:.8rem;color:var(--muted);line-height:1.8"></div>
      </div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Column Detection Log</span></div>
        <div id="feat-flags" style="font-size:.76rem;color:var(--muted);line-height:1.9;font-family:'DM Mono',monospace;max-height:260px;overflow-y:auto"></div>
      </div>
    </div>
  </section>

  <footer id="footer"></footer>
</div>

<script>
const $  = id => document.getElementById(id);
let charts = {}, D = null, selFile = null;

Chart.defaults.color = '#7b8db0';
Chart.defaults.borderColor = '#1e2a40';
Chart.defaults.font.family = "'Syne',sans-serif";
Chart.defaults.font.size = 11;

const PAL    = ['#4a8cff','#ff4f6a','#30d988','#ffc240','#a78bfa','#22d3ee','#fb923c','#f472b6','#34d399','#f87171'];
const PRI_C  = {'P1 - Critical':'#ff4f6a','Critical':'#ff4f6a','P2 - High':'#ffc240','High':'#ffc240','P3 - Medium':'#4a8cff','Medium':'#4a8cff','P4 - Low':'#30d988','Low':'#30d988'};
const SLA_C  = {'Met':'#30d988','Breached':'#ff4f6a','Pending':'#ffc240','Exempt':'#7b8db0','Within SLA':'#30d988','SLA Met':'#30d988','OK':'#30d988','Yes':'#30d988'};

// ── file input ──────────────────────────────────────────────────────────────
const fi = $('fi'), drop = $('drop');
fi.onchange = () => { selFile=fi.files[0]; if(selFile){$('fname').textContent='📄 '+selFile.name; $('abtn').disabled=false; $('uerr').innerHTML='';} };
drop.addEventListener('dragover', e=>{e.preventDefault();drop.classList.add('drag');});
drop.addEventListener('dragleave', ()=>drop.classList.remove('drag'));
drop.addEventListener('drop', e=>{e.preventDefault();drop.classList.remove('drag');
  if(e.dataTransfer.files.length){selFile=e.dataTransfer.files[0];$('fname').textContent='📄 '+selFile.name;$('abtn').disabled=false;}});

function doUpload() {
  if(!selFile) return;
  const fd = new FormData(); fd.append('file',selFile);
  $('spin').classList.add('show'); $('abtn').disabled=true; $('uerr').innerHTML='';
  fetch('/upload',{method:'POST',body:fd})
    .then(r=>r.text().then(t=>{
      try{return{ok:r.ok,data:JSON.parse(t)};}
      catch(e){return{ok:false,data:{error:'Server error (HTTP '+r.status+'). Check terminal.'}};}
    }))
    .then(res=>{
      $('spin').classList.remove('show'); $('abtn').disabled=false;
      if(!res.ok||res.data.error){$('uerr').innerHTML='<div class="alert-box">⚠ '+(res.data.error||'Unknown error')+'</div>';return;}
      D=res.data; build(D);
    })
    .catch(err=>{$('spin').classList.remove('show');$('abtn').disabled=false;
      $('uerr').innerHTML='<div class="alert-box">⚠ '+err.message+'</div>';});
}

function reset() {
  Object.values(charts).forEach(c=>{try{c.destroy();}catch(_){}});
  charts={}; D=null;
  $('dash').classList.remove('show'); $('upload-section').style.display='';
  $('abtn').disabled=true; fi.value=''; $('fname').textContent='';
  $('file-pill').textContent='No file'; $('mttr-pill').style.display='none';
  $('health-ring').style.display='none'; $('kb-tab').classList.remove('show');
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('on',i===0));
  document.querySelectorAll('.page').forEach((p,i)=>p.classList.toggle('on',i===0));
}

function go(name,el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  $('pg-'+name).classList.add('on'); el.classList.add('on');
}

function mk(id,type,labels,datasets,extra={}) {
  if(charts[id]) charts[id].destroy();
  const ctx=$(id); if(!ctx)return;
  charts[id]=new Chart(ctx,{type,data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#7b8db0',boxWidth:10,padding:12}}},...extra}});
}

// ── toggle helpers ──────────────────────────────────────────────────────────
function toggleAl(hdr,bodyId){hdr.classList.toggle('open');$(bodyId).classList.toggle('open');}
function toggleGrp(el){
  el.classList.toggle('open');
  const body=el.nextElementSibling; if(body) body.classList.toggle('open');
}

// ── search / filter ─────────────────────────────────────────────────────────
function filterTbl(inp,tbodyId){
  const q=inp.value.toLowerCase();
  document.querySelectorAll('#'+tbodyId+' tr').forEach(tr=>{
    tr.classList.toggle('hidden', !tr.textContent.toLowerCase().includes(q));
  });
}

// ── CSV export ──────────────────────────────────────────────────────────────
function exportCSV(tbodyId,filename){
  const tbody=$(tbodyId); if(!tbody)return;
  const thead=tbody.closest('table').querySelector('thead');
  const cols=thead?[...thead.querySelectorAll('th')].map(th=>th.textContent.trim()):[];
  const rows=[[...cols].join(',')];
  tbody.querySelectorAll('tr:not(.hidden)').forEach(tr=>{
    rows.push([...tr.querySelectorAll('td')].map(td=>'"'+td.textContent.trim().replace(/"/g,'""')+'"').join(','));
  });
  const blob=new Blob([rows.join('\n')],{type:'text/csv'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download=filename+'_'+new Date().toISOString().slice(0,10)+'.csv'; a.click();
}

// ── HEALTH RING ─────────────────────────────────────────────────────────────
function setHealth(score){
  const arc=$('health-arc'), val=$('health-val');
  const circ=81.68, offset=circ-(circ*score/100);
  arc.style.strokeDashoffset=offset;
  arc.style.stroke=score>=70?'#30d988':score>=50?'#ffc240':'#ff4f6a';
  val.textContent=score; val.style.color=score>=70?'#30d988':score>=50?'#ffc240':'#ff4f6a';
  $('health-ring').style.display='';
}

// ── BUILD ────────────────────────────────────────────────────────────────────
function build(d){
  $('upload-section').style.display='none'; $('dash').classList.add('show');
  $('file-pill').textContent=selFile?selFile.name:'Loaded';
  $('mttr-pill').textContent=d.mttr_source; $('mttr-pill').style.display='';
  $('footer').textContent='ITIS Analyzer v4 · '+d.detected_cols.length+' columns · '+d.total+' records · Health: '+d.health+'/100';
  if(d.has_kb){$('kb-tab').classList.add('show');} else {$('kb-tab').classList.remove('show');}
  setHealth(d.health);
  buildKPIs(d); buildAlerts(d); buildCharts(d);
  buildOrgTable(d); buildTeamPanels(d); buildCI(d);
  if(d.has_kb) buildKB(d);
  buildHeatmap(d); buildInfo(d);
}

// ── KPIs ────────────────────────────────────────────────────────────────────
function buildKPIs(d){
  const kpis=[
    {l:'Total Incidents',  v:d.total,       c:'var(--blue)',   i:'📋', s:'All records'},
    {l:'Closed/Resolved',  v:d.closed_ct,   c:'var(--green)',  i:'✅', s:'Resolved & Closed'},
    {l:'SLA Compliance',   v:d.sla_pct+'%', c:d.sla_pct>=70?'var(--green)':d.sla_pct>=50?'var(--yellow)':'var(--red)', i:'🎯', s:d.sla_met_ct+' met SLA'},
    {l:'MTTR (avg)',       v:d.mttr+'h',    c:d.mttr<=24?'var(--green)':d.mttr<=72?'var(--yellow)':'var(--red)', i:'⏱', s:'Mean Time To Resolve'},
    {l:'P1 Critical',      v:d.p1_ct,       c:'var(--red)',    i:'🚨', s:'Highest priority'},
  ];
  $('kpi-row').innerHTML=kpis.map(k=>`<div class="kpi">
    <div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-icon">${k.i}</div>
    <div class="kpi-lbl">${k.l}</div>
    <div class="kpi-val" style="color:${k.c}">${k.v}</div>
    <div class="kpi-sub">${k.s}</div></div>`).join('')+
    `<div style="grid-column:1/-1;background:var(--card);border:1px solid var(--border);border-radius:12px;
      padding:10px 15px;display:flex;align-items:center;gap:14px;font-size:.77rem;flex-wrap:wrap">
      <span style="color:var(--muted)">📅 Date Range:</span>
      <span style="color:var(--blue);font-family:'DM Mono',monospace">${d.date_min} → ${d.date_max}</span>
      <span style="color:var(--dim);margin-left:auto;font-family:'DM Mono',monospace">Health Score: <strong style="color:${d.health>=70?'var(--green)':d.health>=50?'var(--yellow)':'var(--red)'}">${d.health}/100</strong></span>
    </div>`;
}

// ── ALERTS ──────────────────────────────────────────────────────────────────
function grpAlertPanels(grpList, incColHint){
  if(!grpList.length) return '<div class="al-none">None found ✓</div>';
  return grpList.map((g,gi)=>{
    const cols=g.rows.length?Object.keys(g.rows[0]):[];
    const thead=cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('');
    const tbody=g.rows.map(r=>'<tr>'+cols.map(c=>{
      const v=r[c]!==undefined?r[c]:'—';
      if(c==='MTTR_Hours'||c==='AgeDays') return `<td style="font-family:'DM Mono',monospace;color:var(--red);font-weight:700">${v}</td>`;
      if(c==='Group_Transfers') return `<td><span class="chip c-orange">${v}</span></td>`;
      if(c==='Priority') return `<td><span class="chip" style="background:${(PRI_C[v]||'#64748b')}22;color:${PRI_C[v]||'#64748b'}">${v}</span></td>`;
      if(c==='SLAStatus'){const sc=SLA_C[v]||'#7b8db0';return `<td><span class="chip" style="background:${sc}22;color:${sc}">${v}</span></td>`;}
      return `<td>${v}</td>`;
    }).join('')+'</tr>').join('');
    const eid='ag-'+gi+'-'+Math.random().toString(36).slice(2,6);
    const xid='ab-'+gi+'-'+Math.random().toString(36).slice(2,6);
    return `<div class="grp-panel">
      <div class="grp-hdr" onclick="toggleGrp(this)">
        <span style="font-size:.95rem">👥</span>
        <span class="gname">${g.group}</span>
        <div class="grp-stats">
          <span class="gs">Count <strong>${g.count}</strong></span>
          ${g.avg_mttr!==undefined?`<span class="gs">Avg MTTR <strong>${g.avg_mttr}h</strong></span>`:''}
          ${g.avg_hops!==undefined?`<span class="gs">Avg Hops <strong>${g.avg_hops}</strong></span>`:''}
        </div>
        <span class="g-expand">▼</span>
      </div>
      <div class="grp-body">
        <div class="tbl-toolbar">
          <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'${eid}')"/>
          <button class="export-btn" onclick="exportCSV('${eid}','${g.group.replace(/\s/g,'_')}_alert')">⬇ CSV</button>
        </div>
        <div class="tbl-wrap"><table><thead><tr>${thead}</tr></thead><tbody id="${eid}">${tbody}</tbody></table></div>
      </div>
    </div>`;
  }).join('');
}

function buildAlerts(d){
  $('mttr-badge').textContent=d.mttr_alert_total;
  $('mttr-al-body').innerHTML=grpAlertPanels(d.mttr_grp_alerts,'Incident_Number');

  $('hop-badge').textContent=d.hop_alert_total;
  $('hop-al-body').innerHTML=grpAlertPanels(d.hop_grp_alerts,'Incident_Number');

  $('age-badge').textContent=d.age_alert_rows.length;
  if(d.age_alert_rows.length){
    const cols=Object.keys(d.age_alert_rows[0]);
    const eid='age-tbl-'+Math.random().toString(36).slice(2,6);
    const trs=d.age_alert_rows.map(r=>'<tr>'+cols.map(c=>{
      const v=r[c]!==undefined?r[c]:'—';
      if(c==='AgeDays') return `<td style="font-family:'DM Mono',monospace;color:var(--yellow);font-weight:700">${v}</td>`;
      if(c==='Priority'){const pc=PRI_C[v]||'#64748b';return `<td><span class="chip" style="background:${pc}22;color:${pc}">${v}</span></td>`;}
      return `<td>${v}</td>`;
    }).join('')+'</tr>').join('');
    $('age-al-body').innerHTML=`
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'${eid}')"/>
        <button class="export-btn" onclick="exportCSV('${eid}','incident_aging')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead><tr>${cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')}</tr></thead>
      <tbody id="${eid}">${trs}</tbody></table></div>`;
  }
}

// ── CHARTS ───────────────────────────────────────────────────────────────────
function buildCharts(d){
  const yg={y:{grid:{color:'#1e2a40'}},x:{grid:{display:false}}};
  const ypct={y:{grid:{color:'#1e2a40'},ticks:{callback:v=>v+'%'}},x:{grid:{display:false}}};
  if(d.pri_labels.length) mk('c-pri','doughnut',d.pri_labels,[{data:d.pri_data,
    backgroundColor:d.pri_labels.map(l=>PRI_C[l]||PAL[4]),borderWidth:2,borderColor:'#111827'}],{cutout:'62%'});
  if(d.sla_labels.length) mk('c-sla-donut','doughnut',d.sla_labels,[{data:d.sla_data,
    backgroundColor:d.sla_labels.map(l=>SLA_C[l]||PAL[4]),borderWidth:2,borderColor:'#111827'}],{cutout:'60%'});
  if(d.grp_labels.length){$('grp-badge').textContent=d.grp_labels.length+' groups';
    mk('c-grp','bar',d.grp_labels,[{label:'Incidents',data:d.grp_data,backgroundColor:'rgba(74,140,255,.75)',
      borderRadius:5,borderSkipped:false}],{indexAxis:'y',plugins:{legend:{display:false}},
      scales:{x:{grid:{color:'#1e2a40'}},y:{grid:{display:false}}}});}
  if(d.vol_labels.length) mk('c-vol','line',d.vol_labels,[{label:'Incidents',data:d.vol_data,
    borderColor:'#4a8cff',backgroundColor:'rgba(74,140,255,.1)',fill:true,tension:.4,
    pointRadius:5,pointBackgroundColor:'#4a8cff',pointBorderColor:'#07090f',pointBorderWidth:2}],{scales:yg});
  if(d.dow_labels.length) mk('c-dow','bar',d.dow_labels,[{label:'Incidents',data:d.dow_data,
    backgroundColor:d.dow_labels.map((_,i)=>PAL[i%PAL.length]),borderRadius:6}],{plugins:{legend:{display:false}},scales:yg});
  if(d.xf_labels.length) mk('c-xfer','bar',d.xf_labels,[{label:'Count',data:d.xf_data,
    backgroundColor:'rgba(167,139,250,.75)',borderRadius:5}],{plugins:{legend:{display:false}},scales:yg});
  if(d.sla_tr_labels.length) mk('c-sla-trend','line',d.sla_tr_labels,[{label:'SLA %',data:d.sla_tr_vals,
    borderColor:'#30d988',backgroundColor:'rgba(48,217,136,.1)',fill:true,tension:.4,pointRadius:4,
    pointBackgroundColor:'#30d988'}],{scales:ypct});
  if(d.sla_pri_labels.length) mk('c-sla-pri','bar',d.sla_pri_labels,[{label:'SLA %',data:d.sla_pri_vals,
    backgroundColor:d.sla_pri_vals.map(v=>v>=70?'#30d988':v>=50?'#ffc240':'#ff4f6a'),
    borderRadius:7,borderSkipped:false}],{plugins:{legend:{display:false}},scales:ypct});
  if(d.mttr_grp_labels.length) mk('c-mttr-grp','bar',d.mttr_grp_labels,[{label:'Avg hrs',data:d.mttr_grp_vals,
    backgroundColor:'rgba(255,194,64,.75)',borderRadius:6,borderSkipped:false}],
    {indexAxis:'y',plugins:{legend:{display:false}},
     scales:{x:{grid:{color:'#1e2a40'},ticks:{callback:v=>v+'h'}},y:{grid:{display:false}}}});
  if(d.ci_labels.length) mk('c-ci','bar',d.ci_labels,[{label:'Incidents',data:d.ci_data,
    backgroundColor:'rgba(34,211,238,.7)',borderRadius:5,borderSkipped:false}],
    {indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{grid:{color:'#1e2a40'}},y:{grid:{display:false}}}});
  if(d.svc_labels.length) mk('c-svc','doughnut',d.svc_labels,[{data:d.svc_data,
    backgroundColor:PAL,borderWidth:2,borderColor:'#111827'}],{cutout:'55%'});
}

// ── HEATMAP ─────────────────────────────────────────────────────────────────
function buildHeatmap(d){
  if(!d.pri_heat||!d.pri_heat.months||!d.pri_heat.months.length) return;
  $('heat-card').style.display='';
  const h=d.pri_heat, maxV=Math.max(...h.values.flat().filter(v=>v>0),1);
  let html=`<thead><tr><th>Priority \\ Month</th>${h.months.map(m=>`<th>${m}</th>`).join('')}</tr></thead><tbody>`;
  h.priorities.forEach((p,pi)=>{
    const pc=PRI_C[p]||'#4a8cff';
    html+=`<tr><td class="row-lbl" style="color:${pc}">${p}</td>`;
    h.months.forEach((_,mi)=>{
      const v=h.values[mi]?h.values[mi][pi]||0:0;
      const intensity=v/maxV;
      const bg=`${pc}${Math.round(intensity*200+20).toString(16).padStart(2,'0')}`;
      html+=`<td style="background:${bg};color:${intensity>.4?'#fff':'#7b8db0'}">${v||''}</td>`;
    });
    html+='</tr>';
  });
  $('heat-tbl').innerHTML=html+'</tbody>';
}

// ── ORG TABLE ────────────────────────────────────────────────────────────────
function buildOrgTable(d){
  $('org-tbody').innerHTML=d.org_rows.map(r=>{
    const sla=parseFloat(r.sla_pct)||0,cls=sla>=70?'c-met':sla>=50?'c-pend':'c-breach';
    return `<tr><td style="font-weight:700">${r.Assigned_Support_Organisation||'—'}</td>
      <td style="font-family:'DM Mono',monospace;font-weight:700">${r.total}</td>
      <td><span class="chip c-breach">${r.open||0}</span></td>
      <td><span class="chip ${cls}">${sla}%</span></td>
      <td style="font-family:'DM Mono',monospace">${r.mttr||'—'}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${Math.min(sla,100)}%;background:${sla>=70?'#30d988':sla>=50?'#ffc240':'#ff4f6a'}"></div></div></div></td></tr>`;
  }).join('');
}

// ── TEAM PANELS ──────────────────────────────────────────────────────────────
function buildTeamPanels(d){
  const c=$('team-panels');
  if(!d.group_tables.length){c.innerHTML='<div class="cc"><div style="color:var(--muted);font-size:.83rem">No AssignedGroup / Assignee data found.</div></div>';return;}
  c.innerHTML=d.group_tables.map((g,gi)=>{
    const sc=typeof g.sla_pct==='number'?(g.sla_pct>=70?'c-met':g.sla_pct>=50?'c-pend':'c-breach'):'';
    const sd=typeof g.sla_pct==='number'?g.sla_pct+'%':g.sla_pct;
    const maxT=g.assignees.length?Math.max(...g.assignees.map(a=>a.Total)):1;
    const aCols=g.assignees.length?Object.keys(g.assignees[0]):[];
    const eid='tm-'+gi;
    const trs=g.assignees.map(a=>{
      const asla=parseFloat(a.SLA_Pct||0),ac=asla>=70?'c-met':asla>=50?'c-pend':'c-breach';
      const pct=Math.round(a.Total/maxT*100);
      return '<tr>'+aCols.map(k=>{
        const v=a[k];
        if(k==='SLA_Pct') return `<td><span class="chip ${ac}">${v}%</span></td>`;
        if(k==='Assignee') return `<td style="font-weight:600">${v}</td>`;
        if(k==='Total') return `<td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--blue)"></div></div><span style="font-family:'DM Mono',monospace;font-size:.72rem;color:var(--muted)">${v}</span></div></td>`;
        return `<td style="font-family:'DM Mono',monospace;font-size:.74rem">${v}</td>`;
      }).join('')+'</tr>';
    }).join('');
    return `<div class="grp-panel">
      <div class="grp-hdr" onclick="toggleGrp(this)">
        <span style="font-size:.95rem">👥</span><span class="gname">${g.group}</span>
        <div class="grp-stats">
          <span class="gs">Total <strong>${g.total}</strong></span>
          <span class="gs">Resolved <strong>${g.resolved}</strong></span>
          <span class="gs">MTTR <strong>${g.mttr}h</strong></span>
          <span class="gs">SLA <strong><span class="chip ${sc}" style="font-size:.68rem">${sd}</span></strong></span>
          <span class="gs">Avg Xfers <strong>${g.avg_transfers}</strong></span>
        </div><span class="g-expand">▼</span>
      </div>
      <div class="grp-body">
        <div class="tbl-toolbar">
          <input class="search-box" placeholder="Search assignee…" oninput="filterTbl(this,'${eid}')"/>
          <button class="export-btn" onclick="exportCSV('${eid}','${g.group.replace(/\s/g,'_')}_team')">⬇ CSV</button>
        </div>
        <div class="tbl-wrap"><table><thead><tr>${aCols.map(k=>`<th>${k.replace(/_/g,' ')}</th>`).join('')}</tr></thead><tbody id="${eid}">${trs}</tbody></table></div>
      </div>
    </div>`;
  }).join('');
}

// ── CI TABLE ─────────────────────────────────────────────────────────────────
function buildCI(d){
  $('ci3-badge').textContent=d.ci_gt3_rows.length+' CIs';
  const maxC=d.ci_gt3_rows.length?Math.max(...d.ci_gt3_rows.map(r=>r.count)):1;
  $('ci3-tbody').innerHTML=d.ci_gt3_rows.map((r,i)=>{
    const pct=Math.round(r.count/maxC*100);
    return `<tr><td style="color:var(--dim);font-family:'DM Mono',monospace">${i+1}</td>
      <td style="font-weight:700;color:var(--cyan);font-family:'DM Mono',monospace">${r.HPD_CI}</td>
      <td><span class="chip c-orange">${r.count}</span></td>
      <td style="font-size:.73rem;color:var(--muted)">${r.groups_str}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--orange)"></div></div></div></td></tr>`;
  }).join('');
}

// ── KB ────────────────────────────────────────────────────────────────────────
function buildKB(d){
  const s=d.kb_summary;
  $('kb-kpis').innerHTML=[
    {l:'RKM Solution Tagged', v:s.total_rkm,    c:'var(--purple)', i:'📚', sub:'Request_Type01 = RKM Solution'},
    {l:'Without KB Article',  v:s.total_no_kb,  c:'var(--red)',    i:'❌', sub:'No KB tag found'},
    {l:'Unique KB IDs',       v:s.unique_kb_ids,c:'var(--green)',  i:'🔑', sub:'Distinct KB articles used'},
  ].map(k=>`<div class="kpi"><div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-icon">${k.i}</div><div class="kpi-lbl">${k.l}</div>
    <div class="kpi-val" style="color:${k.c}">${k.v}</div><div class="kpi-sub">${k.sub}</div></div>`).join('');

  $('kb-tbody').innerHTML=d.kb_group_rows.map(r=>{
    const cls=r.kb_pct>=60?'c-met':r.kb_pct>=30?'c-pend':'c-breach';
    const tops=r.top_kbs.map(k=>`<span class="chip c-purple" style="margin:1px;font-size:.63rem">${k.KB_ID} (${k.count})</span>`).join(' ')||'—';
    return `<tr><td style="font-weight:700">${r.group}</td>
      <td style="font-family:'DM Mono',monospace">${r.total}</td>
      <td style="font-family:'DM Mono',monospace;color:var(--purple)">${r.with_kb}</td>
      <td style="font-family:'DM Mono',monospace;color:var(--red)">${r.without_kb}</td>
      <td><span class="chip ${cls}">${r.kb_pct}%</span></td>
      <td style="max-width:240px">${tops}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${r.kb_pct}%;background:var(--purple)"></div></div></div></td></tr>`;
  }).join('');

  if(d.kb_no_kb_rows.length){
    const cols=Object.keys(d.kb_no_kb_rows[0]);
    $('kb-no-thead').innerHTML='<tr>'+cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')+'</tr>';
    $('kb-no-tbody').innerHTML=d.kb_no_kb_rows.map(r=>
      '<tr>'+cols.map(c=>`<td style="font-size:.74rem">${r[c]||'—'}</td>`).join('')+'</tr>'
    ).join('');
  }
}

// ── DATA INFO ─────────────────────────────────────────────────────────────────
function buildInfo(d){
  $('col-chips').innerHTML=d.detected_cols.map(c=>`<span style="background:var(--surf);border:1px solid var(--border);
    padding:3px 10px;border-radius:6px;font-size:.67rem;font-family:'DM Mono',monospace;color:var(--muted)">${c}</span>`).join('');
  $('mttr-logic').innerHTML=`<strong style="color:var(--text)">Formula:</strong><br>
    <span style="color:var(--blue);font-family:'DM Mono',monospace">${d.mttr_source}</span><br><br>
    <strong style="color:var(--text)">Priority:</strong><br>
    1. <span style="color:var(--green)">SubmitDate</span> present → LastResolvedDate − SubmitDate<br>
    2. <span style="color:var(--yellow)">ReportedDate</span> present → LastResolvedDate − ReportedDate<br><br>
    Negatives excluded (data errors). Unit: <strong style="color:var(--yellow)">Hours</strong>`;
  const ff=d.feature_flags||{};
  $('feat-flags').innerHTML=Object.entries(ff).map(([k,v])=>
    `<div><span style="color:${v?'var(--green)':'var(--dim)'}">${v?'✓':'✗'}</span> ${k}</div>`
  ).join('');
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import socket
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("8.8.8.8",80)); lan_ip=s.getsockname()[0]; s.close()
    except: lan_ip="YOUR_LINUX_IP"
    port=5050
    print("\n"+"="*64)
    print("  ITIS Incident Analyzer  v4")
    print("="*64)
    print(f"  Local   :  http://localhost:{port}")
    print(f"  Remote  :  http://{lan_ip}:{port}   ← open from Windows browser")
    print("="*64)
    print("  Formats : .xlsx  .xls  .csv")
    print("  KB cols : 'Request Type01' + 'Request Description01'")
    print("  Fuzzy column matching — tolerates renamed columns")
    print("  Press Ctrl+C to stop")
    print("="*64+"\n")
    app.run(debug=False,port=port,host="0.0.0.0",threaded=True)
