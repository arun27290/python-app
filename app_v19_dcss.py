"""
DCSS Incident Management Analyzer  v5
=======================================
Run  :  python app.py
Open :  http://localhost:5050

100% OFFLINE — no internet required.
All charts generated server-side with matplotlib (no Chart.js CDN).
No Google Fonts CDN — uses system fonts only.
Single file, no external assets needed.

Designed by aawasthi
"""

import io, json, warnings, logging, re, difflib, base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, jsonify

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── matplotlib offline setup ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                       # no display needed — pure server-side
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# Dark theme colours matching the UI
# -- Accent colours (shared between themes) --
BLUE   = "#4a8cff"; RED    = "#ff4f6a"; GREEN  = "#30d988"
YELLOW = "#ffc240"; PURPLE = "#a78bfa"; CYAN   = "#22d3ee"
ORANGE = "#fb923c"
PALETTE = [BLUE, RED, GREEN, YELLOW, PURPLE, CYAN, ORANGE, "#f472b6", "#34d399", "#f87171"]
PRI_COLORS = {"P1 - Critical": RED, "Critical": RED,
              "P2 - High": YELLOW, "High": YELLOW,
              "P3 - Medium": BLUE, "Medium": BLUE,
              "P4 - Low": GREEN, "Low": GREEN}
# SLA colour rules — GREEN is ONLY for "Met" and its direct synonyms.
# Every other status gets a distinct non-green colour regardless of position.
SLA_COLORS = {
    # ── MET (green family) ────────────────────────────────────────────────────
    "Met":          GREEN,  "Within SLA": GREEN,  "SLA Met":   GREEN,
    "OK":           GREEN,  "Yes":        GREEN,  "Compliant": GREEN,
    "On Track":     GREEN,
    # ── BREACHED / MISSED (red) ───────────────────────────────────────────────
    "Breached":     RED,    "SLA Breached": RED,  "Missed":    RED,
    "Violated":     RED,    "Failed":       RED,  "No":        RED,
    "Overdue":      RED,    "Non-Compliant": RED,
    # ── IN-PROGRESS / PENDING (yellow/orange — not yet determined) ────────────
    "Pending":      YELLOW, "In Progress": YELLOW, "Open":     YELLOW,
    "Active":       YELLOW, "Running":     YELLOW,
    # ── INVALID / EXEMPT / EXCLUDED (grey — not counted either way) ──────────
    "Invalid":      "#94a3b8", "Exempt":   "#94a3b8", "Excluded": "#94a3b8",
    "N/A":          "#94a3b8", "NA":       "#94a3b8", "Not Applicable": "#94a3b8",
    "Cancelled":    "#94a3b8", "Withdrawn": "#94a3b8",
}

def sla_color_for(label):
    """Return the colour for an SLA status label.
    Falls back to a non-green colour so that 'Met' stays uniquely green."""
    explicit = SLA_COLORS.get(label)
    if explicit:
        return explicit
    # Unknown value — use orange so it's visible but never green
    return ORANGE

# -- Theme palettes --
THEMES = {
    "dark": {
        "BG": "#07090f", "SURFACE": "#111827", "BORDER": "#1e2a40",
        "MUTED": "#7b8db0", "TEXT": "#edf2ff", "GRID": "#1e2a40",
        "LEG_BG": "#111827", "LEG_EDGE": "#1e2a40",
    },
    "light": {
        "BG": "#f8fafc", "SURFACE": "#ffffff", "BORDER": "#e2e8f0",
        "MUTED": "#64748b", "TEXT": "#0f172a", "GRID": "#e2e8f0",
        "LEG_BG": "#ffffff", "LEG_EDGE": "#cbd5e1",
    },
}

def _apply_rc(theme_name):
    p = THEMES[theme_name]
    plt.rcParams.update({
        "figure.facecolor": p["BG"],   "axes.facecolor":   p["SURFACE"],
        "axes.edgecolor":   p["BORDER"],"axes.labelcolor":  p["MUTED"],
        "xtick.color":      p["MUTED"], "ytick.color":      p["MUTED"],
        "text.color":       p["TEXT"],  "grid.color":       p["GRID"],
        "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
        "font.size": 9, "axes.titlesize": 10,
        "axes.titlecolor":  p["TEXT"],  "axes.titlepad": 8,
        "legend.facecolor": p["LEG_BG"],"legend.edgecolor": p["LEG_EDGE"],
        "legend.fontsize": 8,
    })
    return p

# Initialise with dark (module-level default)
_dp = _apply_rc("dark")
BG = _dp["BG"]; SURFACE = _dp["SURFACE"]; BORDER = _dp["BORDER"]
MUTED = _dp["MUTED"]; TEXT = _dp["TEXT"]

def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def img(src, alt="", cls="chart-img"):
    return f'<img src="{src}" alt="{alt}" class="{cls}"/>'

# ── chart generators ──────────────────────────────────────────────────────────

def make_donut(labels, values, colors=None, title="", size=(4.5, 3.8)):
    if not labels or not values or sum(values) == 0:
        return None
    all_cols = colors if colors else [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    total    = sum(values)

    # Legend always shows every category with its real count and correct color
    legend_patches = [mpatches.Patch(color=c, label=f"{l} ({v})")
                      for l, v, c in zip(labels, values, all_cols)]

    # For rendering: give every non-zero category a minimum visible slice of 1.5%
    # so tiny values (e.g. 3 out of 2292) are still visible as a coloured wedge.
    # We scale UP tiny slices to the minimum, then normalise so they still sum correctly.
    min_pct   = 0.015          # minimum 1.5% wedge angle
    min_count = max(1, round(total * min_pct))
    render_vals = [max(v, min_count) if v > 0 else 0 for v in values]

    # Filter out true zeros (keep their legend entry but don't render)
    nz = [(v_r, c) for v_r, v_orig, c in zip(render_vals, values, all_cols) if v_orig > 0]
    if not nz:
        return None
    nz_render, nz_cols = zip(*nz)

    # Explode tiny slices outward so they're visually distinct
    nz_orig = [v for v in values if v > 0]
    explode  = [0.08 if v < total * 0.03 else 0 for v in nz_orig]

    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.pie(nz_render, colors=nz_cols, startangle=90, explode=explode,
           wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2.5))
    ax.text(0, 0, str(total), ha="center", va="center",
            fontsize=16, fontweight="bold", color=TEXT)
    ax.set_title(title, color=TEXT, pad=6)
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, framealpha=0, fontsize=7.5)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_hbar(labels, values, color=BLUE, title="", xlabel="", size=(5.5, 0.45)):
    if not labels or not values:
        return None
    n = len(labels)
    h = max(2.5, n * size[1])
    fig, ax = plt.subplots(figsize=(size[0], h))
    y = range(n)
    bars = ax.barh(list(y), values, color=color, height=0.6,
                   edgecolor=BG, linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top","right","left"]].set_visible(False)
    # value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val) if isinstance(val, int) else f"{val:.1f}",
                va="center", fontsize=7.5, color=TEXT)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_hbar_colored(labels, values, title="", xlabel="", lo=50, hi=70, size=(5.5, 0.45)):
    """Horizontal bar with green/yellow/red coloring per value."""
    if not labels or not values:
        return None
    cols = [GREEN if v >= hi else (YELLOW if v >= lo else RED) for v in values]
    return make_hbar(labels, values, color=cols[0] if len(set(cols))==1 else BLUE,
                     title=title, xlabel=xlabel, size=size)

def make_vbar(labels, values, colors=None, title="", ylabel="", size=(7, 3.2)):
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=size)
    x = range(len(labels))
    cols = colors if colors else [BLUE] * len(labels)
    ax.bar(list(x), values, color=cols, edgecolor=BG, linewidth=0.5, width=0.65)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_line(labels, values, color=BLUE, title="", ylabel="", pct=False, size=(8, 3.2)):
    if not labels or not values:
        return None
    fig, ax = plt.subplots(figsize=size)
    ax.plot(labels, values, color=color, linewidth=2.2, marker="o",
            markersize=5, markerfacecolor=color, markeredgecolor=BG, markeredgewidth=1.5)
    ax.fill_between(range(len(labels)), values, alpha=0.12, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    if pct:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return fig_to_b64(fig)

def make_multiline(labels, series, title="", ylabel="", pct=False, size=(8, 3.2)):
    """series = list of (name, values, color)"""
    if not labels or not series:
        return None
    fig, ax = plt.subplots(figsize=size)
    for name, vals, col in series:
        ax.plot(labels, vals, color=col, linewidth=2, marker="o",
                markersize=4, label=name, markeredgecolor=BG, markeredgewidth=1)
        ax.fill_between(range(len(labels)), vals, alpha=0.07, color=col)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(framealpha=0.2)
    if pct:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return fig_to_b64(fig)

def make_heatmap(months, priorities, values, title="Priority × Month Heatmap", size=(10, 3.5)):
    if not months or not priorities or not values:
        return None
    data = np.array(values, dtype=float).T   # priorities × months
    fig, ax = plt.subplots(figsize=size)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dcss", [BG, "#1a2a50", BLUE, CYAN])
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(priorities)))
    ax.set_yticklabels(priorities, fontsize=8)
    for i in range(len(priorities)):
        for j in range(len(months)):
            val = int(data[i, j]) if not np.isnan(data[i, j]) else 0
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7.5, color=TEXT, fontweight="bold")
    ax.set_title(title, color=TEXT)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)

def make_stacked_bar(labels, series, title="", ylabel="", size=(7, 3.5)):
    """series = list of (name, values, color)
    All series are always plotted and shown in legend, even if all-zero,
    so the color mapping stays stable regardless of data.
    """
    if not labels or not series: return None
    n   = len(labels)
    fig, ax = plt.subplots(figsize=size)
    x   = np.arange(n)
    bottoms = np.zeros(n)
    for name, vals, col in series:
        vals_arr = np.array(vals, dtype=float)
        # Always add to chart (even if all zeros) so legend color stays correct
        ax.bar(x, vals_arr, bottom=bottoms, label=f"{name} ({int(sum(vals_arr))})",
               color=col, edgecolor=BG, linewidth=0.4, width=0.65)
        bottoms += vals_arr
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(framealpha=0.2, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig_to_b64(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  COLUMN REGISTRY + FUZZY MATCHING
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL = {
    "Incident_Number":  ["Incident_Number","IncidentNumber","Incident_Nember","IncidentID",
                         "Incident ID","Incident Number","Inc Number","INC No"],
    "ReportedDate":     ["ReportedDate","Reported Date","Reported_Date","Open Date","OpenDate"],
    "LastResolvedDate": ["LastResolvedDate","Last Resolved Date","ResolvedDate",
                         "Resolved Date","Close Date","CloseDate","Resolution Date"],
    "SubmitDate":       ["SubmitDate","Submit Date","Submit_Date","Submission Date",
                         "Created Date","CreateDate","Create Date"],
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
    "Request_Type01":   ["Request_Type01","RequestType01","Request Type01","Request_Type_01",
                         "Request Type 01","RequestType 01","Req Type01","Req_Type01"],
    "Request_Desc01":   ["Request_Desc01","RequestDesc01","Request Description01",
                         "Request_Description01","Request Desc01","Request_Type_Description",
                         "RequestTypeDescription","Request Type Description",
                         "Req Description01","Req Desc01","Description01"],
}

_ALIAS_MAP = {}
for canon, aliases in CANONICAL.items():
    for a in aliases:
        _ALIAS_MAP[a.lower().strip()] = canon

def fuzzy_match(col_name, threshold=0.70):
    key = col_name.lower().strip()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    best_score, best_canon = 0, None
    for alias, canon in _ALIAS_MAP.items():
        score = difflib.SequenceMatcher(None, key, alias).ratio()
        if score > best_score:
            best_score, best_canon = score, canon
    if best_score >= threshold:
        log.info("Fuzzy matched '%s' → '%s' (%.0f%%)", col_name, best_canon, best_score * 100)
        return best_canon
    return None

def normalise_columns(df):
    df.columns = [c.strip() for c in df.columns]
    rename_map, mapped = {}, set()
    for col in df.columns:
        canon = fuzzy_match(col)
        if canon and canon not in mapped:
            rename_map[col] = canon
            mapped.add(canon)
    if rename_map:
        log.info("Columns mapped: %s", rename_map)
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
            "P3 - Medium":2,"Medium":2,"P4 - Low":3,"Low":3}.get(str(p), 99)

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

    F = {k: k in df.columns for k in [
        "ReportedDate","LastResolvedDate","SubmitDate","Status","SLAStatus",
        "Priority","AssignedGroup","Assignee","Assigned_Support_Organisation",
        "HPD_CI","Service_Type","Group_Transfers","Request_Type01","Request_Desc01"
    ]}
    inc_col = "Incident_Number" if "Incident_Number" in df.columns else df.columns[0]

    # date reference for Month / Week / DayOfWeek
    date_ref = next((c for c in ["ReportedDate","SubmitDate","LastResolvedDate"]
                     if F.get(c) and df[c].notna().any()), None)
    if date_ref:
        df["Month"]     = df[date_ref].dt.to_period("M").astype(str)
        df["DayOfWeek"] = df[date_ref].dt.day_name()
        # ISO week label: e.g. "2024-W03"  (year of the ISO week, not calendar year)
        df["Week"] = (
            df[date_ref].dt.isocalendar().year.astype(str) + "-W" +
            df[date_ref].dt.isocalendar().week.astype(str).str.zfill(2)
        )

    # MTTR
    if F["LastResolvedDate"]:
        start = "SubmitDate" if F["SubmitDate"] else ("ReportedDate" if F["ReportedDate"] else None)
        if start:
            df["MTTR_Hours"] = ((df["LastResolvedDate"] - df[start])
                                 .dt.total_seconds() / 3600).round(2)
            df.loc[df["MTTR_Hours"] < 0, "MTTR_Hours"] = np.nan
            mttr_source = f"LastResolvedDate − {start}"
        else:
            df["MTTR_Hours"] = np.nan; mttr_source = "N/A"
    else:
        df["MTTR_Hours"] = np.nan; mttr_source = "N/A"

    # Age in days
    if F["LastResolvedDate"]:
        age_s = "SubmitDate" if F["SubmitDate"] else ("ReportedDate" if F["ReportedDate"] else None)
        if age_s:
            df["AgeDays"] = ((df["LastResolvedDate"] - df[age_s]).dt.total_seconds() / 86400).round(1)
            df.loc[df["AgeDays"] < 0, "AgeDays"] = np.nan

    if F["Group_Transfers"]:
        df["Group_Transfers"] = pd.to_numeric(df["Group_Transfers"], errors="coerce")

    # subsets
    if F["Status"]:
        st_l = df["Status"].str.strip().str.lower()
        closed = df[st_l.isin(CLOSED_STATUSES)].copy()
        open_ct = int(st_l.isin(OPEN_STATUSES).sum()); closed_ct = len(closed)
    elif F["LastResolvedDate"]:
        closed = df[df["LastResolvedDate"].notna()].copy()
        open_ct = int(df["LastResolvedDate"].isna().sum()); closed_ct = len(closed)
    else:
        closed = df.copy(); open_ct = 0; closed_ct = len(df)

    total = len(df)

    # date range
    date_min = date_max = "N/A"
    for c in ["SubmitDate","ReportedDate","LastResolvedDate"]:
        if F.get(c) and df[c].notna().any():
            date_min = df[c].min().strftime("%d-%b-%Y")
            date_max = df[c].max().strftime("%d-%b-%Y")
            break

    # SLA
    if F["SLAStatus"]:
        sla_l = df["SLAStatus"].str.strip().str.lower()
        sla_met_ct = int(sla_l.isin(MET_VALUES).sum())
        sla_pct = round(sla_met_ct / total * 100, 1) if total else 0
    else:
        sla_met_ct = sla_pct = 0

    mttr = round(float(df["MTTR_Hours"].mean()), 1) if df["MTTR_Hours"].notna().any() else 0
    p1_ct = int(df["Priority"].str.lower().str.contains("critical|p1", na=False).sum()) if F["Priority"] else 0

    hop_dist_rows  = []
    mttr_dist_rows = []

    # ── First-Time-Fix Rate per AssignedGroup ─────────────────────────────────
    # Incidents resolved without any group transfer (Group_Transfers == 0)
    ftf_rows = []
    if F["AssignedGroup"] and F["Group_Transfers"]:
        ftf_grp = []
        for gname, gdf in df.groupby("AssignedGroup"):
            g_total   = len(gdf)
            g_xf      = pd.to_numeric(gdf["Group_Transfers"], errors="coerce")
            g_ftf     = int((g_xf == 0).sum())
            ftf_grp.append({
                "group":    gname,
                "total":    g_total,
                "ftf":      g_ftf,
                "ftf_pct":  round(g_ftf / g_total * 100, 1) if g_total else 0,
            })
        ftf_rows = sorted(ftf_grp, key=lambda x: x["ftf_pct"], reverse=True)

    # ── P1 SLA Breach detail list ─────────────────────────────────────────────
    p1_breach_rows = []
    if F["Priority"] and F["SLAStatus"]:
        p1_mask    = df["Priority"].str.lower().str.contains("critical|p1", na=False)
        breach_mask= df["SLAStatus"].str.strip().str.lower().isin({"breached","sla breached","violated"})
        p1_breach  = df[p1_mask & breach_mask]
        cols_want  = [c for c in [inc_col,"AssignedGroup","Priority","SLAStatus",
                                   "Status","MTTR_Hours","Assignee"] if c in p1_breach.columns]
        if "MTTR_Hours" in p1_breach.columns:
            p1_breach = p1_breach.copy()
            p1_breach["MTTR_Hours"] = p1_breach["MTTR_Hours"].round(1)
        p1_breach_rows = p1_breach[cols_want].fillna("—").to_dict("records")

    # ── Repeat Incident Rate: CIs appearing 5+ times ─────────────────────────
    repeat_ci_rows = []
    if F["HPD_CI"]:
        repeat = (df.groupby("HPD_CI").size().reset_index(name="count")
                    .query("count >= 5")
                    .sort_values("count", ascending=False))
        repeat_ci_rows = repeat.to_dict("records")

    # ── HOP Distribution (Group_Transfers banded) ─────────────────────────────
    HOP_BANDS = [
        ("0 – 1",  0,   1),
        ("2",      2,   2),
        ("3 – 4",  3,   4),
        ("5 – 6",  5,   6),
        ("7 – 8",  7,   8),
        ("9 – 10", 9,  10),
        ("10+",   11, None),
    ]
    hop_dist_rows = []
    if F["Group_Transfers"]:
        xf_num = pd.to_numeric(df["Group_Transfers"], errors="coerce").dropna()
        xf_total = len(xf_num)
        band_total = 0
        for label, lo, hi in HOP_BANDS:
            if hi is None:
                mask = xf_num >= lo
            else:
                mask = (xf_num >= lo) & (xf_num <= hi)
            cnt = int(mask.sum())
            band_total += cnt
            hop_dist_rows.append({
                "HOPs":          label,
                "Incident_Count": cnt,
                "Pct":           f"{round(cnt / xf_total * 100, 1)}%" if xf_total else "0.0%",
            })
        # Total row
        hop_dist_rows.append({
            "HOPs":           "Total",
            "Incident_Count":  xf_total,
            "Pct":            "100.0%",
        })

    # ── MTTR Distribution (resolution time banded) ────────────────────────────
    MTTR_BANDS = [
        ("0 – 1 hrs",      0,       1),
        ("1 – 2 hrs",      1,       2),
        ("2 – 4 hrs",      2,       4),
        ("4 – 8 hrs",      4,       8),
        ("8 – 16 hrs",     8,      16),
        ("16 – 24 hrs",   16,      24),
        ("24 – 48 hrs",   24,      48),
        ("2 – 4 days",    48,      96),
        ("4 – 8 days",    96,     192),
        ("8 – 16 days",  192,     384),
        ("16 – 30 days", 384,     720),
        ("30 days+",     720,    None),
    ]
    mttr_dist_rows = []
    if "MTTR_Hours" in df.columns and df["MTTR_Hours"].notna().any():
        mh = df["MTTR_Hours"].dropna()
        mh_total = len(mh)
        for label, lo, hi in MTTR_BANDS:
            if hi is None:
                mask = mh >= lo
            else:
                mask = (mh >= lo) & (mh < hi)
            cnt = int(mask.sum())
            mttr_dist_rows.append({
                "Resolution_Time_Band": label,
                "Incident_Count":       cnt,
                "Pct":                  f"{round(cnt / mh_total * 100, 1)}%" if mh_total else "0.0%",
            })
        # Total row
        mttr_dist_rows.append({
            "Resolution_Time_Band": "Total",
            "Incident_Count":       mh_total,
            "Pct":                  "100.0%",
        })

    # Health score
    health = 100
    if sla_pct < 70:   health -= min(30, int(70 - sla_pct))
    if mttr > 24:      health -= min(25, int((mttr - 24) / 4))
    mttr_hi_ct = int((df["MTTR_Hours"] > 200).sum()) if "MTTR_Hours" in df.columns else 0
    hop_hi_ct  = int((df["Group_Transfers"] >= 5).sum()) if F["Group_Transfers"] else 0
    health -= min(25, mttr_hi_ct * 3)
    health -= min(20, hop_hi_ct)
    health = max(0, min(100, health))

    # ── alert lists ───────────────────────────────────────────────────────────
    def make_grp_alerts(mask_col, threshold, comp, extra_cols, stat_key):
        groups = []
        if mask_col not in df.columns: return groups
        hi = df[df[mask_col] > threshold].copy()
        if mask_col == "MTTR_Hours": hi[mask_col] = hi[mask_col].round(1)
        cols = [c for c in [inc_col,"AssignedGroup", mask_col] + extra_cols if c in hi.columns]
        if F["AssignedGroup"]:
            for gname, gdf in hi.groupby("AssignedGroup"):
                groups.append({
                    "group": gname, "count": len(gdf),
                    stat_key: round(float(gdf[mask_col].mean()), 1),
                    "rows": gdf[cols].fillna("—").to_dict("records"),
                })
            groups.sort(key=lambda x: x["count"], reverse=True)
        else:
            groups = [{"group":"All","count":len(hi), stat_key: round(float(hi[mask_col].mean()),1),
                       "rows": hi[cols].fillna("—").to_dict("records")}]
        return groups

    mttr_grp_alerts = make_grp_alerts("MTTR_Hours", 200, ">",
                                      ["Priority","SLAStatus","Status","HPD_CI"], "avg_mttr")
    hop_grp_alerts  = make_grp_alerts("Group_Transfers", 4, ">",
                                      ["Priority","SLAStatus","Status","MTTR_Hours"], "avg_hops")

    age_alert_rows = []
    if "AgeDays" in df.columns:
        age_hi = df[df["AgeDays"] > 30].copy()
        age_hi["AgeDays"] = age_hi["AgeDays"].round(1)
        cols = [c for c in [inc_col,"AssignedGroup","AgeDays","Priority","Status"] if c in age_hi.columns]
        age_alert_rows = age_hi[cols].sort_values("AgeDays", ascending=False).fillna("—").to_dict("records")

    # ── chart data ────────────────────────────────────────────────────────────
    vol_labels = vol_data = []
    if "Month" in df.columns:
        v = df.groupby("Month").size().reset_index(name="count").sort_values("Month")
        vol_labels, vol_data = v["Month"].tolist(), v["count"].tolist()

    pri_labels = pri_data = []
    if F["Priority"]:
        p = df["Priority"].value_counts().reset_index()
        p.columns = ["Priority","count"]
        p["s"] = p["Priority"].apply(priority_sort_key)
        p = p.sort_values("s").drop("s",axis=1)
        pri_labels, pri_data = p["Priority"].tolist(), p["count"].tolist()

    sla_labels = sla_data = []
    if F["SLAStatus"]:
        s = df["SLAStatus"].value_counts().reset_index(); s.columns=["SLAStatus","count"]
        sla_labels, sla_data = s["SLAStatus"].tolist(), s["count"].tolist()

    sla_pri_labels = sla_pri_vals = []
    if F["Priority"] and F["SLAStatus"]:
        def _sp(x): return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)
        sp = df.groupby("Priority").apply(_sp).reset_index(name="pct")
        sp["s"] = sp["Priority"].apply(priority_sort_key)
        sp = sp.sort_values("s").drop("s",axis=1)
        sla_pri_labels, sla_pri_vals = sp["Priority"].tolist(), sp["pct"].tolist()

    sla_tr_labels = sla_tr_vals = []
    if F["SLAStatus"] and "Month" in df.columns:
        def _sm(x): return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)
        st = df.groupby("Month").apply(_sm).reset_index(name="pct").sort_values("Month")
        sla_tr_labels, sla_tr_vals = st["Month"].tolist(), st["pct"].tolist()

    # ── Weekly chart data ─────────────────────────────────────────────────────
    wk_vol_labels = wk_vol_data = []
    wk_sla_labels = wk_sla_vals = []
    wk_heat       = {}

    if "Week" in df.columns:
        # Weekly volume
        wv = (df.groupby("Week").size().reset_index(name="count")
                .sort_values("Week"))
        wk_vol_labels = wv["Week"].tolist()
        wk_vol_data   = wv["count"].tolist()

        # Weekly SLA %
        if F["SLAStatus"]:
            def _wsla(x):
                return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100, 1)
            ws = df.groupby("Week").apply(_wsla).reset_index(name="pct").sort_values("Week")
            wk_sla_labels = ws["Week"].tolist()
            wk_sla_vals   = ws["pct"].tolist()

        # Weekly Priority heatmap
        if F["Priority"]:
            wh = df.groupby(["Week","Priority"]).size().unstack(fill_value=0)
            wk_heat = {
                "weeks":      wh.index.tolist(),
                "priorities": wh.columns.tolist(),
                "values":     wh.values.tolist(),
            }

    grp_labels = grp_data = []
    if F["AssignedGroup"]:
        g = df.groupby("AssignedGroup").size().reset_index(name="count").sort_values("count",ascending=False)
        grp_labels, grp_data = g["AssignedGroup"].tolist(), g["count"].tolist()

    mttr_grp_labels = mttr_grp_vals = []
    if F["AssignedGroup"] and df["MTTR_Hours"].notna().any():
        mg = df.groupby("AssignedGroup")["MTTR_Hours"].mean().round(1).reset_index()
        mg.columns = ["group","mttr"]; mg = mg.sort_values("mttr")
        mttr_grp_labels, mttr_grp_vals = mg["group"].tolist(), mg["mttr"].tolist()

    dow_labels = dow_data = []
    if "DayOfWeek" in df.columns:
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df["DayOfWeek"].value_counts().reindex(days, fill_value=0)
        dow_labels, dow_data = dow.index.tolist(), dow.values.tolist()

    xf_labels = xf_data = []
    if F["Group_Transfers"]:
        xf = df["Group_Transfers"].value_counts().sort_index().reset_index()
        xf.columns = ["transfers","count"]
        xf_labels = [str(int(v)) for v in xf["transfers"].tolist()]
        xf_data = xf["count"].tolist()

    svc_labels = svc_data = []
    svc_sla_labels = svc_sla_vals = []
    svc_mttr_labels = svc_mttr_vals = []
    svc_vol_labels = svc_vol_data = []
    svc_vol_months = []; svc_vol_series = []   # always initialised
    if F["Service_Type"]:
        sv = df["Service_Type"].value_counts().reset_index(); sv.columns=["type","count"]
        svc_labels, svc_data = sv["type"].tolist(), sv["count"].tolist()

        # SLA % by Service_Type
        if F["SLAStatus"]:
            def _svc_sla(x):
                return round(x["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100, 1)
            svc_sla = df.groupby("Service_Type").apply(_svc_sla).reset_index(name="pct")
            svc_sla = svc_sla.sort_values("pct", ascending=False)
            svc_sla_labels = svc_sla["Service_Type"].tolist()
            svc_sla_vals   = svc_sla["pct"].tolist()

        # MTTR by Service_Type
        if df["MTTR_Hours"].notna().any():
            svc_mttr = (df.groupby("Service_Type")["MTTR_Hours"]
                          .mean().round(1).reset_index()
                          .sort_values("MTTR_Hours"))
            svc_mttr_labels = svc_mttr["Service_Type"].tolist()
            svc_mttr_vals   = svc_mttr["MTTR_Hours"].tolist()

        # Volume by Service_Type per month (for trend)
        if "Month" in df.columns:
            svc_monthly = (df.groupby(["Month","Service_Type"])
                             .size().unstack(fill_value=0)
                             .reset_index())
            svc_vol_months = svc_monthly["Month"].tolist()
            svc_types      = [c for c in svc_monthly.columns if c != "Month"]
            svc_vol_series = [(t, svc_monthly[t].tolist()) for t in svc_types]
        else:
            svc_vol_months = []; svc_vol_series = []

    ci_labels = ci_data = []; ci_gt3_list = []; ci_by_group = []
    if F["HPD_CI"]:
        ci_all = df.groupby("HPD_CI").size().reset_index(name="count").sort_values("count",ascending=False)
        ci_labels = ci_all.head(15)["HPD_CI"].tolist()
        ci_data   = ci_all.head(15)["count"].tolist()
        ci_ge3    = ci_all[ci_all["count"] >= 3].copy()

        for _, row in ci_ge3.iterrows():
            sub = df[df["HPD_CI"] == row["HPD_CI"]]
            grp_bd = []
            if F["AssignedGroup"]:
                grp_bd = (sub.groupby("AssignedGroup").size().reset_index(name="cnt")
                            .sort_values("cnt",ascending=False)
                            .apply(lambda r: f"{r['AssignedGroup']} ({r['cnt']})",axis=1).tolist())
            ci_gt3_list.append({"HPD_CI":row["HPD_CI"],"count":int(row["count"]),
                                  "groups_str":" · ".join(grp_bd) if grp_bd else "—"})

        # NEW: group CIs by AssignedGroup — each group panel shows its CIs >=3
        # Determine best date column for "Last Incident Date"
        _date_col_ci = next((c for c in ["ReportedDate","SubmitDate","LastResolvedDate"]
                             if c in df.columns and df[c].notna().any()), None)

        if F["AssignedGroup"]:
            for gname, gdf in df.groupby("AssignedGroup"):
                grp_ci = (gdf.groupby("HPD_CI").size()
                              .reset_index(name="count")
                              .sort_values("count", ascending=False))
                grp_ci_ge3 = grp_ci[grp_ci["count"] >= 3]
                if len(grp_ci_ge3) == 0:
                    continue
                max_c = int(grp_ci_ge3["count"].max())
                ci_rows = []
                for _, r in grp_ci_ge3.iterrows():
                    ci_name = r["HPD_CI"]
                    cnt     = int(r["count"])
                    # Last incident date for this CI within this group
                    last_date = "—"
                    if _date_col_ci:
                        ci_dates = gdf.loc[gdf["HPD_CI"] == ci_name, _date_col_ci].dropna()
                        if len(ci_dates) > 0:
                            last_date = ci_dates.max().strftime("%d-%b-%Y")
                    ci_rows.append({
                        "HPD_CI":           ci_name,
                        "count":            cnt,
                        "pct":              round(cnt / max_c * 100),
                        "Last_Incident_Date": last_date,
                    })
                ci_by_group.append({
                    "group":     gname,
                    "total_ci":  len(grp_ci_ge3),
                    "total_inc": int(gdf.shape[0]),
                    "rows":      ci_rows,
                })
            ci_by_group.sort(key=lambda x: x["total_inc"], reverse=True)

    # priority heatmap
    pri_heat = {}
    if F["Priority"] and "Month" in df.columns:
        heat = df.groupby(["Month","Priority"]).size().unstack(fill_value=0)
        pri_heat = {"months": heat.index.tolist(), "priorities": heat.columns.tolist(),
                    "values": heat.values.tolist()}

    # group tables
    group_tables = []
    if F["AssignedGroup"] and F["Assignee"]:
        for gname, gdf in df.groupby("AssignedGroup"):
            a_rows = []
            for aname, adf in gdf.groupby("Assignee"):
                r = {"Assignee": aname, "Total": len(adf)}
                if F["Status"]:    r["Resolved"] = int(adf["Status"].str.strip().str.lower().isin(CLOSED_STATUSES).sum())
                if df["MTTR_Hours"].notna().any(): r["Avg_MTTR_hrs"] = round(float(adf["MTTR_Hours"].mean()),1) if adf["MTTR_Hours"].notna().any() else "—"
                if F["SLAStatus"]: r["SLA_Pct"] = round(adf["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/len(adf)*100,1)
                if F["Group_Transfers"]: r["Avg_Transfers"] = round(float(adf["Group_Transfers"].mean()),1) if adf["Group_Transfers"].notna().any() else "—"
                a_rows.append(r)
            a_rows.sort(key=lambda x: x["Total"], reverse=True)
            g_total = len(gdf)
            g_res   = int(gdf["Status"].str.strip().str.lower().isin(CLOSED_STATUSES).sum()) if F["Status"] else g_total
            g_mttr  = round(float(gdf["MTTR_Hours"].mean()),1) if gdf["MTTR_Hours"].notna().any() else "—"
            g_sla   = round(gdf["SLAStatus"].str.strip().str.lower().isin(MET_VALUES).sum()/g_total*100,1) if F["SLAStatus"] and g_total else "—"
            g_xfer  = round(float(gdf["Group_Transfers"].mean()),1) if F["Group_Transfers"] and gdf["Group_Transfers"].notna().any() else "—"
            group_tables.append({"group":gname,"total":g_total,"resolved":g_res,
                                  "mttr":g_mttr,"sla_pct":g_sla,"avg_transfers":g_xfer,
                                  "assignees":a_rows})
        group_tables.sort(key=lambda x: x["total"], reverse=True)

    # org summary
    org_rows = []
    if F["Assigned_Support_Organisation"]:
        og = df.groupby("Assigned_Support_Organisation")
        org_agg = og.size().reset_index(name="total")
        if F["Status"]:
            org_agg = org_agg.merge(og["Status"].apply(lambda x: x.str.strip().str.lower().isin(OPEN_STATUSES).sum()).reset_index(name="open"), on="Assigned_Support_Organisation")
        if F["SLAStatus"]:
            org_agg = org_agg.merge(og["SLAStatus"].apply(lambda x: round(x.str.strip().str.lower().isin(MET_VALUES).sum()/len(x)*100,1)).reset_index(name="sla_pct"), on="Assigned_Support_Organisation")
        if df["MTTR_Hours"].notna().any():
            org_agg = org_agg.merge(og["MTTR_Hours"].mean().round(1).reset_index(name="mttr"), on="Assigned_Support_Organisation")
        org_rows = org_agg.fillna("—").to_dict("records")

    # ── KB / Request-type analysis ──────────────────────────────────────────────
    kb_available = F["Request_Type01"] and F["Request_Desc01"]
    kb_group_rows    = []
    kb_no_kb_rows    = []
    kb_summary_data  = {}
    kb_article_rows   = []   # RKM Solution desc — grouped by AssignedGroup
    ke_desc_rows      = []   # Known Error desc — grouped
    pi_desc_rows      = []   # Problem Investigation desc — grouped
    untagged_grp_rows = []   # Untagged incidents — grouped
    nokb_grp_rows     = []   # No-KB incidents — grouped
    tagging_summary  = {}      # NEW: RKM Solution / Known Error / Problem Investigation / Untagged
    tagging_grp_rows = []      # NEW: per-group tagging dashboard
    untagged_rows    = []      # NEW: incidents with none of the three tags
    log.info("KB check — Request_Type01:%s  Request_Desc01:%s", F["Request_Type01"], F["Request_Desc01"])

    # Tag classification values
    RKM_VALS     = {"rkm solution","rkmsolution","rkm_solution","kb solution","knowledge","rkm"}
    KNOWN_ERR    = {"known error","knownerror","known_error","ke"}
    PROB_SOL     = {
        # Problem Investigation variants (primary)
        "problem investigation","probleminvestigation","problem_investigation",
        "prob investigation","prob invest","pi","p investigation",
        "problem invest","prb investigation","problem invstigation",
        "problem invesigation","problem investigaton","problem investgation",
        # Problem Solution variants (legacy / alternate wording kept)
        "problem solution","problemsolution","problem_solution","prob solution",
        "problem sol","prob sol","ps",
    }
    ALL_TAG_VALS = RKM_VALS | KNOWN_ERR | PROB_SOL

    def extract_kb_desc(text):
        """Extract description part after the colon in 'KBxxxxxx: description'."""
        if pd.isna(text): return ""
        s = str(text).strip()
        # pattern: KB digits : anything
        m = re.search(r'KB\d{4,10}\s*[:]\s*(.*)', s, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # fallback: anything after first colon
        if ":" in s:
            return s.split(":", 1)[1].strip()
        return ""

    if kb_available:
        type_col = df["Request_Type01"].str.strip().str.lower().fillna("")
        df["_kb_id"]   = df["Request_Desc01"].apply(extract_kb)
        df["_kb_desc"] = df["Request_Desc01"].apply(extract_kb_desc)

        # masks for each tag type
        mask_rkm  = type_col.isin(RKM_VALS)
        mask_ke   = type_col.isin(KNOWN_ERR)
        mask_ps   = type_col.isin(PROB_SOL)
        mask_any  = mask_rkm | mask_ke | mask_ps
        mask_none = ~mask_any & type_col.notna() & (type_col != "")

        rkm_df   = df[mask_rkm]
        ke_df    = df[mask_ke]
        ps_df    = df[mask_ps]
        none_df  = df[~mask_any]   # truly untagged (includes blanks + other values)

        kb_summary_data = {
            "total_rkm":     len(rkm_df),
            "total_ke":      len(ke_df),
            "total_ps":      len(ps_df),
            "total_untagged": len(none_df),
            "unique_kb_ids": int(rkm_df["_kb_id"].nunique()),
        }

        # ── Tagging type summary (for new dashboard) ──────────────────────────
        tagging_summary = {
            "labels":  ["RKM Solution", "Known Error", "Problem Investigation", "Untagged"],
            "values":  [len(rkm_df), len(ke_df), len(ps_df), len(none_df)],
            "colors":  [PURPLE, ORANGE, CYAN, RED],
        }

        # ── Per-group tagging dashboard ───────────────────────────────────────
        if F["AssignedGroup"]:
            for gname, gdf in df.groupby("AssignedGroup"):
                tc = gdf["Request_Type01"].str.strip().str.lower().fillna("")
                g_rkm  = int(tc.isin(RKM_VALS).sum())
                g_ke   = int(tc.isin(KNOWN_ERR).sum())
                g_ps   = int(tc.isin(PROB_SOL).sum())
                g_none = int((~(tc.isin(ALL_TAG_VALS))).sum())
                g_tot  = len(gdf)
                tagging_grp_rows.append({
                    "group": gname, "total": g_tot,
                    "rkm": g_rkm, "ke": g_ke, "ps": g_ps, "untagged": g_none,
                    "tagged_pct": round((g_rkm + g_ke + g_ps) / g_tot * 100, 1) if g_tot else 0,
                })
            tagging_grp_rows.sort(key=lambda x: x["total"], reverse=True)

        # ── Untagged incidents list ───────────────────────────────────────────
        untag_cols = [c for c in [inc_col, "AssignedGroup", "Priority", "Status",
                                   "Request_Type01"] if c in df.columns]
        untagged_rows = none_df[untag_cols].head(300).fillna("—").to_dict("records")

        # ── Per-group KB coverage (existing) ─────────────────────────────────
        if F["AssignedGroup"]:
            for gname, gdf in df.groupby("AssignedGroup"):
                grp_rkm  = gdf[gdf["Request_Type01"].str.strip().str.lower().isin(RKM_VALS)]
                grp_nokb = gdf[~gdf.index.isin(grp_rkm.index)]
                kb_ids   = grp_rkm["_kb_id"].dropna().value_counts().head(5).reset_index()
                kb_ids.columns = ["KB_ID","count"]
                kb_group_rows.append({
                    "group":      gname, "total": len(gdf),
                    "with_kb":    len(grp_rkm), "without_kb": len(grp_nokb),
                    "top_kbs":    kb_ids.to_dict("records"),
                    "kb_pct":     round(len(grp_rkm)/len(gdf)*100,1) if len(gdf) else 0,
                })
            kb_group_rows.sort(key=lambda x: x["with_kb"], reverse=True)

        no_cols = [c for c in [inc_col,"AssignedGroup","Priority","Status"] if c in df.columns]
        kb_no_kb_rows = none_df[no_cols].head(200).fillna("—").to_dict("records")

        # ── Request Description tables — simple pivot, no parsing ────────────────
        # For each tag type (RKM Solution / Known Error / Problem Investigation):
        #   Group by Request_Desc01 AS-IS → count incidents → show AssignedGroup
        # Column names: Request_Description | Incident_Count | AssignedGroup

        def make_desc_by_group(source_df, tag_label):
            """
            Returns list of group-panel dicts, one per AssignedGroup.
            Each dict:
              { group, total_count, rows: [{Request_Description, Incident_Count, pct}] }
            Inside each group, rows are sorted by Incident_Count desc.
            If no AssignedGroup column, returns single "All" group.
            """
            if source_df is None or len(source_df) == 0:
                return []
            work = source_df.copy()
            desc_col = "Request_Desc01"
            work[desc_col] = work[desc_col].fillna("(blank)").astype(str).str.strip()
            work[desc_col] = work[desc_col].replace("", "(blank)")

            groups_out = []
            if F["AssignedGroup"] and "AssignedGroup" in work.columns:
                for gname, gdf in work.groupby("AssignedGroup", dropna=False):
                    desc_counts = (
                        gdf.groupby(desc_col, dropna=False)
                        .size()
                        .reset_index(name="Incident_Count")
                        .sort_values("Incident_Count", ascending=False)
                    )
                    g_total = len(gdf)
                    max_c   = int(desc_counts["Incident_Count"].max()) if len(desc_counts) else 1
                    rows_out = []
                    for _, r in desc_counts.iterrows():
                        cnt = int(r["Incident_Count"])
                        rows_out.append({
                            "Request_Description": r[desc_col],
                            "Incident_Count":      cnt,
                            "pct":                 round(cnt / max_c * 100),
                        })
                    groups_out.append({
                        "group":       str(gname),
                        "total_count": g_total,
                        "rows":        rows_out,
                    })
                groups_out.sort(key=lambda x: x["total_count"], reverse=True)
            else:
                # No group column — single panel "All"
                desc_counts = (
                    work.groupby(desc_col, dropna=False)
                    .size()
                    .reset_index(name="Incident_Count")
                    .sort_values("Incident_Count", ascending=False)
                )
                max_c = int(desc_counts["Incident_Count"].max()) if len(desc_counts) else 1
                groups_out = [{"group": "All", "total_count": len(work),
                               "rows": [{"Request_Description": r[desc_col],
                                         "Incident_Count": int(r["Incident_Count"]),
                                         "pct": round(int(r["Incident_Count"])/max_c*100)}
                                        for _, r in desc_counts.iterrows()]}]
            return groups_out

        def make_incident_by_group(source_df, detail_cols, label):
            """
            Returns list of group-panel dicts for flat incident lists.
            Each dict: { group, total_count, rows: [col→value ...] }
            """
            if source_df is None or len(source_df) == 0:
                return []
            work = source_df.copy()
            keep = [c for c in detail_cols if c in work.columns]
            groups_out = []
            if F["AssignedGroup"] and "AssignedGroup" in work.columns:
                for gname, gdf in work.groupby("AssignedGroup", dropna=False):
                    groups_out.append({
                        "group":       str(gname),
                        "total_count": len(gdf),
                        "rows":        gdf[keep].fillna("—").to_dict("records"),
                    })
                groups_out.sort(key=lambda x: x["total_count"], reverse=True)
            else:
                groups_out = [{"group":"All","total_count":len(work),
                               "rows": work[keep].fillna("—").to_dict("records")}]
            return groups_out

        # ── Build all five grouped structures ─────────────────────────────────
        kb_article_rows = make_desc_by_group(rkm_df, "RKM Solution")
        ke_desc_rows    = make_desc_by_group(ke_df,  "Known Error")
        pi_desc_rows    = make_desc_by_group(ps_df,  "Problem Investigation")

        # Untagged incidents: all columns (no limit)
        untag_detail_cols = [inc_col, "Priority", "Request_Type01", "Status", "Assignee"]
        untagged_grp_rows = make_incident_by_group(none_df, untag_detail_cols, "Untagged")

        # No-KB incidents = ALL incidents where Request_Desc01 has no KBxxxxxx pattern
        # Source: entire df (not just none_df) filtered where _kb_id is null
        # _kb_id was set earlier: df["_kb_id"] = df["Request_Desc01"].apply(extract_kb)
        nokb_src = df[df["_kb_id"].isna()].copy() if "_kb_id" in df.columns else none_df
        nokb_detail_cols = [inc_col, "Status", "Assignee", "Service_Type"]
        nokb_grp_rows    = make_incident_by_group(nokb_src, nokb_detail_cols, "No KB")

        log.info("Desc groups — RKM:%d  KE:%d  PI:%d  Untag:%d  NoKB:%d",
                 len(kb_article_rows), len(ke_desc_rows), len(pi_desc_rows),
                 len(untagged_grp_rows), len(nokb_grp_rows))

    # ── Chart generation — runs twice: once per theme ────────────────────────
    # Each call applies the appropriate matplotlib colour palette, renders all
    # charts, then returns a {key: base64_png} dict.
    def _gen_all_charts(theme_name):
        global BG, SURFACE, BORDER, MUTED, TEXT
        p = _apply_rc(theme_name)
        BG = p["BG"]; SURFACE = p["SURFACE"]; BORDER = p["BORDER"]
        MUTED = p["MUTED"]; TEXT = p["TEXT"]

        kbc = {}
        if kb_available and tagging_summary:
            kbc["tagging_donut"] = make_donut(
                tagging_summary["labels"], tagging_summary["values"],
                colors=tagging_summary["colors"],
                title="Request Type Tagging Distribution"
            )
            if tagging_grp_rows:
                g_names = [r["group"] for r in tagging_grp_rows]
                kbc["tagging_stacked"] = make_stacked_bar(
                    g_names,
                    [
                        ("RKM Solution",    [r["rkm"]      for r in tagging_grp_rows], PURPLE),
                        ("Known Error",     [r["ke"]       for r in tagging_grp_rows], ORANGE),
                        ("Problem Investigation",[r["ps"]  for r in tagging_grp_rows], CYAN),
                        ("Untagged",        [r["untagged"] for r in tagging_grp_rows], RED),
                    ],
                    title="Tagging by AssignedGroup",
                    ylabel="Incidents",
                )
            kbc = {k: v for k, v in kbc.items() if v}

        c = {}
        c["pri"]       = make_donut(pri_labels, pri_data,
                                    colors=[PRI_COLORS.get(l, PALETTE[i]) for i,l in enumerate(pri_labels)],
                                    title="Priority Distribution")
        c["sla_donut"] = make_donut(sla_labels, sla_data,
                                    colors=[sla_color_for(l) for l in sla_labels],
                                    title="SLA Status")
        c["grp"]       = make_hbar(grp_labels[:20], grp_data[:20], color=BLUE,
                                   title="Incidents by AssignedGroup", xlabel="Count")
        c["vol"]       = make_line(vol_labels, vol_data, color=BLUE,
                                   title="Monthly Incident Volume", ylabel="Incidents")
        c["dow"]       = make_vbar(dow_labels, dow_data,
                                   colors=[PALETTE[i%len(PALETTE)] for i in range(len(dow_labels))],
                                   title="Incidents by Day of Week", ylabel="Count")
        c["xfer"]      = make_vbar(xf_labels, xf_data, colors=[PURPLE]*len(xf_labels),
                                   title="Group Transfers Distribution", ylabel="Incidents")
        c["sla_trend"] = make_line(sla_tr_labels, sla_tr_vals, color=GREEN,
                                   title="Monthly SLA Compliance %", ylabel="SLA %", pct=True)
        c["sla_pri"]   = make_hbar_colored(sla_pri_labels, sla_pri_vals,
                                           title="SLA % by Priority", xlabel="SLA Met %")
        c["mttr_grp"]  = make_hbar(mttr_grp_labels, mttr_grp_vals, color=YELLOW,
                                   title="MTTR by Group (avg hrs)", xlabel="Hours")
        c["ci"]        = make_hbar(ci_labels[:15], ci_data[:15], color=CYAN,
                                   title="Top HPD_CI by Incident Count", xlabel="Count")
        c["svc"]       = make_donut(svc_labels, svc_data, title="Service Type Mix")
        if pri_heat:
            c["heatmap"] = make_heatmap(pri_heat["months"], pri_heat["priorities"],
                                        pri_heat["values"])

        # Weekly charts
        if wk_vol_labels:
            c["wk_vol"] = make_line(
                wk_vol_labels, wk_vol_data, color=CYAN,
                title="Weekly Incident Volume", ylabel="Incidents"
            )
        if wk_sla_labels:
            c["wk_sla"] = make_line(
                wk_sla_labels, wk_sla_vals, color=GREEN,
                title="Weekly SLA Compliance %", ylabel="SLA %", pct=True
            )
        if wk_heat:
            c["wk_heatmap"] = make_heatmap(
                wk_heat["weeks"], wk_heat["priorities"], wk_heat["values"],
                title="Priority × Week Heatmap", size=(max(10, len(wk_heat["weeks"]) * 0.55), 3.5)
            )
        if svc_sla_labels:
            c["svc_sla"] = make_hbar_colored(svc_sla_labels, svc_sla_vals,
                                             title="SLA Compliance % by Service Type", xlabel="SLA Met %")
        if svc_mttr_labels:
            c["svc_mttr"] = make_hbar(svc_mttr_labels, svc_mttr_vals, color=YELLOW,
                                      title="Avg MTTR by Service Type (hrs)", xlabel="Hours")
        if svc_labels:
            c["svc_donut"] = make_donut(svc_labels, svc_data,
                                        colors=[PALETTE[i%len(PALETTE)] for i in range(len(svc_labels))],
                                        title="Incident Volume by Service Type")
        if svc_vol_months and svc_vol_series:
            svc_series_fc = [(name, vals, PALETTE[i%len(PALETTE)])
                             for i, (name, vals) in enumerate(svc_vol_series)]
            c["svc_trend"] = make_stacked_bar(svc_vol_months, svc_series_fc,
                                              title="Monthly Volume by Service Type", ylabel="Incidents")
        c = {k: v for k, v in c.items() if v}
        return c, kbc

    charts,       kb_charts       = _gen_all_charts("dark")
    charts_light, kb_charts_light = _gen_all_charts("light")

    # Restore dark as default for any subsequent module-level calls
    _p = _apply_rc("dark")
    BG = _p["BG"]; SURFACE = _p["SURFACE"]; BORDER = _p["BORDER"]
    MUTED = _p["MUTED"]; TEXT = _p["TEXT"]

    return {
        "total": total, "closed_ct": closed_ct,
        "date_min": date_min, "date_max": date_max,
        "sla_met_ct": sla_met_ct, "sla_pct": sla_pct,
        "mttr": mttr, "mttr_source": mttr_source,
        "p1_ct": p1_ct, "health": health,
        "detected_cols": [c for c in df.columns if not c.startswith("_")],
        "has_kb": kb_available,
        "feature_flags": {k: v for k, v in F.items()},
        "charts":        charts,
        "charts_light":  charts_light,
        "kb_charts":       kb_charts       if kb_available else {},
        "kb_charts_light": kb_charts_light if kb_available else {},
        "grp_count": len(grp_labels),
        "wk_vol_labels":  wk_vol_labels,
        "wk_sla_labels":  wk_sla_labels,
        "wk_heat_priorities": wk_heat.get("priorities",[]) if wk_heat else [],
        "svc_sla_labels": svc_sla_labels, "svc_sla_vals": svc_sla_vals,
        "svc_mttr_labels": svc_mttr_labels, "svc_mttr_vals": svc_mttr_vals,
        "ci_gt3_count": len(ci_gt3_list),
        "mttr_alert_total": sum(g["count"] for g in mttr_grp_alerts),
        "hop_alert_total":  sum(g["count"] for g in hop_grp_alerts),
        "age_alert_total":  len(age_alert_rows),
        "mttr_grp_alerts": mttr_grp_alerts,
        "hop_grp_alerts":  hop_grp_alerts,
        "age_alert_rows":  age_alert_rows,
        "org_rows": org_rows,
        "group_tables": group_tables,
        "ci_gt3_rows": ci_gt3_list,
        "ci_by_group": ci_by_group,
        "kb_summary": kb_summary_data,
        "kb_group_rows": kb_group_rows,
        "kb_no_kb_rows": kb_no_kb_rows,
        "kb_article_rows":    kb_article_rows,    # RKM Solution — grouped by AssignedGroup
        "ke_desc_rows":       ke_desc_rows,        # Known Error  — grouped by AssignedGroup
        "pi_desc_rows":       pi_desc_rows,         # Problem Investigation — grouped
        "untagged_grp_rows":  untagged_grp_rows,   # Untagged incidents — grouped
        "nokb_grp_rows":      nokb_grp_rows,        # No-KB incidents — grouped
        "tagging_summary":    tagging_summary,
        "tagging_grp_rows":   tagging_grp_rows,
        "untagged_rows":      untagged_grp_rows,    # kept for compatibility
        # new analytical features
        "ftf_rows":           ftf_rows,
        "p1_breach_rows":     p1_breach_rows,
        "repeat_ci_rows":     repeat_ci_rows,
        "hop_dist_rows":      hop_dist_rows,
        "mttr_dist_rows":     mttr_dist_rows,
        # service type analytics (for SLA tab)
        "svc_vol_months":     svc_vol_months   if F["Service_Type"] else [],
        "svc_vol_series":     svc_vol_series   if F["Service_Type"] else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────
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
    try: fb = io.BytesIO(f.read())
    except Exception as e: return jsonify({"error": f"Could not receive file: {e}"}), 400
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
    log.info("Read OK — %d rows, cols: %s", len(df), list(df.columns)[:8])
    try:
        result = analyse(df)
        log.info("Done — health:%d  rows:%d  charts:%d  KB:%s",
                 result["health"], result["total"], len(result["charts"]), result["has_kb"])
    except Exception as e:
        log.exception("Analysis failed"); return jsonify({"error": f"Analysis failed: {e}"}), 500
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE — zero external dependencies, 100% offline
# ─────────────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DCSS Incident Analyzer</title>
<!-- NO CDN. NO INTERNET REQUIRED. All charts are server-side PNG images. -->
<style>
/* ── DARK THEME (default) ── */
:root,[data-theme="dark"]{
  --bg:#07090f;--surf:#0d111e;--card:#111827;--card2:#161e30;
  --border:#1e2a40;--border2:#263047;
  --blue:#4a8cff;--red:#ff4f6a;--green:#30d988;--yellow:#ffc240;
  --purple:#a78bfa;--cyan:#22d3ee;--orange:#fb923c;
  --text:#edf2ff;--muted:#7b8db0;--dim:#3a4b6b;
  --hdr-bg:rgba(7,9,15,.96);--spin-bg:rgba(7,9,15,.88);
}
/* ── LIGHT THEME ── */
[data-theme="light"]{
  --bg:#f0f4f8;--surf:#e2e8f0;--card:#ffffff;--card2:#f8fafc;
  --border:#cbd5e1;--border2:#94a3b8;
  --blue:#2563eb;--red:#dc2626;--green:#16a34a;--yellow:#d97706;
  --purple:#7c3aed;--cyan:#0891b2;--orange:#ea580c;
  --text:#0f172a;--muted:#475569;--dim:#94a3b8;
  --hdr-bg:rgba(240,244,248,.97);--spin-bg:rgba(240,244,248,.92);
}
/* ── Light mode structural overrides ── */
[data-theme="light"] .tabs{background:var(--surf)}
[data-theme="light"] .cc,[data-theme="light"] .al-card,
[data-theme="light"] .grp-panel,[data-theme="light"] .kpi,
[data-theme="light"] .ucard{background:var(--card);border-color:var(--border)}
[data-theme="light"] .grp-hdr{background:var(--card2)}
[data-theme="light"] .drop{background:var(--card2);border-color:var(--border2)}
[data-theme="light"] .drop:hover,[data-theme="light"] .drop.drag{background:rgba(37,99,235,.06);border-color:var(--blue)}
[data-theme="light"] .new-btn{background:var(--card);border-color:var(--border)}
[data-theme="light"] thead th{background:var(--surf)}
[data-theme="light"] tbody tr:hover td{background:rgba(37,99,235,.04)}
[data-theme="light"] .search-box{background:var(--card);border-color:var(--border2);color:var(--text)}
[data-theme="light"] .tbl-toolbar{background:var(--surf);border-color:var(--border)}
[data-theme="light"] .export-btn{background:rgba(37,99,235,.1);border-color:rgba(37,99,235,.3);color:var(--blue)}
[data-theme="light"] .mbar-bg{background:var(--border)}
[data-theme="light"] .al-body,[data-theme="light"] .grp-body{background:var(--card)}
[data-theme="light"] .health-pill,[data-theme="light"] .date-bar{background:var(--card);border-color:var(--border)}
[data-theme="light"] #upload-section{background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(37,99,235,.07),transparent 70%)}
[data-theme="light"] .offline-badge{background:rgba(22,163,74,.1);border-color:rgba(22,163,74,.3);color:var(--green)}
[data-theme="light"] .fmt{background:var(--surf);border-color:var(--border);color:var(--muted)}
[data-theme="light"] .chip.c-blue{background:rgba(37,99,235,.12);color:var(--blue)}
[data-theme="light"] .chip.c-met{background:rgba(22,163,74,.15);color:var(--green)}
[data-theme="light"] .chip.c-breach{background:rgba(220,38,38,.15);color:var(--red)}
[data-theme="light"] .chip.c-pend{background:rgba(217,119,6,.15);color:var(--yellow)}
[data-theme="light"] .chip.c-purple{background:rgba(124,58,237,.12);color:var(--purple)}
[data-theme="light"] .chip.c-orange{background:rgba(234,88,12,.12);color:var(--orange)}
[data-theme="light"] .chip.c-cyan{background:rgba(8,145,178,.12);color:var(--cyan)}
[data-theme="light"] .tab:hover{color:var(--text)}
[data-theme="light"] .tab.on{color:var(--blue);border-bottom-color:var(--blue)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;min-height:100vh}
/* HEADER */
.hdr{position:sticky;top:0;z-index:500;background:var(--hdr-bg);
  border-bottom:1px solid var(--border);padding:12px 26px;
  display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:15px}
.brand-name{font-size:.92rem;font-weight:800;letter-spacing:-.02em}
.brand-name span{color:var(--blue)}
.hdr-right{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.pill{background:var(--card);border:1px solid var(--border);padding:3px 10px;border-radius:18px;
  font-size:.65rem;color:var(--muted);font-family:monospace}
.pill.live{border-color:var(--green);color:var(--green)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);
  box-shadow:0 0 6px var(--green);display:inline-block;margin-right:4px;
  animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* HEALTH */
.health-pill{display:flex;align-items:center;gap:5px;background:var(--card);
  border:1px solid var(--border);padding:3px 10px;border-radius:18px;font-size:.67rem;font-weight:700}
/* UPLOAD */
#upload-section{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:calc(100vh - 58px);padding:40px 20px;
  background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(74,140,255,.07),transparent 70%)}
.ucard{width:100%;max-width:580px;background:var(--card);border:1px solid var(--border);
  border-radius:18px;padding:40px 36px;text-align:center;position:relative;overflow:hidden}
.ucard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--blue),var(--purple),var(--cyan))}
.u-title{font-size:1.5rem;font-weight:800;letter-spacing:-.03em;margin-bottom:6px}
.u-title span{color:var(--blue)}
.u-sub{color:var(--muted);font-size:.83rem;margin-bottom:24px;line-height:1.65}
.offline-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(48,217,136,.1);
  border:1px solid rgba(48,217,136,.3);color:var(--green);padding:4px 12px;border-radius:20px;
  font-size:.72rem;font-weight:700;margin-bottom:18px}
.drop{border:2px dashed var(--border2);border-radius:12px;padding:34px 20px;cursor:pointer;
  transition:all .2s;position:relative;background:var(--card2)}
.drop:hover,.drop.drag{border-color:var(--blue);background:rgba(74,140,255,.06)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.fmts{margin-top:8px;display:flex;gap:6px;justify-content:center}
.fmt{background:var(--surf);border:1px solid var(--border);padding:2px 9px;border-radius:5px;
  font-size:.65rem;font-family:monospace;color:var(--muted)}
.ubtn{margin-top:18px;background:linear-gradient(135deg,var(--blue),var(--purple));border:none;
  color:#fff;padding:12px 32px;border-radius:11px;font-size:.9rem;font-weight:700;cursor:pointer;
  font-family:inherit;transition:opacity .2s,transform .15s;width:100%}
.ubtn:hover{opacity:.9;transform:translateY(-1px)}
.ubtn:disabled{opacity:.38;cursor:not-allowed;transform:none}
.fname{margin-top:9px;font-size:.75rem;color:var(--green);font-family:monospace}
.alert-box{background:rgba(255,79,106,.1);border:1px solid rgba(255,79,106,.3);
  border-radius:9px;padding:11px 14px;color:var(--red);font-size:.82rem;margin-top:10px}
#spin{display:none;position:fixed;inset:0;background:var(--spin-bg);z-index:900;
  align-items:center;justify-content:center;flex-direction:column;gap:12px}
#spin.show{display:flex}
.spinner{width:44px;height:44px;border:3px solid var(--border2);border-top-color:var(--blue);
  border-radius:50%;animation:rot .8s linear infinite}
.spin-sub{color:var(--muted);font-size:.82rem;margin-top:4px}
@keyframes rot{to{transform:rotate(360deg)}}
/* DASHBOARD */
#dash{display:none}#dash.show{display:block}
.new-btn{display:inline-flex;align-items:center;gap:7px;background:var(--card);
  border:1px solid var(--border);color:var(--muted);padding:8px 16px;border-radius:9px;
  cursor:pointer;font-family:inherit;font-size:.78rem;font-weight:600;transition:all .2s;
  margin:14px 26px 0}
.new-btn:hover{border-color:var(--blue);color:var(--blue)}
/* TABS */
.tabs{background:var(--surf);border-bottom:1px solid var(--border);padding:0 26px;
  display:flex;gap:2px;overflow-x:auto;position:sticky;top:58px;z-index:400}
.tab{padding:11px 16px;cursor:pointer;font-size:.78rem;font-weight:600;color:var(--muted);
  border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;user-select:none}
.tab:hover{color:var(--text)}.tab.on{color:var(--blue);border-bottom-color:var(--blue)}
.kb-tab{display:none}.kb-tab.show{display:block}
.page{display:none;padding:20px 26px}.page.on{display:block}
/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:16px 14px;position:relative;overflow:hidden;transition:transform .15s,border-color .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi-bar{position:absolute;top:0;left:0;right:0;height:2px}
.kpi-lbl{font-size:.61rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:7px}
.kpi-val{font-size:1.7rem;font-weight:800;font-family:monospace;line-height:1;letter-spacing:-.02em}
.kpi-sub{font-size:.61rem;color:var(--dim);margin-top:4px}
.kpi-icon{position:absolute;right:12px;top:12px;font-size:1.1rem;opacity:.2}
.date-bar{grid-column:1/-1;background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:10px 15px;display:flex;align-items:center;gap:14px;font-size:.77rem;flex-wrap:wrap}
/* SECTION */
.sec{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);margin:18px 0 10px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
/* CARD */
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:17px}
.cc-hd{display:flex;align-items:center;gap:7px;margin-bottom:12px}
.cc-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.cc-title{font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.cc-badge{margin-left:auto;background:var(--surf);border:1px solid var(--border);
  padding:2px 7px;border-radius:7px;font-size:.62rem;color:var(--dim);font-family:monospace}
/* GRIDS */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.full{margin-bottom:14px}
/* CHART IMG */
.chart-img{width:100%;height:auto;border-radius:8px;display:block}
/* ALERT */
.al-card{background:var(--card);border-radius:12px;border:1px solid var(--border);margin-bottom:14px;overflow:hidden}
.al-hdr{padding:12px 16px;display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--border);cursor:pointer;user-select:none}
.al-info .al-title{font-size:.8rem;font-weight:700}
.al-info .al-sub{font-size:.65rem;color:var(--muted);margin-top:1px}
.al-badge{margin-left:auto;padding:3px 10px;border-radius:8px;font-size:.68rem;font-weight:700;flex-shrink:0}
.al-expand{font-size:.78rem;color:var(--dim);margin-left:4px;flex-shrink:0;transition:transform .2s}
.al-hdr.open .al-expand{transform:rotate(180deg)}
.al-body{display:none}.al-body.open{display:block}
.al-none{padding:14px 16px;color:var(--dim);font-size:.8rem}
/* GROUP PANELS */
.grp-panel{background:var(--card);border:1px solid var(--border);border-radius:12px;margin-bottom:12px;overflow:hidden}
.grp-hdr{padding:12px 16px;background:var(--card2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;flex-wrap:wrap}
.gname{font-weight:700;font-size:.86rem;letter-spacing:-.01em}
.grp-stats{display:flex;gap:12px;margin-left:auto;flex-wrap:wrap}
.gs{font-size:.7rem;color:var(--muted);font-family:monospace}
.gs strong{color:var(--text)}
.g-expand{font-size:.78rem;color:var(--dim);margin-left:4px;flex-shrink:0;transition:transform .2s}
.grp-hdr.open .g-expand{transform:rotate(180deg)}
.grp-body{display:none}.grp-body.open{display:block}
/* TOOLBAR */
.tbl-toolbar{padding:9px 14px;display:flex;align-items:center;gap:8px;
  border-bottom:1px solid var(--border);background:var(--surf)}
.search-box{flex:1;background:var(--card);border:1px solid var(--border2);border-radius:7px;
  padding:5px 10px;color:var(--text);font-family:monospace;font-size:.75rem;outline:none}
.search-box:focus{border-color:var(--blue)}
.export-btn{background:rgba(74,140,255,.12);border:1px solid rgba(74,140,255,.3);color:var(--blue);
  padding:5px 11px;border-radius:7px;cursor:pointer;font-size:.7rem;font-weight:700;
  font-family:inherit;transition:all .15s;white-space:nowrap}
.export-btn:hover{background:rgba(74,140,255,.22)}
/* TABLES */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.76rem}
thead th{padding:8px 10px;background:var(--surf);color:var(--muted);font-weight:700;
  font-size:.63rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;
  border-bottom:1px solid var(--border);white-space:nowrap}
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
/* FOOTER */
footer{text-align:center;padding:18px;color:var(--dim);font-size:.67rem;
  border-top:1px solid var(--border);margin-top:4px;line-height:1.8}
@media(max-width:1100px){.kpi-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.g2,.g3{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}}
@media(max-width:500px){.kpi-row{grid-template-columns:1fr}.page{padding:13px 11px}}
</style>
</head>
<body>

<header class="hdr">
  <div class="brand">
    <div class="brand-icon">⚡</div>
    <div class="brand-name">DCSS <span>Incident</span> Analyzer</div>
  </div>
  <div class="hdr-right">
    <span class="pill live"><span class="dot"></span>Live</span>
    <span class="pill" style="color:var(--green);border-color:rgba(48,217,136,.3)">🔌 Offline</span>
    <span class="pill" id="file-pill">No file</span>
    <span class="pill" id="mttr-pill" style="display:none"></span>
    <div class="health-pill" id="health-pill" style="display:none">
      Health: <span id="health-val" style="color:var(--green)">—</span>
    </div>
    <button id="theme-btn" onclick="toggleTheme()"
      style="background:var(--card);border:1px solid var(--border);color:var(--muted);
             padding:4px 12px;border-radius:18px;cursor:pointer;font-size:.76rem;
             font-family:inherit;display:flex;align-items:center;gap:5px;
             transition:all .2s;white-space:nowrap;line-height:1">
      <span id="theme-icon">☀️</span><span id="theme-label">Light</span>
    </button>
  </div>
</header>

<div id="spin">
  <div class="spinner"></div>
  <div style="color:var(--text);font-weight:700;font-size:.95rem">Analysing Incident Data…</div>
  <div class="spin-sub">Generating charts server-side (offline mode)</div>
</div>

<!-- UPLOAD -->
<section id="upload-section">
  <div class="ucard">
    <div class="u-title">DCSS <span>Incident</span> Analyzer</div>
    <div class="offline-badge">🔌 100% Offline — No Internet Required</div>
    <div class="u-sub">Upload any DCSS incident Excel / CSV export.<br>
      Columns auto-detected with fuzzy matching.<br>
      All charts generated locally — works without any CDN.</div>
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

<!-- DASHBOARD -->
<div id="dash">
  <button class="new-btn" onclick="reset()">← Upload New File</button>
  <nav class="tabs">
    <div class="tab on"  onclick="go('ov',this)">📊 Overview</div>
    <div class="tab"     onclick="go('tr',this)">📈 Trends</div>
    <div class="tab"     onclick="go('sl',this)">🎯 SLA</div>
    <div class="tab"     onclick="go('tm',this)">👥 Team</div>
    <div class="tab"     onclick="go('ci',this)">🖥 CI &amp; Service</div>
    <div class="tab kb-tab" id="kb-tab" onclick="go('kb',this)">🏷 Incident Tagging</div>
    <div class="tab"     onclick="go('af',this)">💡 Advanced Insights</div>
    <div class="tab"     onclick="go('nf',this)">🔍 Data Info</div>
  </nav>

  <!-- OVERVIEW -->
  <section class="page on" id="pg-ov">
    <div class="kpi-row" id="kpi-row"></div>
    <div class="g3" id="ov-charts-top"></div>
    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span>
      <span class="cc-title">Organisation Summary</span></div>
      <div class="tbl-wrap"><table><thead><tr><th>Organisation</th><th>Total</th><th>Open</th><th>SLA %</th><th>MTTR (hrs)</th><th>SLA Bar</th></tr></thead>
      <tbody id="org-tbody"></tbody></table></div>
    </div>
    <div class="sec">🚨 Alerts &amp; Exceptions</div>
    <!-- MTTR > 200 -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(255,79,106,.06);border-left:3px solid var(--red)" onclick="toggleAl(this,'mttr-body')">
        <span style="font-size:1rem">⏱</span>
        <div class="al-info"><div class="al-title">MTTR &gt; 200 Hours — by AssignedGroup</div>
          <div class="al-sub">Incidents with extreme resolution time, grouped by team</div></div>
        <span class="al-badge" id="mttr-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="mttr-body"><div class="al-none">No incidents exceed 200 hours ✓</div></div>
    </div>
    <!-- HOP > 5 -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(251,146,60,.06);border-left:3px solid var(--orange)" onclick="toggleAl(this,'hop-body')">
        <span style="font-size:1rem">🔄</span>
        <div class="al-info"><div class="al-title">High HOP Count — Group Transfers ≥ 5 — by AssignedGroup</div>
          <div class="al-sub">Incidents bounced across more than 5 groups</div></div>
        <span class="al-badge" id="hop-badge" style="background:rgba(251,146,60,.15);color:var(--orange)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="hop-body"><div class="al-none">No incidents have Group Transfers ≥ 5 ✓</div></div>
    </div>
    <!-- Aging > 30d -->
    <div class="al-card">
      <div class="al-hdr" style="background:rgba(255,194,64,.06);border-left:3px solid var(--yellow)" onclick="toggleAl(this,'age-body')">
        <span style="font-size:1rem">📅</span>
        <div class="al-info"><div class="al-title">Incident Aging &gt; 30 Days</div>
          <div class="al-sub">Resolution date minus Submit/Reported date exceeds 30 days</div></div>
        <span class="al-badge" id="age-badge" style="background:rgba(255,194,64,.15);color:var(--yellow)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="age-body"><div class="al-none">No incidents exceed 30 days ✓</div></div>
    </div>
  </section>

  <!-- TRENDS -->
  <section class="page" id="pg-tr">

    <div class="sec">📅 Monthly View</div>

    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Monthly Incident Volume</span></div><div id="c-vol"></div></div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">Incidents by Day of Week</span></div><div id="c-dow"></div></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Group Transfers Distribution</span></div><div id="c-xfer"></div></div>
    </div>
    <div class="full cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance Trend — Monthly %</span></div><div id="c-sla-trend"></div></div>
    <div class="full cc" id="heatmap-card" style="display:none">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Priority × Month Heatmap</span></div><div id="c-heatmap"></div>
    </div>

    <div class="sec">📆 Weekly View <span style="font-size:.68rem;font-weight:400;color:var(--dim);margin-left:6px">ISO week numbers (Mon–Sun) · shown only when date column is present</span></div>

    <div class="full cc" id="wk-vol-card" style="display:none">
      <div class="cc-hd">
        <span class="cc-dot" style="background:var(--cyan)"></span>
        <span class="cc-title">Weekly Incident Volume</span>
        <span class="cc-badge" id="wk-vol-badge"></span>
      </div>
      <div id="c-wk-vol"></div>
    </div>

    <div class="full cc" id="wk-sla-card" style="display:none">
      <div class="cc-hd">
        <span class="cc-dot" style="background:var(--green)"></span>
        <span class="cc-title">SLA Compliance Trend — Weekly %</span>
        <span class="cc-badge" id="wk-sla-badge"></span>
      </div>
      <div id="c-wk-sla"></div>
    </div>

    <div class="full cc" id="wk-heat-card" style="display:none">
      <div class="cc-hd">
        <span class="cc-dot" style="background:var(--purple)"></span>
        <span class="cc-title">Priority × Week Heatmap</span>
        <span class="cc-badge" id="wk-heat-badge"></span>
      </div>
      <div id="c-wk-heatmap"></div>
    </div>

    <div id="wk-na" style="display:none;padding:16px;color:var(--dim);font-size:.82rem;
      background:var(--card);border:1px solid var(--border);border-radius:12px;margin-top:8px">
      ℹ Weekly charts are not available — no date column (ReportedDate / SubmitDate) found in this file.
    </div>

  </section>

  <!-- SLA -->
  <section class="page" id="pg-sl">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">SLA Compliance % by Priority</span></div><div id="c-sla-pri"></div></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span><span class="cc-title">MTTR by AssignedGroup (hrs)</span></div><div id="c-mttr-grp"></div></div>
    </div>

    <div class="sec">📋 Service Type Analysis</div>

    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span>
        <span class="cc-title">Incident Volume by Service Type</span></div>
        <div id="c-svc-donut"></div>
      </div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span>
        <span class="cc-title">SLA Compliance % by Service Type</span>
        <span class="cc-badge" style="background:rgba(48,217,136,.1);color:var(--green)">green ≥70% · yellow ≥50%</span></div>
        <div id="c-svc-sla"></div>
      </div>
    </div>

    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--yellow)"></span>
        <span class="cc-title">Avg MTTR by Service Type (hrs)</span></div>
        <div id="c-svc-mttr"></div>
      </div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span>
        <span class="cc-title">Monthly Volume by Service Type</span></div>
        <div id="c-svc-trend"></div>
      </div>
    </div>

    <!-- SLA by Service Type — summary table -->
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--cyan)"></span>
        <span class="cc-title">SLA &amp; MTTR Summary by Service Type</span>
      </div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>Service Type</th><th>Total</th><th>SLA Met %</th><th>Avg MTTR (hrs)</th>
        <th>SLA Bar</th>
      </tr></thead><tbody id="svc-tbl-tbody"></tbody></table></div>
    </div>
  </section>

  <!-- TEAM -->
  <section class="page" id="pg-tm">
    <div class="sec">Per-Group Performance — click header to expand assignees</div>
    <div id="team-panels"></div>
  </section>

  <!-- CI -->
  <section class="page" id="pg-ci">
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--cyan)"></span><span class="cc-title">Top HPD_CI by Count</span></div><div id="c-ci"></div></div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Service Type Mix</span></div><div id="c-svc"></div></div>
    </div>

    <!-- NEW: CI ≥3 grouped by AssignedGroup — collapsible panels -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--blue)" onclick="toggleAl(this,'ci-grp-body')">
        <span style="font-size:1rem">👥</span>
        <div class="al-info">
          <div class="al-title">HPD_CI (≥ 3 incidents) — Grouped by AssignedGroup</div>
          <div class="al-sub">Expand each group to see which assets/servers that group handles most</div>
        </div>
        <span class="al-badge" id="ci-grp-badge" style="background:rgba(74,140,255,.12);color:var(--blue)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="ci-grp-body">
        <div id="ci-grp-panels"></div>
      </div>
    </div>

    <!-- Flat CI table with AssignedGroup breakdown -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--cyan);cursor:default">
        <span style="font-size:1rem">🖥</span>
        <div class="al-info"><div class="al-title">HPD_CI Incident Count ≥ 3 — with AssignedGroup Breakdown</div></div>
        <span class="al-badge" id="ci3-badge" style="background:rgba(34,211,238,.12);color:var(--cyan)">0</span>
      </div>
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search CI or group…" oninput="filterTbl(this,'ci3-tbody')"/>
        <button class="export-btn" onclick="exportCSV('ci3-tbody','ci_repeat_offenders')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead><tr><th>#</th><th>HPD_CI</th><th>Total</th><th>AssignedGroup Breakdown</th><th>Volume</th></tr></thead>
      <tbody id="ci3-tbody"></tbody></table></div>
    </div>
  </section>

  <!-- KB -->
  <section class="page" id="pg-kb">
    <!-- KPI row: 4 metrics -->
    <div class="kpi-row" style="grid-template-columns:repeat(4,1fr)" id="kb-kpis"></div>

    <!-- Tagging dashboard -->
    <div class="sec">📊 Request Type Tagging Dashboard</div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span>
        <span class="cc-title">Tagging Distribution — All Incidents</span></div><div id="c-tag-donut"></div>
      </div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--cyan)"></span>
        <span class="cc-title">Tagging Breakdown by AssignedGroup</span></div><div id="c-tag-stacked"></div>
      </div>
    </div>

    <!-- Per-group tagging table -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--purple);cursor:default">
        <span style="font-size:1rem">🏷</span>
        <div class="al-info">
          <div class="al-title">Tagging by AssignedGroup — RKM Solution · Known Error · Problem Investigation · Untagged</div>
          <div class="al-sub">Shows how each group uses the three tag types and how many incidents remain untagged</div>
        </div>
        <span class="al-badge" id="tag-grp-badge" style="background:rgba(167,139,250,.15);color:var(--purple)">0</span>
      </div>
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search group…" oninput="filterTbl(this,'tag-grp-tbody')"/>
        <button class="export-btn" onclick="exportCSV('tag-grp-tbody','tagging_by_group')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>AssignedGroup</th><th>Total</th>
        <th style="color:var(--purple)">RKM Solution</th>
        <th style="color:var(--orange)">Known Error</th>
        <th style="color:var(--cyan)">Problem Investigation</th>
        <th style="color:var(--red)">Untagged</th>
        <th>Tagged %</th><th>Coverage Bar</th>
      </tr></thead><tbody id="tag-grp-tbody"></tbody></table></div>
    </div>

    <!-- Untagged incidents — grouped by AssignedGroup, collapsible -->
    <div class="full al-card">
      <div class="al-hdr" style="background:rgba(255,79,106,.06);border-left:3px solid var(--red)" onclick="toggleAl(this,'untag-body')">
        <span style="font-size:1rem">🚫</span>
        <div class="al-info">
          <div class="al-title">Untagged Incidents — No RKM Solution, Known Error or Problem Investigation</div>
          <div class="al-sub">Grouped by AssignedGroup — click to expand · Shows all records</div>
        </div>
        <span class="al-badge" id="untag-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="untag-body">
        <div id="untag-panels"></div>
      </div>
    </div>

    <div class="sec">📚 KB Article Coverage (RKM Solution only)</div>

    <!-- KB group coverage summary -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--purple);cursor:default">
        <span style="font-size:1rem">📚</span>
        <div class="al-info">
          <div class="al-title">KB Article Coverage by AssignedGroup</div>
          <div class="al-sub">Request_Type01 = RKM Solution</div>
        </div>
      </div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>AssignedGroup</th><th>Total</th><th>With KB</th><th>Without KB</th><th>KB %</th><th>Top KB IDs</th><th>Coverage</th>
      </tr></thead><tbody id="kb-tbody"></tbody></table></div>
    </div>

    <!-- Incidents Without KB Article — grouped by AssignedGroup, collapsible -->
    <div class="full al-card">
      <div class="al-hdr" style="background:rgba(255,79,106,.06);border-left:3px solid var(--red)" onclick="toggleAl(this,'nokb-body')">
        <span style="font-size:1rem">❌</span>
        <div class="al-info">
          <div class="al-title">Incidents Without KB Article Tag</div>
          <div class="al-sub">Grouped by AssignedGroup — click to expand · Shows all records</div>
        </div>
        <span class="al-badge" id="nokb-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="nokb-body">
        <div id="nokb-panels"></div>
      </div>
    </div>

    <!-- ── Three Request Description sections — grouped by AssignedGroup ── -->
    <div class="sec">📋 Request Description — by Tag Type &amp; AssignedGroup</div>

    <!-- RKM Solution — grouped -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--purple)" onclick="toggleAl(this,'rkm-body')">
        <span style="font-size:1rem">📚</span>
        <div class="al-info">
          <div class="al-title">RKM Solution — Request Description · Incident Count by AssignedGroup</div>
          <div class="al-sub">Expand each group to see all descriptions and their incident counts</div>
        </div>
        <span class="al-badge" id="rkm-desc-badge" style="background:rgba(167,139,250,.15);color:var(--purple)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="rkm-body">
        <div id="rkm-desc-panels"></div>
      </div>
    </div>

    <!-- Known Error — grouped -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--orange)" onclick="toggleAl(this,'ke-body')">
        <span style="font-size:1rem">🔶</span>
        <div class="al-info">
          <div class="al-title">Known Error — Request Description · Incident Count by AssignedGroup</div>
          <div class="al-sub">Expand each group to see all descriptions and their incident counts</div>
        </div>
        <span class="al-badge" id="ke-desc-badge" style="background:rgba(251,146,60,.15);color:var(--orange)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="ke-body">
        <div id="ke-desc-panels"></div>
      </div>
    </div>

    <!-- Problem Investigation — grouped -->
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--cyan)" onclick="toggleAl(this,'pi-body')">
        <span style="font-size:1rem">🔷</span>
        <div class="al-info">
          <div class="al-title">Problem Investigation — Request Description · Incident Count by AssignedGroup</div>
          <div class="al-sub">Expand each group to see all descriptions and their incident counts</div>
        </div>
        <span class="al-badge" id="pi-desc-badge" style="background:rgba(34,211,238,.15);color:var(--cyan)">0</span>
        <span class="al-expand">▼</span>
      </div>
      <div class="al-body" id="pi-body">
        <div id="pi-desc-panels"></div>
      </div>
    </div>
  </section>

  <!-- ADVANCED INSIGHTS -->
  <section class="page" id="pg-af">
    <div class="sec">🏆 First-Time-Fix Rate by AssignedGroup</div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span>
        <span class="cc-title">First-Time-Fix Rate — Incidents Resolved Without Any Group Transfer</span>
        <span style="font-size:.7rem;color:var(--dim);margin-left:8px">Group_Transfers = 0</span>
      </div>
      <div class="tbl-wrap"><table><thead><tr>
        <th>AssignedGroup</th><th>Total Incidents</th><th>First-Time-Fix</th><th>FTF %</th><th>FTF Bar</th>
      </tr></thead><tbody id="ftf-tbody"></tbody></table></div>
    </div>

    <div class="sec">🚨 P1 Critical — SLA Breached Incidents</div>
    <div class="full al-card">
      <div class="al-hdr" style="border-left:3px solid var(--red);cursor:default">
        <span style="font-size:1rem">🚨</span>
        <div class="al-info">
          <div class="al-title">P1 Critical Incidents with SLA Breached</div>
          <div class="al-sub">Highest priority incidents that failed to meet SLA</div>
        </div>
        <span class="al-badge" id="p1b-badge" style="background:rgba(255,79,106,.15);color:var(--red)">0</span>
      </div>
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'p1b-tbody')"/>
        <button class="export-btn" onclick="exportCSV('p1b-tbody','p1_sla_breached')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead id="p1b-thead"></thead><tbody id="p1b-tbody"></tbody></table></div>
    </div>

    <div class="sec">🔁 Repeat Offender Assets — HPD_CI with 5+ Incidents</div>
    <div class="full cc">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span>
        <span class="cc-title">HPD_CI appearing in 5 or more incidents — chronic problem assets</span>
      </div>
      <div class="tbl-toolbar" style="padding:0 0 10px">
        <input class="search-box" placeholder="Search CI…" oninput="filterTbl(this,'rep-ci-tbody')"/>
        <button class="export-btn" onclick="exportCSV('rep-ci-tbody','repeat_offender_ci')">⬇ CSV</button>
      </div>
      <div class="tbl-wrap"><table><thead><tr><th>#</th><th>HPD_CI</th><th>Incident Count</th><th>Volume</th></tr></thead>
      <tbody id="rep-ci-tbody"></tbody></table></div>
    </div>

    <div class="g2">

      <!-- HOP Distribution Table -->
      <div class="cc">
        <div class="cc-hd">
          <span class="cc-dot" style="background:var(--purple)"></span>
          <span class="cc-title">HOP Count Distribution</span>
          <span class="cc-badge">Group_Transfers banded</span>
        </div>
        <div class="tbl-wrap"><table>
          <thead><tr>
            <th>HOPs (Group Transfers)</th>
            <th style="text-align:right">Incident Count</th>
            <th style="text-align:right">% of Total</th>
          </tr></thead>
          <tbody id="hop-dist-tbody"></tbody>
        </table></div>
        <div id="hop-dist-na" style="display:none;padding:14px;color:var(--dim);font-size:.8rem">
          Group_Transfers column not found in this file
        </div>
      </div>

      <!-- MTTR Distribution Table -->
      <div class="cc">
        <div class="cc-hd">
          <span class="cc-dot" style="background:var(--yellow)"></span>
          <span class="cc-title">MTTR Distribution — Resolution Time Bands</span>
          <span class="cc-badge">resolved incidents only</span>
        </div>
        <div class="tbl-wrap"><table>
          <thead><tr>
            <th>Resolution Time Band</th>
            <th style="text-align:right">Incident Count</th>
            <th style="text-align:right">% of Resolved</th>
          </tr></thead>
          <tbody id="mttr-dist-tbody"></tbody>
        </table></div>
        <div id="mttr-dist-na" style="display:none;padding:14px;color:var(--dim);font-size:.8rem">
          MTTR could not be calculated — check that date columns are present
        </div>
      </div>

    </div>
  </section>

  <!-- DATA INFO -->
  <section class="page" id="pg-nf">
    <div class="cc full"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Detected &amp; Mapped Columns</span></div>
      <div id="col-chips" style="display:flex;flex-wrap:wrap;gap:7px;margin-top:4px"></div>
    </div>
    <div class="g2">
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--green)"></span><span class="cc-title">MTTR Calculation Logic</span></div>
        <div id="mttr-logic" style="font-size:.8rem;color:var(--muted);line-height:1.8"></div>
      </div>
      <div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Column Detection Status</span></div>
        <div id="feat-flags" style="font-size:.75rem;color:var(--muted);line-height:1.9;font-family:monospace;max-height:260px;overflow-y:auto"></div>
      </div>
    </div>
  </section>

  <footer>
    DCSS Incident Analyzer · 100% Offline · Charts by Python matplotlib · Zero CDN dependencies<br>
    <span style="color:var(--muted)">Designed by <strong style="color:var(--blue)">aawasthi</strong> &nbsp;·&nbsp; <span id="footer-stats"></span></span>
  </footer>
</div>

<script>
// ── Pure vanilla JS — ZERO external libraries ──────────────────────────────
const $  = id => document.getElementById(id);
let D = null, selFile = null;

// ── THEME ────────────────────────────────────────────────────────────────────
function applyTheme(t, save){
  document.documentElement.setAttribute('data-theme', t);
  const icon  = $('theme-icon');
  const label = $('theme-label');
  const btn   = $('theme-btn');
  if(t === 'light'){
    if(icon)  icon.textContent  = '🌙';
    if(label) label.textContent = 'Dark';
    if(btn)   btn.title = 'Switch to Dark mode';
  } else {
    if(icon)  icon.textContent  = '☀️';
    if(label) label.textContent = 'Light';
    if(btn)   btn.title = 'Switch to Light mode';
  }
  // Swap all chart images to the right theme version
  document.querySelectorAll('img.chart-img[data-dark]').forEach(img=>{
    const dk = decodeURIComponent(img.getAttribute('data-dark')||'');
    const lk = decodeURIComponent(img.getAttribute('data-light')||dk);
    const next = (t==='light' && lk) ? lk : dk;
    if(next && img.src !== next) img.src = next;
  });
  if(save) try{ localStorage.setItem('dcss_theme', t); }catch(e){}
}
function toggleTheme(){
  const cur = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(cur === 'dark' ? 'light' : 'dark', true);
}
// Restore saved preference on load (default = dark)
(function(){
  try{
    const saved = localStorage.getItem('dcss_theme') || 'dark';
    applyTheme(saved, false);
  }catch(e){ applyTheme('dark', false); }
})();

// file input
const fi = $('fi'), drop = $('drop');
fi.onchange = () => { selFile=fi.files[0]; if(selFile){$('fname').textContent='📄 '+selFile.name+' ('+Math.round(selFile.size/1024)+' KB)'; $('abtn').disabled=false; $('uerr').innerHTML='';} };
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag');});
drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');
  if(e.dataTransfer.files.length){selFile=e.dataTransfer.files[0];$('fname').textContent='📄 '+selFile.name;$('abtn').disabled=false;}});

function doUpload(){
  if(!selFile)return;
  const fd=new FormData(); fd.append('file',selFile);
  $('spin').classList.add('show'); $('abtn').disabled=true; $('uerr').innerHTML='';
  fetch('/upload',{method:'POST',body:fd})
    .then(r=>r.text().then(t=>{try{return{ok:r.ok,data:JSON.parse(t)};}catch(e){return{ok:false,data:{error:'Server error (HTTP '+r.status+'). Check terminal for details.'}}; }}))
    .then(res=>{
      $('spin').classList.remove('show'); $('abtn').disabled=false;
      if(!res.ok||res.data.error){$('uerr').innerHTML='<div class="alert-box">⚠ '+(res.data.error||'Unknown error')+'</div>';return;}
      D=res.data; build(D);
    })
    .catch(err=>{$('spin').classList.remove('show');$('abtn').disabled=false;
      $('uerr').innerHTML='<div class="alert-box">⚠ '+err.message+'</div>';});
}

function reset(){
  D=null;
  $('dash').classList.remove('show'); $('upload-section').style.display='';
  $('abtn').disabled=true; fi.value=''; $('fname').textContent='';
  $('file-pill').textContent='No file'; $('mttr-pill').style.display='none';
  $('health-pill').style.display='none'; $('kb-tab').classList.remove('show');
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('on',i===0));
  document.querySelectorAll('.page').forEach((p,i)=>p.classList.toggle('on',i===0));
}

function go(name,el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  $('pg-'+name).classList.add('on'); el.classList.add('on');
}

function toggleAl(hdr,bodyId){hdr.classList.toggle('open');$(bodyId).classList.toggle('open');}
function toggleGrp(el){el.classList.toggle('open');const b=el.nextElementSibling;if(b)b.classList.toggle('open');}

function filterTbl(inp,tbodyId){
  const q=inp.value.toLowerCase();
  document.querySelectorAll('#'+tbodyId+' tr').forEach(tr=>tr.classList.toggle('hidden',!tr.textContent.toLowerCase().includes(q)));
}

function exportCSV(tbodyId,fname){
  const tbody=$(tbodyId); if(!tbody)return;
  const thead=tbody.closest('table').querySelector('thead');
  const cols=thead?[...thead.querySelectorAll('th')].map(th=>th.textContent.trim()):[];
  const rows=[cols.join(',')];
  tbody.querySelectorAll('tr:not(.hidden)').forEach(tr=>{
    rows.push([...tr.querySelectorAll('td')].map(td=>'"'+td.textContent.trim().replace(/"/g,'""')+'"').join(','));
  });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([rows.join('\n')],{type:'text/csv'}));
  a.download=fname+'_'+new Date().toISOString().slice(0,10)+'.csv'; a.click();
}

// Store both versions; swap on theme change
function setImg(divId, darkSrc, lightSrc){
  const d=$(divId); if(!d) return;
  const cur = document.documentElement.getAttribute('data-theme')||'dark';
  const use = (cur==='light' && lightSrc) ? lightSrc : darkSrc;
  if(!use) return;
  d.innerHTML='<img src="'+use+'" class="chart-img" data-dark="'+encodeURIComponent(darkSrc||'')+'" data-light="'+encodeURIComponent(lightSrc||use)+'"/>';
}

// ── BUILD ────────────────────────────────────────────────────────────────────
function build(d){
  $('upload-section').style.display='none'; $('dash').classList.add('show');
  $('file-pill').textContent=selFile?selFile.name:'Loaded';
  $('mttr-pill').textContent=d.mttr_source; $('mttr-pill').style.display='';
  const hcol=d.health>=70?'var(--green)':d.health>=50?'var(--yellow)':'var(--red)';
  $('health-val').textContent=d.health+'/100'; $('health-val').style.color=hcol;
  $('health-pill').style.display='';
  if(d.has_kb){$('kb-tab').classList.add('show');} else {$('kb-tab').classList.remove('show');}
  $('footer-stats').textContent=d.detected_cols.length+' columns · '+d.total+' records';

  buildKPIs(d); buildAlerts(d); buildCharts(d);
  buildOrgTable(d); buildTeamPanels(d); buildCI(d);
  if(d.has_kb) buildKB(d);
  buildAdvanced(d);
  buildInfo(d);
}

function buildKPIs(d){
  const slac=d.sla_pct>=70?'var(--green)':d.sla_pct>=50?'var(--yellow)':'var(--red)';
  const mttrc=d.mttr<=24?'var(--green)':d.mttr<=72?'var(--yellow)':'var(--red)';
  const hc=d.health>=70?'var(--green)':d.health>=50?'var(--yellow)':'var(--red)';
  const kpis=[
    {l:'Total Incidents',  v:d.total,       c:'var(--blue)',   i:'📋', s:'All records in file'},
    {l:'Closed/Resolved',  v:d.closed_ct,   c:'var(--green)',  i:'✅', s:'Resolved & Closed status'},
    {l:'SLA Compliance',   v:d.sla_pct+'%', c:slac,            i:'🎯', s:d.sla_met_ct+' met SLA'},
    {l:'MTTR (avg)',       v:d.mttr+'h',    c:mttrc,           i:'⏱', s:'Mean Time To Resolve'},
    {l:'P1 Critical',      v:d.p1_ct,       c:'var(--red)',    i:'🚨', s:'Highest priority count'},
  ];
  $('kpi-row').innerHTML=kpis.map(k=>`<div class="kpi">
    <div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-icon">${k.i}</div>
    <div class="kpi-lbl">${k.l}</div>
    <div class="kpi-val" style="color:${k.c}">${k.v}</div>
    <div class="kpi-sub">${k.s}</div></div>`).join('')+
    `<div class="date-bar">
      <span style="color:var(--muted)">📅 Date Range:</span>
      <span style="color:var(--blue);font-family:monospace">${d.date_min} → ${d.date_max}</span>
      <span style="color:var(--dim);margin-left:auto;font-family:monospace">Health: <strong style="color:${hc}">${d.health}/100</strong></span>
    </div>`;
  // overview charts row
  const topRow=$('ov-charts-top');
  topRow.innerHTML='';
  const ov=[
    {key:'pri',   dot:'var(--blue)',   title:'Priority Distribution'},
    {key:'sla_donut', dot:'var(--green)', title:'SLA Status'},
    {key:'grp',   dot:'var(--yellow)', title:'Incidents by Group', badge:d.grp_count+' groups'},
  ];
  ov.forEach(o=>{
    const div=document.createElement('div'); div.className='cc';
    div.innerHTML=`<div class="cc-hd"><span class="cc-dot" style="background:${o.dot}"></span>
      <span class="cc-title">${o.title}</span>${o.badge?`<span class="cc-badge">${o.badge}</span>`:''}</div>
      <div id="ov-${o.key}"></div>`;
    topRow.appendChild(div);
    if(d.charts[o.key]) setImg('ov-'+o.key, d.charts[o.key], (d.charts_light||{})[o.key]);
  });
}

// ── alert group panels ──────────────────────────────────────────────────────
function grpAlertHTML(grpList){
  if(!grpList||!grpList.length) return '<div class="al-none">None found ✓</div>';
  return grpList.map(g=>{
    const cols=g.rows.length?Object.keys(g.rows[0]):[];
    const eid='ag'+Math.random().toString(36).slice(2,8);
    const thead=cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('');
    const tbody=g.rows.map(r=>'<tr>'+cols.map(c=>{
      const v=r[c]!==undefined?r[c]:'—';
      if(c==='MTTR_Hours'||c==='AgeDays') return `<td style="font-family:monospace;color:var(--red);font-weight:700">${v}</td>`;
      if(c==='Group_Transfers') return `<td><span class="chip c-orange">${v}</span></td>`;
      if(c==='Priority'){const PRIMAP={'P1 - Critical':'#ff4f6a','Critical':'#ff4f6a','P2 - High':'#ffc240','High':'#ffc240','P3 - Medium':'#4a8cff','Medium':'#4a8cff','P4 - Low':'#30d988','Low':'#30d988'};const pc=PRIMAP[v]||'#7b8db0';return `<td><span class="chip" style="background:${pc}22;color:${pc}">${v}</span></td>`;}
      if(c==='SLAStatus'){const SLAM={'Met':'#30d988','Within SLA':'#30d988','SLA Met':'#30d988','OK':'#30d988','Yes':'#30d988','Compliant':'#30d988','Breached':'#ff4f6a','Missed':'#ff4f6a','Violated':'#ff4f6a','Failed':'#ff4f6a','No':'#ff4f6a','Overdue':'#ff4f6a','Pending':'#ffc240','In Progress':'#ffc240','Active':'#ffc240','Invalid':'#94a3b8','Exempt':'#94a3b8','N/A':'#94a3b8','Cancelled':'#94a3b8'};const sc=SLAM[v]||'#fb923c';return `<td><span class="chip" style="background:${sc}22;color:${sc}">${v}</span></td>`;}
      return `<td>${v}</td>`;
    }).join('')+'</tr>').join('');
    const statKey=g.avg_mttr!==undefined?`Avg MTTR <strong>${g.avg_mttr}h</strong>`:`Avg Hops <strong>${g.avg_hops||'—'}</strong>`;
    return `<div class="grp-panel">
      <div class="grp-hdr" onclick="toggleGrp(this)">
        <span>👥</span><span class="gname">${g.group}</span>
        <div class="grp-stats"><span class="gs">Count <strong>${g.count}</strong></span><span class="gs">${statKey}</span></div>
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
  $('mttr-body').innerHTML=grpAlertHTML(d.mttr_grp_alerts);

  $('hop-badge').textContent=d.hop_alert_total;
  $('hop-body').innerHTML=grpAlertHTML(d.hop_grp_alerts);

  $('age-badge').textContent=d.age_alert_total;
  if(d.age_alert_rows&&d.age_alert_rows.length){
    const cols=Object.keys(d.age_alert_rows[0]);
    const eid='age'+Math.random().toString(36).slice(2,8);
    const trs=d.age_alert_rows.map(r=>'<tr>'+cols.map(c=>{
      const v=r[c]!==undefined?r[c]:'—';
      if(c==='AgeDays') return `<td style="font-family:monospace;color:var(--yellow);font-weight:700">${v}</td>`;
      return `<td>${v}</td>`;
    }).join('')+'</tr>').join('');
    $('age-body').innerHTML=`<div class="tbl-toolbar">
      <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'${eid}')"/>
      <button class="export-btn" onclick="exportCSV('${eid}','incident_aging')">⬇ CSV</button>
    </div>
    <div class="tbl-wrap"><table><thead><tr>${cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')}</tr></thead>
    <tbody id="${eid}">${trs}</tbody></table></div>`;
  }
}

function buildCharts(d){
  // trends
  setImg('c-vol', d.charts.vol, (d.charts_light||{}).vol);
  setImg('c-dow', d.charts.dow, (d.charts_light||{}).dow);
  setImg('c-xfer', d.charts.xfer, (d.charts_light||{}).xfer);
  setImg('c-sla-trend', d.charts.sla_trend, (d.charts_light||{}).sla_trend);
  if(d.charts.heatmap){$('heatmap-card').style.display='';setImg('c-heatmap', d.charts.heatmap, (d.charts_light||{}).heatmap);}
  // sla
  setImg('c-sla-pri', d.charts.sla_pri, (d.charts_light||{}).sla_pri);
  setImg('c-mttr-grp', d.charts.mttr_grp, (d.charts_light||{}).mttr_grp);
  // ci
  setImg('c-ci', d.charts.ci, (d.charts_light||{}).ci);
  setImg('c-svc', d.charts.svc, (d.charts_light||{}).svc);
  // service type charts (SLA tab)
  setImg('c-svc-donut', (d.charts||{}).svc_donut, (d.charts_light||{}).svc_donut);
  setImg('c-svc-sla',   (d.charts||{}).svc_sla,   (d.charts_light||{}).svc_sla);
  setImg('c-svc-mttr',  (d.charts||{}).svc_mttr,  (d.charts_light||{}).svc_mttr);
  setImg('c-svc-trend', (d.charts||{}).svc_trend, (d.charts_light||{}).svc_trend);
  // SLA & MTTR summary table by Service Type
  (function(){
    const tbody=$('svc-tbl-tbody');
    if(!tbody) return;
    const slaLabels=d.svc_sla_labels||[];
    const slaVals=d.svc_sla_vals||[];
    const volMap={};(d.svc_labels||[]).forEach((l,i)=>volMap[l]=d.svc_data[i]||0);
    const mttrMap={};(d.svc_mttr_labels||[]).forEach((l,i)=>mttrMap[l]=d.svc_mttr_vals[i]||'—');
    tbody.innerHTML=slaLabels.map(l=>{
      const sla=slaVals[slaLabels.indexOf(l)]||0;
      const cls=sla>=70?'c-met':sla>=50?'c-pend':'c-breach';
      const bc=sla>=70?'#30d988':sla>=50?'#ffc240':'#ff4f6a';
      return `<tr>
        <td style="font-weight:700">${l}</td>
        <td style="font-family:monospace">${volMap[l]||0}</td>
        <td><span class="chip ${cls}">${sla}%</span></td>
        <td style="font-family:monospace;color:var(--yellow)">${mttrMap[l]||'—'}</td>
        <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill"
          style="width:${Math.min(sla,100)}%;background:${bc}"></div></div></div></td>
      </tr>`;
    }).join('') || '<tr><td colspan="5" style="color:var(--dim);padding:12px">SLAStatus or Service_Type column not found</td></tr>';
  })();
}

function buildOrgTable(d){
  $('org-tbody').innerHTML=(d.org_rows||[]).map(r=>{
    const sla=parseFloat(r.sla_pct)||0,cls=sla>=70?'c-met':sla>=50?'c-pend':'c-breach';
    const bc=sla>=70?'#30d988':sla>=50?'#ffc240':'#ff4f6a';
    return `<tr><td style="font-weight:700">${r.Assigned_Support_Organisation||'—'}</td>
      <td style="font-family:monospace;font-weight:700">${r.total}</td>
      <td><span class="chip c-breach">${r.open||0}</span></td>
      <td><span class="chip ${cls}">${sla}%</span></td>
      <td style="font-family:monospace">${r.mttr||'—'}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${Math.min(sla,100)}%;background:${bc}"></div></div></div></td></tr>`;
  }).join('');
}

function buildTeamPanels(d){
  const c=$('team-panels');
  if(!d.group_tables||!d.group_tables.length){c.innerHTML='<div class="cc"><div style="color:var(--muted);font-size:.83rem">No AssignedGroup/Assignee data found.</div></div>';return;}
  c.innerHTML=d.group_tables.map(g=>{
    const sc=typeof g.sla_pct==='number'?(g.sla_pct>=70?'c-met':g.sla_pct>=50?'c-pend':'c-breach'):'';
    const sd=typeof g.sla_pct==='number'?g.sla_pct+'%':g.sla_pct;
    const maxT=g.assignees.length?Math.max(...g.assignees.map(a=>a.Total)):1;
    const aCols=g.assignees.length?Object.keys(g.assignees[0]):[];
    const eid='tm'+Math.random().toString(36).slice(2,8);
    const trs=g.assignees.map(a=>{
      const asla=parseFloat(a.SLA_Pct||0),ac=asla>=70?'c-met':asla>=50?'c-pend':'c-breach';
      const pct=Math.round(a.Total/maxT*100);
      return '<tr>'+aCols.map(k=>{const v=a[k];
        if(k==='SLA_Pct')return `<td><span class="chip ${ac}">${v}%</span></td>`;
        if(k==='Assignee')return `<td style="font-weight:600">${v}</td>`;
        if(k==='Total')return `<td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--blue)"></div></div><span style="font-family:monospace;font-size:.72rem;color:var(--muted)">${v}</span></div></td>`;
        return `<td style="font-family:monospace;font-size:.74rem">${v}</td>`;
      }).join('')+'</tr>';
    }).join('');
    return `<div class="grp-panel">
      <div class="grp-hdr" onclick="toggleGrp(this)">
        <span>👥</span><span class="gname">${g.group}</span>
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

function buildCI(d){
  // ── Grouped by AssignedGroup panels ──────────────────────────────────────────
  const ciGrp = d.ci_by_group||[];
  const ciGrpBadge = $('ci-grp-badge');
  const ciGrpContainer = $('ci-grp-panels');
  if(ciGrpBadge) ciGrpBadge.textContent = ciGrp.length+' groups';
  if(ciGrpContainer){
    if(!ciGrp.length){
      ciGrpContainer.innerHTML='<div class="al-none">No HPD_CI with ≥ 3 incidents found ✓</div>';
    } else {
      ciGrpContainer.innerHTML = ciGrp.map((g,gi)=>{
        const eid='cig'+gi+'_'+Math.random().toString(36).slice(2,6);
        const maxC=g.rows.length?Math.max(...g.rows.map(r=>r.count)):1;
        const trs=g.rows.map((r,i)=>{
          const pct=Math.round(r.count/maxC*100);
          const lastDate=r.Last_Incident_Date||'—';
          return `<tr>
            <td style="color:var(--dim);font-family:monospace;font-size:.7rem">${i+1}</td>
            <td style="font-weight:700;color:var(--cyan);font-family:monospace;font-size:.8rem">${r.HPD_CI}</td>
            <td><span class="chip c-orange" style="font-family:monospace">${r.count}</span></td>
            <td style="font-family:monospace;font-size:.76rem;color:${lastDate!=='—'?'var(--muted)':'var(--dim)'}">${lastDate}</td>
            <td style="min-width:100px"><div class="mbar"><div class="mbar-bg"><div class="mbar-fill"
              style="width:${pct}%;background:var(--cyan)"></div></div></div></td>
          </tr>`;
        }).join('');
        return `<div class="grp-panel">
          <div class="grp-hdr" onclick="toggleGrp(this)">
            <span style="font-size:.9rem">🖥</span>
            <span class="gname">${g.group}</span>
            <div class="grp-stats">
              <span class="gs">CIs with ≥3 <strong>${g.total_ci}</strong></span>
              <span class="gs">Total Incidents <strong>${g.total_inc}</strong></span>
            </div>
            <span class="g-expand">▼</span>
          </div>
          <div class="grp-body">
            <div class="tbl-toolbar">
              <input class="search-box" placeholder="Search CI…" oninput="filterTbl(this,'${eid}')"/>
              <button class="export-btn" onclick="exportCSV('${eid}','ci_${g.group.replace(/\s/g,'_')}')">⬇ CSV</button>
            </div>
            <div class="tbl-wrap"><table>
              <thead><tr><th>#</th><th>HPD_CI</th><th>Incidents in this Group</th><th>Last Incident Date</th><th>Volume</th></tr></thead>
              <tbody id="${eid}">${trs}</tbody>
            </table></div>
          </div>
        </div>`;
      }).join('');
    }
  }

  // ── Flat table ────────────────────────────────────────────────────────────────
  $('ci3-badge').textContent=(d.ci_gt3_rows||[]).length+' CIs';
  const maxC=(d.ci_gt3_rows||[]).length?Math.max(...d.ci_gt3_rows.map(r=>r.count)):1;
  $('ci3-tbody').innerHTML=(d.ci_gt3_rows||[]).map((r,i)=>{
    const pct=Math.round(r.count/maxC*100);
    return `<tr><td style="color:var(--dim);font-family:monospace">${i+1}</td>
      <td style="font-weight:700;color:var(--cyan);font-family:monospace">${r.HPD_CI}</td>
      <td><span class="chip c-orange">${r.count}</span></td>
      <td style="font-size:.73rem;color:var(--muted)">${r.groups_str}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${pct}%;background:var(--orange)"></div></div></div></td></tr>`;
  }).join('');
}

function buildKB(d){
  const s=d.kb_summary||{};

  // ── KPIs ────────────────────────────────────────────────────────────────────
  $('kb-kpis').innerHTML=[
    {l:'RKM Solution',          v:s.total_rkm||0,     c:'var(--purple)', i:'📚', sub:'KB-tagged incidents'},
    {l:'Known Error',           v:s.total_ke||0,       c:'var(--orange)', i:'🔶', sub:'Known Error tagged'},
    {l:'Problem Investigation', v:s.total_ps||0,       c:'var(--cyan)',   i:'🔷', sub:'Problem Investigation tagged'},
    {l:'Untagged',              v:s.total_untagged||0, c:'var(--red)',    i:'🚫', sub:'No recognised tag'},
  ].map(k=>`<div class="kpi"><div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-icon">${k.i}</div><div class="kpi-lbl">${k.l}</div>
    <div class="kpi-val" style="color:${k.c}">${k.v}</div><div class="kpi-sub">${k.sub}</div></div>`).join('');

  // ── Tagging charts ───────────────────────────────────────────────────────────
  if(d.kb_charts){
    setImg('c-tag-donut',   d.kb_charts.tagging_donut,   (d.kb_charts_light||{}).tagging_donut);
    setImg('c-tag-stacked', d.kb_charts.tagging_stacked, (d.kb_charts_light||{}).tagging_stacked);
  }

  // ── Tagging by group summary table ───────────────────────────────────────────
  const tgr=d.tagging_grp_rows||[];
  $('tag-grp-badge').textContent=tgr.length+' groups';
  $('tag-grp-tbody').innerHTML=tgr.map(r=>{
    const cls=r.tagged_pct>=70?'c-met':r.tagged_pct>=40?'c-pend':'c-breach';
    return `<tr>
      <td style="font-weight:700">${r.group}</td>
      <td style="font-family:monospace;font-weight:700">${r.total}</td>
      <td style="font-family:monospace;color:var(--purple)">${r.rkm}</td>
      <td style="font-family:monospace;color:var(--orange)">${r.ke}</td>
      <td style="font-family:monospace;color:var(--cyan)">${r.ps}</td>
      <td style="font-family:monospace;color:var(--red)">${r.untagged}</td>
      <td><span class="chip ${cls}">${r.tagged_pct}%</span></td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${r.tagged_pct}%;background:var(--purple)"></div></div></div></td>
    </tr>`;
  }).join('');

  // ── SHARED HELPERS ───────────────────────────────────────────────────────────

  // Build collapsible group panels for FLAT INCIDENT lists
  // cols = array of {key, label, style}
  function renderIncidentGrpPanels(containerId, badgeId, grpList, cols, accentColor, exportPrefix) {
    const badge=$(badgeId), container=$(containerId);
    if(!grpList||!grpList.length){
      if(badge) badge.textContent='0';
      if(container) container.innerHTML='<div class="al-none">No incidents ✓</div>';
      return;
    }
    const totalAll = grpList.reduce((s,g)=>s+g.total_count,0);
    if(badge) badge.textContent=totalAll+' incidents · '+grpList.length+' groups';
    const thead='<tr>'+cols.map(c=>`<th>${c.label}</th>`).join('')+'</tr>';
    container.innerHTML=grpList.map((g,gi)=>{
      const eid='ig'+gi+'_'+Math.random().toString(36).slice(2,6);
      const trs=g.rows.map(r=>'<tr>'+cols.map(c=>{
        const v=r[c.key]!==undefined?r[c.key]:'—';
        return `<td style="${c.style||'font-size:.75rem'}">${v}</td>`;
      }).join('')+'</tr>').join('');
      return `<div class="grp-panel">
        <div class="grp-hdr" onclick="toggleGrp(this)">
          <span style="font-size:.9rem">${accentColor.includes('red')?'🚫':'❌'}</span>
          <span class="gname">${g.group}</span>
          <div class="grp-stats"><span class="gs">Incidents <strong>${g.total_count}</strong></span></div>
          <span class="g-expand">▼</span>
        </div>
        <div class="grp-body">
          <div class="tbl-toolbar">
            <input class="search-box" placeholder="Search…" oninput="filterTbl(this,'${eid}')"/>
            <button class="export-btn" onclick="exportCSV('${eid}','${exportPrefix}_${g.group.replace(/\s/g,'_')}')">⬇ CSV</button>
          </div>
          <div class="tbl-wrap"><table><thead>${thead}</thead><tbody id="${eid}">${trs}</tbody></table></div>
        </div>
      </div>`;
    }).join('');
  }

  // Build collapsible group panels for DESC COUNT tables
  // On expand: Request Description | Incident Count | Volume bar
  function renderDescGrpPanels(containerId, badgeId, grpList, accentColor, exportPrefix) {
    const badge=$(badgeId), container=$(containerId);
    if(!grpList||!grpList.length){
      if(badge) badge.textContent='0';
      if(container) container.innerHTML='<div class="al-none">No data for this tag type</div>';
      return;
    }
    const totalAll=grpList.reduce((s,g)=>s+g.total_count,0);
    if(badge) badge.textContent=totalAll+' incidents · '+grpList.length+' groups';
    container.innerHTML=grpList.map((g,gi)=>{
      const eid='dg'+gi+'_'+Math.random().toString(36).slice(2,6);
      const maxC=g.rows.length?Math.max(...g.rows.map(r=>r.Incident_Count)):1;
      const trs=g.rows.map((r,i)=>{
        const pct=Math.round((r.Incident_Count||0)/maxC*100);
        return `<tr>
          <td style="color:var(--dim);font-family:monospace;font-size:.68rem;width:32px">${i+1}</td>
          <td style="font-size:.78rem;white-space:normal;line-height:1.5">${r.Request_Description||'—'}</td>
          <td style="width:80px"><span class="chip c-blue" style="font-family:monospace">${r.Incident_Count}</span></td>
          <td style="width:110px"><div class="mbar"><div class="mbar-bg"><div class="mbar-fill"
            style="width:${pct}%;background:${accentColor}"></div></div></div></td>
        </tr>`;
      }).join('');
      return `<div class="grp-panel">
        <div class="grp-hdr" onclick="toggleGrp(this)">
          <span style="font-size:.9rem">📂</span>
          <span class="gname">${g.group}</span>
          <div class="grp-stats">
            <span class="gs">Total <strong>${g.total_count}</strong></span>
            <span class="gs">Descriptions <strong>${g.rows.length}</strong></span>
          </div>
          <span class="g-expand">▼</span>
        </div>
        <div class="grp-body">
          <div class="tbl-toolbar">
            <input class="search-box" placeholder="Search description…" oninput="filterTbl(this,'${eid}')"/>
            <button class="export-btn" onclick="exportCSV('${eid}','${exportPrefix}_${g.group.replace(/\s/g,'_')}')">⬇ CSV</button>
          </div>
          <div class="tbl-wrap"><table>
            <thead><tr><th>#</th><th>Request Description</th><th>Incident Count</th><th>Volume</th></tr></thead>
            <tbody id="${eid}">${trs}</tbody>
          </table></div>
        </div>
      </div>`;
    }).join('');
  }

  // ── Untagged incidents — grouped panels ──────────────────────────────────────
  renderIncidentGrpPanels('untag-panels','untag-badge',
    d.untagged_grp_rows||[],
    [
      {key:'Incident_Number', label:'Incident #',   style:'font-family:monospace;font-weight:700;font-size:.75rem;color:var(--cyan)'},
      {key:'Priority',        label:'Priority',     style:'font-size:.74rem'},
      {key:'Request_Type01',  label:'Request Type', style:'font-size:.73rem;color:var(--muted)'},
      {key:'Status',          label:'Status',       style:'font-size:.73rem'},
      {key:'Assignee',        label:'Assignee',     style:'font-size:.74rem'},
    ],
    'var(--red)', 'untagged');

  // ── KB group coverage summary table ─────────────────────────────────────────
  $('kb-tbody').innerHTML=(d.kb_group_rows||[]).map(r=>{
    const cls=r.kb_pct>=60?'c-met':r.kb_pct>=30?'c-pend':'c-breach';
    const tops=(r.top_kbs||[]).map(k=>`<span class="chip c-purple" style="margin:1px;font-size:.63rem">${k.KB_ID} (${k.count})</span>`).join(' ')||'—';
    return `<tr><td style="font-weight:700">${r.group}</td>
      <td style="font-family:monospace">${r.total}</td>
      <td style="font-family:monospace;color:var(--purple)">${r.with_kb}</td>
      <td style="font-family:monospace;color:var(--red)">${r.without_kb}</td>
      <td><span class="chip ${cls}">${r.kb_pct}%</span></td>
      <td style="max-width:220px">${tops}</td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${r.kb_pct}%;background:var(--purple)"></div></div></div></td></tr>`;
  }).join('');

  // ── No-KB incidents — grouped panels (cols derived dynamically from data) ────
  (function(){
    const grpList = d.nokb_grp_rows||[];
    if(!grpList.length){ 
      const b=$('nokb-badge'); const c=$('nokb-panels');
      if(b) b.textContent='0';
      if(c) c.innerHTML='<div class="al-none">No incidents without KB Article ✓</div>';
      return;
    }
    // Derive column keys from the first row of the first group
    const sampleRow = (grpList[0].rows||[])[0]||{};
    const allKeys = Object.keys(sampleRow);
    // Style rules per key
    const styleMap = {
      'Incident_Number': 'font-family:monospace;font-weight:700;font-size:.75rem;color:var(--cyan)',
      'Status':          'font-size:.74rem',
      'Assignee':        'font-size:.74rem',
      'Service_Type':    'font-size:.74rem;color:var(--muted)',
    };
    const labelMap = {
      'Incident_Number': 'Incident #',
      'Service_Type':    'Service Type',
    };
    const cols = allKeys.map(k=>({
      key:   k,
      label: labelMap[k] || k.replace(/_/g,' '),
      style: styleMap[k] || 'font-size:.74rem',
    }));
    renderIncidentGrpPanels('nokb-panels','nokb-badge', grpList, cols, 'var(--red)', 'no_kb');
  })();

  // ── Three description group panels ───────────────────────────────────────────
  renderDescGrpPanels('rkm-desc-panels','rkm-desc-badge', d.kb_article_rows||[], 'var(--purple)', 'rkm');
  renderDescGrpPanels('ke-desc-panels', 'ke-desc-badge',  d.ke_desc_rows||[],    'var(--orange)', 'ke');
  renderDescGrpPanels('pi-desc-panels', 'pi-desc-badge',  d.pi_desc_rows||[],    'var(--cyan)',   'pi');
}

function buildAdvanced(d){
  // ── First-Time-Fix Rate table ─────────────────────────────────────────────
  const ftf=d.ftf_rows||[];
  const maxFtf=ftf.length?Math.max(...ftf.map(r=>r.ftf_pct),1):100;
  $('ftf-tbody').innerHTML=ftf.length?ftf.map(r=>{
    const cls=r.ftf_pct>=70?'c-met':r.ftf_pct>=50?'c-pend':'c-breach';
    return `<tr>
      <td style="font-weight:700">${r.group}</td>
      <td style="font-family:monospace">${r.total}</td>
      <td style="font-family:monospace;color:var(--green)">${r.ftf}</td>
      <td><span class="chip ${cls}">${r.ftf_pct}%</span></td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill"
        style="width:${r.ftf_pct}%;background:${r.ftf_pct>=70?'var(--green)':r.ftf_pct>=50?'var(--yellow)':'var(--red)'}">
        </div></div></div></td>
    </tr>`;
  }).join(''):'<tr><td colspan="5" style="color:var(--dim);padding:14px">Group_Transfers column not found</td></tr>';

  // ── P1 SLA Breached table ─────────────────────────────────────────────────
  const p1b=d.p1_breach_rows||[];
  $('p1b-badge').textContent=p1b.length;
  if(p1b.length){
    const cols=Object.keys(p1b[0]);
    $('p1b-thead').innerHTML='<tr>'+cols.map(c=>`<th>${c.replace(/_/g,' ')}</th>`).join('')+'</tr>';
    $('p1b-tbody').innerHTML=p1b.map(r=>'<tr>'+cols.map(c=>{
      const v=r[c]!==undefined?r[c]:'—';
      if(c==='SLAStatus') return `<td><span class="chip c-breach">${v}</span></td>`;
      if(c==='Priority')  return `<td><span class="chip" style="background:rgba(255,79,106,.15);color:var(--red)">${v}</span></td>`;
      if(c==='MTTR_Hours') return `<td style="font-family:monospace;color:var(--yellow)">${v}h</td>`;
      return `<td style="font-size:.76rem">${v}</td>`;
    }).join('')+'</tr>').join('');
  } else {
    $('p1b-thead').innerHTML='';
    $('p1b-tbody').innerHTML='<tr><td colspan="7" style="color:var(--green);padding:14px">No P1 SLA breaches ✓</td></tr>';
  }

  // ── Repeat Offender CI table ──────────────────────────────────────────────
  const rep=d.repeat_ci_rows||[];
  const maxRep=rep.length?Math.max(...rep.map(r=>r.count)):1;
  $('rep-ci-tbody').innerHTML=rep.length?rep.map((r,i)=>{
    const pct=Math.round(r.count/maxRep*100);
    return `<tr>
      <td style="color:var(--dim);font-family:monospace">${i+1}</td>
      <td style="font-weight:700;color:var(--orange);font-family:monospace">${r.HPD_CI}</td>
      <td><span class="chip c-orange" style="font-family:monospace">${r.count}</span></td>
      <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill"
        style="width:${pct}%;background:var(--orange)"></div></div></div></td>
    </tr>`;
  }).join(''):'<tr><td colspan="4" style="color:var(--green);padding:14px">No HPD_CI with 5+ incidents ✓</td></tr>';

  // ── HOP Count Distribution ──────────────────────────────────────────────────
  (function(){
    const rows = d.hop_dist_rows||[];
    const tbody = $('hop-dist-tbody');
    const na    = $('hop-dist-na');
    if(!rows.length){
      if(tbody) tbody.innerHTML='';
      if(na) na.style.display='';
      return;
    }
    if(na) na.style.display='none';
    // Max count (excluding total row) for bar width
    const dataRows = rows.filter(r=>r.HOPs!=='Total');
    const maxC = dataRows.length ? Math.max(...dataRows.map(r=>r.Incident_Count)) : 1;
    tbody.innerHTML = rows.map(r=>{
      const isTotal = r.HOPs === 'Total';
      const pct     = isTotal ? 100 : Math.round(r.Incident_Count/maxC*100);
      const rowStyle = isTotal
        ? 'background:rgba(167,139,250,.08);font-weight:700;border-top:1px solid var(--border2)'
        : '';
      const barColor = isTotal ? 'var(--purple)' : 'rgba(167,139,250,.7)';
      return `<tr style="${rowStyle}">
        <td style="font-weight:${isTotal?'700':'500'}">${r.HOPs}</td>
        <td style="text-align:right;font-family:monospace;font-weight:${isTotal?'700':'400'};color:${isTotal?'var(--text)':'var(--muted)'}">${r.Incident_Count}</td>
        <td style="text-align:right">
          ${isTotal
            ? `<span style="font-family:monospace;font-weight:700">100.0%</span>`
            : `<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
                <span style="font-family:monospace;font-size:.76rem;color:var(--muted);min-width:44px;text-align:right">${r.Pct}</span>
                <div class="mbar-bg" style="width:60px;flex-shrink:0"><div class="mbar-fill" style="width:${pct}%;background:${barColor}"></div></div>
              </div>`
          }
        </td>
      </tr>`;
    }).join('');
  })();

  // ── MTTR Distribution ───────────────────────────────────────────────────────
  (function(){
    const rows  = d.mttr_dist_rows||[];
    const tbody = $('mttr-dist-tbody');
    const na    = $('mttr-dist-na');
    if(!rows.length){
      if(tbody) tbody.innerHTML='';
      if(na) na.style.display='';
      return;
    }
    if(na) na.style.display='none';
    const dataRows = rows.filter(r=>r.Resolution_Time_Band!=='Total');
    const maxC = dataRows.length ? Math.max(...dataRows.map(r=>r.Incident_Count)) : 1;
    // Colour bands: quick → green, medium → yellow, slow → orange, very slow → red
    function bandColor(label){
      if(label.includes('0 – 1')||label.includes('1 – 2')||label.includes('2 – 4 hrs')) return '#30d988';
      if(label.includes('4 – 8 hrs')||label.includes('8 – 16 hrs')) return '#ffc240';
      if(label.includes('16 – 24')||label.includes('24 – 48')) return '#fb923c';
      return '#ff4f6a';
    }
    tbody.innerHTML = rows.map(r=>{
      const isTotal = r.Resolution_Time_Band === 'Total';
      const pct     = isTotal ? 100 : Math.round(r.Incident_Count/maxC*100);
      const col     = isTotal ? 'var(--text)' : bandColor(r.Resolution_Time_Band);
      const rowStyle = isTotal
        ? 'background:rgba(255,194,64,.08);font-weight:700;border-top:1px solid var(--border2)'
        : '';
      return `<tr style="${rowStyle}">
        <td style="font-weight:${isTotal?'700':'500'};color:${isTotal?'var(--text)':col}">${r.Resolution_Time_Band}</td>
        <td style="text-align:right;font-family:monospace;font-weight:${isTotal?'700':'400'};color:${isTotal?'var(--text)':'var(--muted)'}">${r.Incident_Count}</td>
        <td style="text-align:right">
          ${isTotal
            ? `<span style="font-family:monospace;font-weight:700">100.0%</span>`
            : `<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
                <span style="font-family:monospace;font-size:.76rem;color:var(--muted);min-width:44px;text-align:right">${r.Pct}</span>
                <div class="mbar-bg" style="width:60px;flex-shrink:0"><div class="mbar-fill" style="width:${pct}%;background:${col}"></div></div>
              </div>`
          }
        </td>
      </tr>`;
    }).join('');
  })();
}

function buildInfo(d){
  $('col-chips').innerHTML=(d.detected_cols||[]).map(c=>`<span style="background:var(--surf);border:1px solid var(--border);
    padding:3px 10px;border-radius:6px;font-size:.67rem;font-family:monospace;color:var(--muted)">${c}</span>`).join('');
  $('mttr-logic').innerHTML=`<strong style="color:var(--text)">Formula in use:</strong><br>
    <span style="color:var(--blue);font-family:monospace">${d.mttr_source}</span><br><br>
    Priority: SubmitDate (if present) → ReportedDate → N/A<br>
    Negatives excluded. Unit: <strong style="color:var(--yellow)">Hours</strong>`;
  const ff=d.feature_flags||{};
  $('feat-flags').innerHTML=Object.entries(ff).map(([k,v])=>
    `<div><span style="color:${v?'var(--green)':'var(--dim)'}">${v?'✓':'✗'}</span> ${k}</div>`).join('');
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import socket
    # Detect LAN IP without any outbound connection —
    # SOCK_DGRAM connect() to an RFC-5737 documentation address never sends packets;
    # we use a private-range address so no frame ever leaves the machine.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.0.1", 1))   # private range — never routed externally
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            lan_ip = "YOUR_SERVER_IP"
    port=5050
    print("\n"+"="*64)
    print("  DCSS Incident Analyzer  v5  — Designed by aawasthi")
    print("="*64)
    print(f"  Local   :  http://localhost:{port}")
    print(f"  Remote  :  http://{lan_ip}:{port}   ← open from Windows browser")
    print("="*64)
    print("  🔌 100% OFFLINE — No internet required")
    print("  Charts generated by Python matplotlib (server-side)")
    print("  No Chart.js CDN · No Google Fonts CDN")
    print("  Formats : .xlsx  .xls  .csv")
    print("  Press Ctrl+C to stop")
    print("="*64+"\n")
    app.run(debug=False,port=port,host="0.0.0.0",threaded=True)
