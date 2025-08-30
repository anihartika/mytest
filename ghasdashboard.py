#!/usr/bin/env python3
"""
ghas_dashboard.py

Create a LOCAL HTML dashboard from a GitHub Security CSV export,
and export aggregated metrics to JSON and Dynatrace line protocol.

Usage:
  python ghas_dashboard.py --csv ghas_export.csv
  python ghas_dashboard.py --csv ghas_export.csv --freq W

Outputs:
  dashboard.html
  metrics_summary.json
  metrics_trend.csv
  dynatrace_metrics.txt
  metrics_by_repo.csv   (if columns present)

Notes:
- Uses pandas/numpy/matplotlib only (no Grafana, no internet).
- If your CSV headers differ, adjust COLUMN_MAP below.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- Column mapping --------------------------------------
COLUMN_MAP = {
    "repository": ["Repository", "repo", "repository_name"],
    "tool": ["Tool", "Alert Type", "Scanner", "AlertType"],
    "severity": ["Severity", "severity"],
    "state": ["State", "Alert State", "Status", "state"],
    "created_at": ["Created At", "CreatedAt", "Created", "created_at"],
    "closed_at": ["Closed At", "ClosedAt", "Resolved At", "closed_at"],
    "contributor": ["Contributor", "Actor", "User", "Opened By"],
    "team": ["Team", "Team Name", "owner_team"],
    # Optional coverage flags, if present
    "code_scanning_enabled": ["Code Scanning Enabled", "code_scanning_enabled"],
    "dependabot_enabled": ["Dependabot Alerts Enabled", "dependabot_enabled"],
    "secret_scanning_enabled": ["Secret Scanning Enabled", "secret_scanning_enabled"],
}

def find_first(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    alias = {}
    for canon, cands in COLUMN_MAP.items():
        col = find_first(df, cands)
        if col:
            alias[col] = canon
    df = df.rename(columns=alias)
    return df

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["created_at", "closed_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df

# ----------------------- Metric functions ------------------------------------
def metric_open_alerts_by_tool(df: pd.DataFrame) -> Dict[str, int]:
    if "state" not in df.columns or "tool" not in df.columns:
        return {}
    mask = df["state"].astype(str).str.lower().eq("open")
    series = df.loc[mask].groupby("tool").size().sort_values(ascending=False)
    return series.to_dict()

def metric_trend_new_alerts(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if "created_at" not in df.columns:
        return pd.DataFrame(columns=["date", "count"])
    tmp = (
        df.set_index("created_at")
          .sort_index()
          .assign(count=1)
          .resample(freq)["count"].sum().fillna(0).astype(int).reset_index()
    )
    tmp.rename(columns={"created_at": "date"}, inplace=True)
    return tmp

def metric_severity_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    if "severity" not in df.columns:
        return {}
    return df.groupby("severity").size().sort_values(ascending=False).to_dict()

def metric_by_repo(df: pd.DataFrame) -> pd.DataFrame:
    if "repository" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["repository", "tool", "severity", "state"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    pivot = (
        df[cols]
          .assign(count=1)
          .pivot_table(
              index="repository",
              columns=[c for c in ["tool", "severity", "state"] if c in cols and c != "repository"],
              values="count", aggfunc="sum", fill_value=0
          )
          .sort_index()
          .reset_index()
    )
    return pivot

def metric_mttr_days(df: pd.DataFrame) -> float:
    if not {"created_at", "closed_at"}.issubset(df.columns):
        return float("nan")
    closed = df.dropna(subset=["created_at", "closed_at"])
    if closed.empty:
        return float("nan")
    delta = (closed["closed_at"] - closed["created_at"]).dt.total_seconds() / 86400.0
    return float(delta.mean())

def metric_coverage(df: pd.DataFrame) -> Dict[str, float]:
    if "repository" not in df.columns:
        return {}
    total_repos = df["repository"].nunique()
    cov = {}
    if total_repos == 0:
        return cov
    if "tool" in df.columns:
        m = df["tool"].astype(str).str.lower()
        for tool in ["code_scanning", "dependabot", "secret_scanning"]:
            mask = m.str.contains(tool.replace("_", " "), na=False) | m.str.contains(tool, na=False)
            repos_with = df.loc[mask, "repository"].nunique()
            cov[tool] = round(100.0 * repos_with / total_repos, 2)
    for canon in ["code_scanning_enabled", "dependabot_enabled", "secret_scanning_enabled"]:
        if canon in df.columns:
            enabled = df.loc[df[canon].astype(str).str.lower().isin(["true", "1", "yes"]), "repository"].nunique()
            cov[canon] = round(100.0 * enabled / total_repos, 2)
    cov["total_repos_in_csv"] = total_repos
    return cov

def metric_unique_contributors(df: pd.DataFrame) -> int:
    if "contributor" not in df.columns:
        return 0
    return df["contributor"].dropna().nunique()

# ----------------------- Dynatrace line protocol -----------------------------
def to_dynatrace_lines(summary: Dict[str, Any], trend_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for tool, val in summary.get("open_alerts_by_tool", {}).items():
        lines.append(f'ghas.alerts.open,tool="{str(tool).lower().replace(" ", "_")}" {int(val)}')
    for sev, val in summary.get("severity_breakdown", {}).items():
        lines.append(f'ghas.alerts.severity,level="{str(sev).lower()}" {int(val)}')
    cov = summary.get("coverage_pct", {})
    for k, v in cov.items():
        if k == "total_repos_in_csv":
            continue
        lines.append(f'ghas.coverage,feature="{k}" {float(v)}')
    mttr = summary.get("mttr_days")
    if mttr is not None and not (isinstance(mttr, float) and np.isnan(mttr)):
        lines.append(f"ghas.mttr.days {float(mttr)}")
    uc = summary.get("unique_contributors")
    if isinstance(uc, (int, float)):
        lines.append(f"ghas.contributors.unique {int(uc)}")
    if trend_df is not None and not trend_df.empty:
        for _, row in trend_df.iterrows():
            day = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            lines.append(f'ghas.alerts.new,day="{day}" {int(row["count"])}')
    return lines

# ----------------------- HTML dashboard generation ---------------------------
def plot_and_save_trend(trend: pd.DataFrame, out_png: Path) -> None:
    if trend.empty:
        return
    plt.figure()
    plt.plot(trend["date"], trend["count"])
    plt.title("New GHAS Alerts Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_and_save_severity(sev_dict: Dict[str, int], out_png: Path) -> None:
    if not sev_dict:
        return
    labels = list(sev_dict.keys())
    values = list(sev_dict.values())
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values)
    plt.title("Alerts by Severity")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.xticks(x, labels, rotation=15)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_and_save_open_by_tool(open_dict: Dict[str, int], out_png: Path) -> None:
    if not open_dict:
        return
    labels = list(open_dict.keys())
    values = list(open_dict.values())
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, values)
    plt.title("Open Alerts by Tool")
    plt.xlabel("Tool")
    plt.ylabel("Open Alerts")
    plt.xticks(x, labels, rotation=15)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def write_html_dashboard(summary: Dict[str, Any],
                         trend_csv_path: Path,
                         repo_csv_path: Path | None,
                         images: Dict[str, Path],
                         out_html: Path) -> None:
    kpis = []
    open_total = sum(summary.get("open_alerts_by_tool", {}).values())
    kpis.append(("Open Alerts (total)", str(open_total)))
    mttr = summary.get("mttr_days")
    kpis.append(("MTTR (days)", "-" if (mttr is None or (isinstance(mttr, float) and np.isnan(mttr))) else f"{mttr:.2f}"))
    kpis.append(("Unique Contributors", str(summary.get("unique_contributors", 0))))

    cov_rows = []
    for k, v in summary.get("coverage_pct", {}).items():
        if k != "total_repos_in_csv":
            cov_rows.append((k, f"{v:.2f}%"))

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>GHAS Dashboard</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;}",
        "h1{margin-top:0;} .kpi{display:inline-block;margin-right:24px;padding:10px 14px;border:1px solid #ddd;border-radius:10px;}",
        "table{border-collapse:collapse;margin-top:12px;} th,td{border:1px solid #ddd;padding:8px;} th{background:#f5f5f5;}",
        ".section{margin-top:28px;} img{max-width:100%;height:auto;border:1px solid #eee;border-radius:8px;padding:6px;}",
        "</style></head><body>",
        "<h1>GHAS Metrics Dashboard</h1>",
        f"<div>Generated at: {summary.get('generated_at_utc','')}</div>",
        "<div class='section'><h2>Key KPIs</h2>",
    ]

    html.append("<div>")
    for name, val in kpis:
        html.append(f"<div class='kpi'><div><b>{name}</b></div><div>{val}</div></div>")
    html.append("</div>")

    html.append("<div class='section'><h2>Open Alerts by Tool</h2>")
    if images.get("open_by_tool"):
        html.append(f"<img src='{images['open_by_tool'].name}' alt='Open by tool'/>")
    else:
        html.append("<p>No data available.</p>")
    html.append("</div>")

    html.append("<div class='section'><h2>Severity Breakdown</h2>")
    if images.get("severity"):
        html.append(f"<img src='{images['severity'].name}' alt='Severity breakdown'/>")
    else:
        html.append("<p>No data available.</p>")
    html.append("</div>")

    html.append("<div class='section'><h2>New Alerts Trend</h2>")
    if images.get("trend"):
        html.append(f"<img src='{images['trend'].name}' alt='New alerts trend'/>")
        html.append(f"<p><a href='{trend_csv_path.name}'>Download trend CSV</a></p>")
    else:
        html.append("<p>No data available.</p>")
    html.append("</div>")

    if repo_csv_path and repo_csv_path.exists():
        html.append("<div class='section'><h2>Alerts by Repository</h2>")
        html.append(f"<p><a href='{repo_csv_path.name}'>Download repo breakdown (CSV)</a></p>")
        html.append("</div>")

    if cov_rows:
        html.append("<div class='section'><h2>Coverage</h2><table><tr><th>Feature</th><th>Coverage</th></tr>")
        for feat, pct in cov_rows:
            html.append(f"<tr><td>{feat}</td><td>{pct}</td></tr>")
        html.append("</table></div>")

    html.append("</body></html>")
    out_html.write_text("\n".join(html), encoding="utf-8")

# ----------------------- Main -------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Local GHAS dashboard + JSON export + Dynatrace lines")
    p.add_argument("--csv", required=True, help="Path to GitHub Security CSV export")
    p.add_argument("--freq", default="D", help="Trend frequency: D, W, or M. Default D.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = normalize_schema(df)
    df = parse_dates(df)

    # Compute metrics
    open_by_tool = metric_open_alerts_by_tool(df)
    trend = metric_trend_new_alerts(df, freq=args.freq)
    sev = metric_severity_breakdown(df)
    by_repo = metric_by_repo(df)
    mttr = metric_mttr_days(df)
    coverage = metric_coverage(df)
    contributors = metric_unique_contributors(df)

    summary = {
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "open_alerts_by_tool": open_by_tool,
        "severity_breakdown": sev,
        "mttr_days": None if (isinstance(mttr, float) and np.isnan(mttr)) else round(float(mttr), 2),
        "coverage_pct": coverage,
        "unique_contributors": int(contributors),
    }

    # Save tabular artifacts
    trend_csv = Path("metrics_trend.csv")
    trend.to_csv(trend_csv, index=False)
    repo_csv = None
    if not by_repo.empty:
        repo_csv = Path("metrics_by_repo.csv")
        by_repo.to_csv(repo_csv, index=False)

    # Save JSON
    summary_json = Path("metrics_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Dynatrace lines
    dt_lines = to_dynatrace_lines(summary, trend)
    Path("dynatrace_metrics.txt").write_text("\n".join(dt_lines) + ("\n" if dt_lines else ""), encoding="utf-8")

    # Charts (saved as PNGs)
    trend_png = Path("chart_trend.png")
    plot_and_save_trend(trend, trend_png)

    sev_png = Path("chart_severity.png")
    plot_and_save_severity(sev, sev_png)

    open_tool_png = Path("chart_open_by_tool.png")
    plot_and_save_open_by_tool(open_by_tool, open_tool_png)

    # HTML dashboard
    write_html_dashboard(
        summary=summary,
        trend_csv_path=trend_csv,
        repo_csv_path=repo_csv,
        images={"trend": trend_png, "severity": sev_png, "open_by_tool": open_tool_png},
        out_html=Path("dashboard.html"),
    )

    print("✅ Generated: dashboard.html, metrics_summary.json, metrics_trend.csv, dynatrace_metrics.txt")
    if repo_csv:
        print("✅ Generated: metrics_by_repo.csv")

if __name__ == "__main__":
    main()
