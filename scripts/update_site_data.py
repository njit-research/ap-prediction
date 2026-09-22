"""Copy the latest realtime forecast JSON into the site data directory.

Runs after `vendor/realtime-regression-sw/scripts/run_realtime.py` inside the
GitHub Actions workflow. On success, the newest JSON under
`vendor/realtime-regression-sw/results/predictions/YYYYMMDD/` is copied to
`site/data/latest.json` and `site/data/status.json` is refreshed with
`status="ok"`. On failure (non-zero inference exit code), `latest.json` is
preserved as-is and `status.json` records the failure so the page can surface
a warning banner.

Every run (success or failure) is also recorded into the per-anchor archives
`forecast_history.json` / `.csv` with a `status` field
(`ok` / `imputed` / `failed`). Because each 30-min anchor is attempted several
times, a "don't-downgrade" rule applies: a retry only overwrites the stored
record for an anchor when its status is the same or better, so a later transient
failure never clobbers an earlier good forecast.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "vendor" / "realtime-regression-sw" / "results"
EVENTS_DIR = REPO_ROOT / "vendor" / "realtime-regression-sw" / "dataset" / "events"
SITE_DATA_DIR = REPO_ROOT / "site" / "data"
HISTORY_STEPS = 96  # 48 hours at 30-min cadence, matches the input window

# Past-forecast archives written alongside latest.json. Currently not plotted on
# the page (data kept for future re-exposure) but always maintained.
FORECAST_HISTORY_JSON = SITE_DATA_DIR / "forecast_history.json"
FORECAST_HISTORY_CSV = SITE_DATA_DIR / "forecast_history.csv"
PLOT_HISTORY_HOURS = 48     # rolling window kept in the plot archive (JSON)
CSV_HISTORY_DAYS = 90       # rolling window kept in the 90-day archive (CSV)
STEP_MINUTES = 30           # forecast cadence
# ap30 is a discrete index (scale min gap = 1); one decimal place recovers the
# nearest level and stays well below model error, while keeping the CSV compact.
AP_DECIMALS = 1

# Per-anchor forecast status recorded in the archives.
IMPUTED_THRESHOLD = 0.05    # filled_fraction above this marks the run "imputed"
STATUS_RANK = {"failed": 0, "imputed": 1, "ok": 2}
DEFAULT_HORIZONS = 12       # forecast length used for a "failed" placeholder row (6 h)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_status() -> dict:
    status_path = SITE_DATA_DIR / "status.json"
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {
        "status": "unknown",
        "last_success_utc": None,
        "last_attempt_utc": None,
        "last_error": None,
    }


def _save_status(status: dict) -> None:
    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (SITE_DATA_DIR / "status.json").open("w", encoding="utf-8") as fp:
        json.dump(status, fp, indent=2, ensure_ascii=False)


def _find_latest_prediction() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(RESULTS_DIR.rglob("*.json"))
    return candidates[-1] if candidates else None


def _locate_event_csv(data: dict) -> Path | None:
    """Find the event CSV referenced by the forecast JSON.

    Prefers the absolute path recorded in `input.event_csv`, falls back to
    `dataset/events/{anchor_stem}.csv` under the vendored engine dir
    (vendor/realtime-regression-sw/).
    """
    recorded = data.get("input", {}).get("event_csv")
    if recorded:
        p = Path(recorded)
        if p.exists():
            return p
    anchor = data.get("anchor_timestamp_utc", "")
    if anchor:
        stem = anchor.replace("-", "").replace(":", "").replace("T", "").rstrip("Z")[:14]
        fallback = EVENTS_DIR / f"{stem}.csv"
        if fallback.exists():
            return fallback
    return None


def _load_history(event_csv: Path, steps: int = HISTORY_STEPS) -> list[dict]:
    """Return the trailing `steps` rows of the event CSV as (timestamp, ap30)."""
    import pandas as pd  # deferred import — only needed on success

    df = pd.read_csv(event_csv, parse_dates=["datetime"])
    tail = df.tail(steps)
    entries: list[dict] = []
    for _, row in tail.iterrows():
        value = row["ap30"]
        if pd.isna(value):
            continue
        ts = row["datetime"]
        entries.append({
            "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ap30": float(value),
        })
    return entries


# run_realtime.py exit codes. 2 and 3 are both transient data gaps -- the banner treats them
# alike -- but they are recorded separately because the archive could not previously say which,
# and the unexplained short gaps in coverage need exactly that distinction.
EXIT_REASON = {
    0: "ok",
    2: "feed_unreachable",
    3: "sparse_data",
}


def _exit_reason(exit_code: int) -> str:
    """Short machine-readable cause for the archive."""
    return EXIT_REASON.get(exit_code, f"error_{exit_code}")


def _error_label(exit_code: int) -> tuple[str, str]:
    """Map realtime CLI exit code to a banner status + human message."""
    if exit_code == 0:
        return "ok", ""
    if exit_code == 2:
        return "warn", "Upstream feed unreachable — waiting for next cycle."
    if exit_code == 3:
        return "warn", "Data too sparse to build a window — waiting for next cycle."
    return "error", f"Inference exited with code {exit_code}."


def _parse_iso(value: str) -> datetime:
    """Parse a `YYYY-MM-DDTHH:MM:SSZ` timestamp into an aware UTC datetime."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _fmt_iso(dt: datetime) -> str:
    """Format an aware datetime as `YYYY-MM-DDTHH:MM:SSZ`."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value) -> float:
    """Parse a value to float, returning 0.0 on missing/invalid input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _status_rank(status) -> int:
    """Rank a status for the don't-downgrade rule (unknown/legacy → ok)."""
    return STATUS_RANK.get(status or "", 2)


def _run_status(exit_code: int, filled_fraction) -> str:
    """Classify a run as ok / imputed / failed."""
    if exit_code != 0:
        return "failed"
    if filled_fraction is not None and float(filled_fraction) > IMPUTED_THRESHOLD:
        return "imputed"
    return "ok"


def _anchor_now() -> str:
    """Current anchor = latest 30-min boundary minus a 2-min publish offset."""
    ref = datetime.now(tz=timezone.utc) - timedelta(minutes=2)
    minute = 30 if ref.minute >= 30 else 0
    return _fmt_iso(ref.replace(minute=minute, second=0, microsecond=0))


def _current_latest_anchor() -> str | None:
    """Anchor of the currently-published latest.json, or None if absent."""
    path = SITE_DATA_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("anchor_timestamp_utc")
    except (ValueError, OSError):
        return None


def _update_forecast_history(target_iso: str, ap30, lower, upper, status: str) -> None:
    """Upsert one first-frame (+30 min) entry into forecast_history.json.

    Maintains a 48-hour rolling 30-min grid; each slot carries the first-horizon
    ap30, its MCD interval (`lower`/`upper`), and the run `status`. The current
    target is written only when its status is the same or better than any
    existing record for that target (don't-downgrade). Never-attempted slots are
    0-filled with a null status.

    Args:
        target_iso: First-horizon target time (anchor + 30 min).
        ap30: First-horizon ap30 (0 for a failed placeholder).
        lower: MCD lower bound (or None).
        upper: MCD upper bound (or None).
        status: One of "ok" / "imputed" / "failed".
    """
    known: dict[str, dict] = {}
    if FORECAST_HISTORY_JSON.exists():
        try:
            for entry in json.loads(FORECAST_HISTORY_JSON.read_text(encoding="utf-8")):
                value = float(entry.get("ap30", 0) or 0)
                if value or entry.get("status") == "failed":
                    known[entry["target_timestamp_utc"]] = {
                        "ap30": value,
                        "lower": entry.get("lower"),
                        "upper": entry.get("upper"),
                        "status": entry.get("status"),
                    }
        except (ValueError, KeyError):
            pass

    existing = known.get(target_iso)
    if existing is None or _status_rank(status) >= _status_rank(existing.get("status")):
        known[target_iso] = {"ap30": ap30, "lower": lower, "upper": upper, "status": status}

    grid_end = max(_parse_iso(k) for k in known)
    step = timedelta(minutes=STEP_MINUTES)
    cursor = grid_end - timedelta(hours=PLOT_HISTORY_HOURS)
    grid: list[dict] = []
    while cursor <= grid_end:
        iso = _fmt_iso(cursor)
        rec = known.get(iso)
        if rec:
            grid.append({"target_timestamp_utc": iso, "ap30": rec["ap30"],
                         "lower": rec["lower"], "upper": rec["upper"],
                         "status": rec.get("status")})
        else:
            grid.append({"target_timestamp_utc": iso, "ap30": 0,
                         "lower": 0, "upper": 0, "status": None})
        cursor += step

    FORECAST_HISTORY_JSON.write_text(
        json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _update_forecast_csv(anchor_iso: str, values: list[float], status: str,
                         reason: str = "") -> None:
    """Upsert one anchor row into the 90-day wide-format CSV archive.

    Columns: `anchor_timestamp_utc, status, m_30 … m_360` (ap30 per horizon lead
    time in minutes). Maintains a rolling `CSV_HISTORY_DAYS` grid. The current
    anchor row is written only when its status is the same or better than any
    existing row for that anchor (don't-downgrade). Never-produced anchors are
    0-filled with an empty status.

    Args:
        anchor_iso: Anchor timestamp.
        values: ap30 per horizon (length = horizons; all-zero for a failure).
        status: One of "ok" / "imputed" / "failed".
        reason: Machine-readable cause -- "ok", "feed_unreachable", "sparse_data",
            "no_forecast_in_payload" or "error_<code>". Rows written before this column
            existed keep an empty value.
    """
    horizons = len(values) if values else DEFAULT_HORIZONS
    lead_cols = [f"m_{h * STEP_MINUTES}" for h in range(1, horizons + 1)]
    columns = ["anchor_timestamp_utc", "status", "reason", *lead_cols]

    known: dict[str, dict] = {}
    if FORECAST_HISTORY_CSV.exists():
        try:
            with FORECAST_HISTORY_CSV.open("r", encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    vals = [_safe_float(row.get(c)) for c in lead_cols]
                    st = row.get("status") or None
                    if any(vals) or st == "failed":
                        known[row["anchor_timestamp_utc"]] = {
                            "values": vals, "status": st, "reason": row.get("reason") or ""}
        except OSError:
            pass

    new_values = [round(float(v), AP_DECIMALS) for v in values] if values else [0.0] * horizons
    existing = known.get(anchor_iso)
    if existing is None or _status_rank(status) >= _status_rank(existing.get("status")):
        known[anchor_iso] = {"values": new_values, "status": status, "reason": reason}

    grid_end = max(_parse_iso(k) for k in known)
    step = timedelta(minutes=STEP_MINUTES)
    cursor = grid_end - timedelta(days=CSV_HISTORY_DAYS)
    rows: list[dict] = []
    while cursor <= grid_end:
        anchor_key = _fmt_iso(cursor)
        rec = known.get(anchor_key)
        vals = rec["values"] if rec else [0.0] * horizons
        row = {"anchor_timestamp_utc": anchor_key,
               "status": (rec.get("status") if rec else "") or "",
               "reason": (rec.get("reason") if rec else "") or ""}
        for col, value in zip(lead_cols, vals):
            row[col] = f"{value:.{AP_DECIMALS}f}"
        rows.append(row)
        cursor += step

    with FORECAST_HISTORY_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _record_placeholder(reason: str) -> None:
    """Write a `failed` row for the current anchor with the reason recorded.

    Every path that ends without a forecast must leave a row. Over 2026-06/07, 680 of 1,384
    anchors had no row at all, which made the missing two thirds of the record impossible to
    diagnose after the fact -- a run that fails early is exactly the run worth explaining.
    """
    anchor_iso = _anchor_now()
    target_iso = _fmt_iso(_parse_iso(anchor_iso) + timedelta(minutes=STEP_MINUTES))
    _update_forecast_history(target_iso, 0, 0, 0, "failed")
    _update_forecast_csv(anchor_iso, [0.0] * DEFAULT_HORIZONS, "failed", reason)


def _record_to_archives(data: dict | None, exit_code: int) -> None:
    """Record this run into both archives with a status (don't-downgrade).

    On success the first-frame value, prediction interval, and full 12-step row are
    recorded as `ok`/`imputed`. On failure a `failed` placeholder is written for
    the current anchor (kept only if no better record exists for it).
    """
    reason = _exit_reason(exit_code)
    if exit_code == 0 and data:
        forecast = data.get("forecast") or []
        anchor_iso = data.get("anchor_timestamp_utc")
        if not forecast or not anchor_iso:
            # Success was reported but the payload is unusable. Record it rather than
            # returning silently -- an anchor with no row at all cannot be diagnosed later.
            _record_placeholder("no_forecast_in_payload")
            return
        filled = (data.get("input") or {}).get("missing_data_filled_fraction")
        status = _run_status(0, filled)
        first = forecast[0]
        target_iso = first["target_timestamp_utc"]
        ap30 = round(float(first["ap30"]), AP_DECIMALS)
        mcd = (data.get("analysis") or {}).get("mcd") or {}
        lower = round(float(mcd["lower"][0]), AP_DECIMALS) if mcd.get("lower") else None
        upper = round(float(mcd["upper"][0]), AP_DECIMALS) if mcd.get("upper") else None
        horizons = len(forecast)
        values = [0.0] * horizons
        for entry in forecast:
            h = int(entry["horizon_steps"])
            if 1 <= h <= horizons:
                values[h - 1] = round(float(entry["ap30"]), AP_DECIMALS)
    else:
        anchor_iso = _anchor_now()
        target_iso = _fmt_iso(_parse_iso(anchor_iso) + timedelta(minutes=STEP_MINUTES))
        ap30, lower, upper = 0, 0, 0
        values = [0.0] * DEFAULT_HORIZONS
        status = "failed"

    _update_forecast_history(target_iso, ap30, lower, upper, status)
    _update_forecast_csv(anchor_iso, values, status, reason)


def _record_archives_safe(data: dict | None, exit_code: int, reason: str | None = None) -> None:
    """Run _record_to_archives, never letting it break primary publishing."""
    try:
        if reason is not None:
            _record_placeholder(reason)
        else:
            _record_to_archives(data, exit_code)
    except Exception as exc:  # noqa: BLE001 - defensive, archive is optional
        print(f"WARN: forecast archive update failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, required=True,
                        help="Exit code from run_realtime.py in the workflow.")
    args = parser.parse_args()

    now_iso = _iso_now()
    status = _load_status()
    status["last_attempt_utc"] = now_iso

    label, message = _error_label(args.exit_code)

    if args.exit_code == 0:
        latest = _find_latest_prediction()
        if latest is None:
            status["status"] = "error"
            status["last_error"] = {
                "code": 0,
                "message": "Inference reported success but no JSON output was found.",
            }
            _save_status(status)
            _record_archives_safe(None, args.exit_code, reason="no_prediction_json")
            print("WARN: no prediction JSON located; status=error written.", file=sys.stderr)
            return 0

        with latest.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        event_csv = _locate_event_csv(data)
        if event_csv is not None:
            data["history"] = _load_history(event_csv)
        else:
            data["history"] = []
            print(f"WARN: event CSV not found; history omitted.", file=sys.stderr)

        SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest = SITE_DATA_DIR / "latest.json"
        with dest.open("w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
        print(f"Wrote {dest} (forecast={len(data['forecast'])}, history={len(data['history'])})")

        _record_archives_safe(data, 0)

        status["status"] = "ok"
        status["last_success_utc"] = now_iso
        status["last_error"] = None
    else:
        # Don't-downgrade the banner: if the current anchor already has a
        # successful forecast (latest.json matches), a transient retry failure
        # must not flip the banner to warn/error — the good forecast still stands
        # and main.js still applies its own staleness / imputed checks.
        if _current_latest_anchor() == _anchor_now():
            status["status"] = "ok"
            status["last_error"] = None
            print(f"Inference failed (exit={args.exit_code}) but anchor "
                  f"{_anchor_now()} already succeeded; banner kept ok.", file=sys.stderr)
        else:
            status["status"] = label
            status["last_error"] = {"code": args.exit_code, "message": message}
            print(f"Inference failed (exit={args.exit_code}); preserving previous latest.json.",
                  file=sys.stderr)
        _record_archives_safe(None, args.exit_code)

    _save_status(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
