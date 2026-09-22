# Architecture

This document explains how `ap-prediction` works end-to-end: which pieces
exist, how data flows from the live upstream feeds to the browser, and how the
public page is served at `https://sites.njit.edu/ap-prediction/`.

---

## 1. Overview

`ap-prediction` publishes a live 6-hour ap30 geomagnetic-index forecast chart
at `https://sites.njit.edu/ap-prediction/`. A GitHub Actions cron re-runs the
inference pipeline every 10 minutes (three attempts per 30-min anchor), writes a
fresh `latest.json`, and deploys the updated static site to GitHub Pages.

**Design tenets**

- **Single self-contained repository.** Everything needed to produce a forecast
  — the inference engine, the model weights, the normalizer stats, the web page
  — lives in this one repo. A plain `git clone` is enough to run it; there is no
  submodule to initialize and no release asset to download.
- **Weights committed in-tree.** The model weights (`model_best.pth`) and
  normalizer stats (`table_stats.pkl`) are committed together under
  `vendor/realtime-regression-sw/checkpoint/` as a matched pair.
- **Static front end.** Everything the browser consumes is a single JSON file
  (`site/data/latest.json`). No backend API, no database, no server-side
  rendering. Just a static site.

---

## 2. Repository layout

This is a single self-contained repository. The inference engine
(`realtime-regression-sw`) is inlined under `vendor/`, and its model checkpoint
is committed in-tree.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  github.com/njit-research/ap-prediction          (this repo)              │
│                                                                           │
│    .github/workflows/forecast.yml     ← cron + build + deploy             │
│    configs/realtime.ci.yaml           ← active profile + path overrides   │
│    scripts/update_site_data.py        ← JSON post-process                 │
│                                                                           │
│    vendor/realtime-regression-sw/     ← inference engine (inlined)        │
│      ├── scripts/run_realtime.py      ← inference CLI                     │
│      ├── src/                         ← fetch / pipeline / inference / io  │
│      └── checkpoint/                                                       │
│          ├── model_best.pth           ← trained weights  (committed)      │
│          └── table_stats.pkl          ← normalizer       (committed)      │
│                                                                           │
│    site/index.html                    ← page shell                        │
│    site/main.js                       ← Chart.js renderer                 │
│    site/data/                                                             │
│      ├── latest.json                  ← most recent forecast (auto-commit)│
│      └── status.json                  ← pipeline health      (auto-commit)│
└──────────────────────┬────────────────────────────────────────────────────┘
                       │ actions/deploy-pages@v4 (artifact)
                       ▼
          njit-research.github.io/ap-prediction/   (GitHub Pages host)
          sites.njit.edu/ap-prediction/            (NJIT domain mapping)
```

---

## 3. Data flow

Every 30 minutes, one full cycle from upstream feed to browser happens:

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ NOAA SWPC plasma     │   │ NOAA SWPC magnetic   │   │ GFZ Hp30/ap30        │
│ (1-min cadence)      │   │ (1-min cadence)      │   │ (30-min cadence)     │
└──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
           └──────────────┬──────────────┘                       │
                          ▼                                      ▼
                ┌─────────────────────────────────────────────────────┐
                │ Engine (vendor/realtime-regression-sw)              │
                │   run_realtime.py                                   │
                │                                                     │
                │  1. Fetch the three HTTP feeds (requests + retry)   │
                │  2. Aggregate 1-min → 30-min bins                   │
                │  3. Compute anchor t_end = floor(now - 2min, 30min) │
                │  4. Build the 12-row × 22-col event window          │
                │  5. Normalize with table_stats.pkl                  │
                │  6. Run model_best.pth (CPU, ~100ms)                │
                │  7. Denormalize, emit forecast 12 steps × ap30 + PI │
                │  8. Write JSON + CSV to results/{YYYYMMDD}/         │
                └──────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                ┌─────────────────────────────────────────────────────┐
                │ scripts/update_site_data.py                         │
                │                                                     │
                │  1. Locate newest JSON under vendor/.../results/    │
                │  2. Read it                                         │
                │  3. Locate the paired event CSV                     │
                │     (dataset/events/{anchor_stem}.csv)              │
                │  4. Extract recent (datetime, ap30) rows →          │
                │     embed as "history" array in payload             │
                │  5. Write to site/data/latest.json                  │
                │  6. Refresh site/data/status.json                   │
                └──────────────────────┬──────────────────────────────┘
                                       │
                                       ▼ (git commit + push to main)
                                       │
                                       ▼ (actions/deploy-pages artifact)
                                       │
                                       ▼
                ┌─────────────────────────────────────────────────────┐
                │ Browser — site/main.js                              │
                │                                                     │
                │  1. fetch latest.json + status.json (no-store)      │
                │  2. Populate metadata; paint status banner          │
                │  3. Render Chart.js: red observed history,          │
                │     blue forecast + 95% prediction interval,        │
                │     vertical "now" divider (full height)            │
                │  4. x-axis tick labels formatted in UTC             │
                └─────────────────────────────────────────────────────┘
```

### 3.1 Input

- **NOAA SWPC RTSW real-time solar wind** — plasma (density, speed, temp) and IMF
  magnetic field (Bx/By/Bz/Bt) from the SOLAR1 primary (ACE/IMAP fill gaps only
  where SOLAR1 is missing). ~24-hour rolling JSON; only the most recent portion
  needed for the input window is used.
- **GFZ Potsdam Hp30/ap30 nowcast** — 30-min geomagnetic index observed values.
  Text file, published within minutes of each 30-min boundary.

### 3.2 Anchor computation

The "anchor time" `t_end` is the most recent completed 30-min boundary, minus a
2-minute safety offset to let the publishers finish posting:

```
t_end = floor(now - 2min, to 30-min boundary)
```

Example: at 14:13 UTC → `t_end = 14:00 UTC`. At 14:45 UTC → `t_end = 14:30 UTC`.

If the final steps of the input window are NaN even after forward-fill, `t_end`
rolls back one 30-min step (up to 2 attempts). Beyond that, the CLI exits with
code 2 (`InsufficientDataError`).

### 3.3 Model I/O shape

The active profile is `in6h_out6h_gnn_transformer`: a GNN spatial encoder with a
Transformer temporal backbone, trained with a focal Huber loss and storm-rise
oversampling, a 6-hour input window, and a 6-hour forecast. The uncertainty
band is the 95 % prediction interval μ ± 1.96 σ_pred with
σ_pred² = σ_MC² + σ_residual² (Monte Carlo dropout variance plus the
checkpoint's validation residual variance, `analysis.mcd.noise_variance` in
`configs/realtime.ci.yaml`; that value belongs to the checkpoint and must be
re-measured when the checkpoint changes).

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input  | `(1, 12, 22)` | 1 batch × 12 timesteps (6 hours × 30-min) × 22 vars |
| Output | `(1, 12, 1)`  | 1 batch × 12 timesteps (6 hours × 30-min) × 1 var (ap30) |

22 input variables: 21 solar-wind parameters (v/np/t ×avg/min/max,
Bx/By/Bz/Bt ×avg/min/max) + ap30.

The input ordering and normalization schema are **safety-critical invariants**:
the 22 variables must be supplied in the exact order the model was trained on,
and the committed `table_stats.pkl` must be the same normalizer the weights were
trained with (see [docs/runtime-invariants.md](runtime-invariants.md) and
[§5.1](#51-committed-in-tree)).

---

## 4. The GitHub Actions workflow

File: [.github/workflows/forecast.yml](../.github/workflows/forecast.yml)

### 4.1 Triggers

```yaml
on:
  schedule:
    - cron: '8,18,28,38,48,58 * * * *'   # every 10 min (3 attempts per anchor)
  workflow_dispatch:            # manual trigger from the UI
    inputs:
      now: {description: 'ISO8601 anchor override', required: false}
```

- **Cron** — every 10 minutes, giving each 30-min anchor three attempts
  (`:08/:18/:28` → anchor `:00`; `:38/:48/:58` → anchor `:30`). A later attempt
  only replaces an earlier one when its status is the same or better (see §4.5),
  so a transient failure never clobbers a good forecast. GitHub may still drop
  scheduled runs under load.
- **workflow_dispatch** — manual trigger with optional `now` parameter for
  replaying a specific anchor (debugging / backfill).

### 4.2 Concurrency

```yaml
concurrency:
  group: forecast
  cancel-in-progress: false
```

If the previous run is still going, queue the next one rather than cancel it.
Prevents the pipeline from eating its own tail under heavy scheduler drift.

### 4.3 Permissions

```yaml
permissions:
  contents: write       # auto-commit site/data/*.json
  pages: write          # for actions/deploy-pages
  id-token: write       # OIDC token required by deploy-pages
```

### 4.4 Steps

Because the engine is inlined and the checkpoint is committed in-tree, there is
no submodule checkout and no release-download step.

| # | Step | Purpose |
|---|------|---------|
| 1 | `actions/checkout@v4` | Plain checkout of this repo (engine + committed checkpoint included) |
| 2 | `actions/setup-python@v5` (3.12, pip cache) | Python runtime + speed up subsequent installs |
| 3 | `pip install torch --index-url .../cpu` | **CPU-only** PyTorch wheel (smaller than the CUDA build) |
| 4 | `pip install -r vendor/realtime-regression-sw/requirements.txt` | numpy, pandas, pyarrow, omegaconf, pyyaml, requests, tqdm, matplotlib |
| 5 | `python scripts/run_realtime.py --config ../../configs/realtime.ci.yaml --device cpu` (run from `vendor/realtime-regression-sw/`) | **Inference**. Captures the real exit code via `set +e` and `$GITHUB_OUTPUT` |
| 6 | `python scripts/update_site_data.py --exit-code X` | Post-process: copy JSON, embed history, update status |
| 7 | `git commit -m "chore: update forecast data"` + `git push` | Persist `site/data/*.json` changes to `main` |
| 8 | Job summary | Append anchor + first-horizon ap30 to the Actions run summary |
| 9 | `actions/configure-pages@v5` | Signal to Pages: "we're deploying now" |
| 10 | `actions/upload-pages-artifact@v3 path:site` | Upload the `site/` tree as a Pages artifact |
| 11 | `actions/deploy-pages@v4` | Publish the artifact to the live site |

### 4.5 Failure handling and forecast status

The workflow itself **never fails** on inference errors; the outcome is recorded
in `status.json` (for the banner) and as a per-anchor `status` in the archives.
Missing inputs are imputed so a forecast is produced whenever any data is
available ("always emit"); each run is classified as:

| Outcome | `status` | Page banner |
|---|---|---|
| exit 0, ≤ 5% imputed | `ok` | Green: "Forecast is current." |
| exit 0, > 5% imputed | `imputed` | Yellow: data partly imputed |
| exit 2 / other (no forecast) | `failed` | Yellow / red: data gap |

When a run fails, `latest.json` is **not overwritten** — the page keeps showing
the last successful forecast. Because each anchor is attempted three times, a
**don't-downgrade** rule applies: a retry only replaces the stored record when
its status is the same or better (`ok` > `imputed` > `failed`), so a transient
failure never clobbers a good forecast and an `imputed` slot upgrades to `ok`
when clean data returns.

### 4.6 Page banner messages

The status banner shows one of three colours — green (ok), yellow (warn), red
(error) — with the messages below, evaluated top-down (the first match wins).

| Banner | Message | When | Meaning |
|---|---|---|---|
| error | `Status file unavailable …` | `status.json` cannot be fetched | Status file missing (pipeline not run yet / Pages issue) |
| error | `Forecast data unavailable …` | `latest.json` cannot be fetched | No forecast output exists yet |
| error | `Pipeline error: Inference exited with code N. Showing last successful forecast.` | `status = "error"` (unexpected non-0/2 exit) | Inference crashed unexpectedly; last good forecast shown |
| warn | `InsufficientDataError — upstream data gap, waiting for next cycle. Showing last successful forecast.` | `status = "warn"` (exit 2) | Data unavailable / unfillable; last good forecast kept (archive `failed`) |
| warn | `Data is stale: last successful run was X.X hours ago.` | ok but last run > 2 h old | Forecast not updated recently (runs dropped) |
| warn | `X.X% of input data was filled from upstream gaps.` | ok, fresh, but imputed > 5% | Forecast produced on imputed inputs (archive `imputed`) |
| ok | `Forecast is current.` | ok, fresh (< 2 h), imputed ≤ 5% | Fresh, clean forecast (archive `ok`) |

The `InsufficientDataError …` text is `status.json.last_error.message`;
`Showing last successful forecast.` is appended by the page. The banner status
maps to the per-anchor archive status: ok ↔ `ok`, "X% filled" ↔ `imputed`,
"InsufficientDataError" / "Pipeline error" ↔ `failed`.

---

## 5. Model assets

### 5.1 Committed in-tree

`model_best.pth` (~7.2 MB) and `table_stats.pkl` are committed under
`vendor/realtime-regression-sw/checkpoint/`. Because they are part of the repo,
the workflow runs from a plain checkout with no download step and no external
dependency. The two files **must be a matched pair from the same training run** —
mismatched files cause silently miscalibrated forecasts, and there is no runtime
check that enforces the pairing.

The committed checkpoint is force-tracked despite the engine's `*.pth`/`*.pkl`
ignore rules; a `.gitignore` inside `checkpoint/` re-includes the two files.

### 5.2 Updating the checkpoint

To replace the model with a newly trained pair:

```bash
# Drop the new matched pair into the checkpoint directory
cp /path/to/new/model_best.pth  vendor/realtime-regression-sw/checkpoint/
cp /path/to/new/table_stats.pkl vendor/realtime-regression-sw/checkpoint/

git add vendor/realtime-regression-sw/checkpoint/model_best.pth \
        vendor/realtime-regression-sw/checkpoint/table_stats.pkl
git commit -m "Update checkpoint to <training-run-id>"
git push
```

Then manually trigger the workflow (Actions → **Forecast** → **Run workflow**)
and confirm the **"Checkpoint SHA"** field in the page metadata block changed to
the first 12 characters of the new `model_best.pth` SHA256.

Rolling back is symmetric: `git revert` the checkpoint commit (or re-commit the
previous pair) and re-trigger.

> If the engine source also changed (new architecture, different variable
> order), update `vendor/realtime-regression-sw/src/` **together with** the
> checkpoint in the same commit. Otherwise the workflow would run new weights
> against old code (or vice versa) and crash.

### 5.3 Does the config need to change?

Short answer: **usually no**. `configs/realtime.ci.yaml` describes the *model
architecture* and the *runtime environment*, not the trained weights themselves.
A "retrain on more data with the same architecture" does **not** require touching
the config.

| Section | Weights only (same architecture) | Architecture/profile change |
|---------|:-:|:-:|
| `profile.*`               | unchanged | **must update** |
| `experiment.name`         | unchanged | **must update** |
| `paths.*`                 | unchanged | unchanged |
| `sources.*`               | unchanged | unchanged |
| `window.lookback_steps`   | unchanged | **update if input length changes** |
| `window.forecast_steps`   | unchanged | **update if output length changes** |
| `window.boundary_offset_minutes` | unchanged | unchanged |
| `runtime.*`               | unchanged | unchanged |
| `analysis.*`              | unchanged | unchanged |
| `model_provenance.*`      | ⚠️ **recommended** (display-only) | **must update** |

`model_provenance.*` (`val_loss_at_train`, `val_mae_at_train`,
`val_rmse_at_train`) is read at inference time and copied into the output JSON's
`model` block; the page renders `val_mae_at_train` under "Val-MAE at train". If
you ship new weights without updating these, the page shows the **old** metrics
alongside the **new** forecasts — functionally fine but misleading.

When the architecture changes, commit the config change **together with** the
checkpoint and engine-source changes in a single commit, so the workflow never
runs a mismatched combination.

---

## 6. GitHub Pages deployment

### 6.1 "Actions" source vs branch source

We use **Source: GitHub Actions** (not "Deploy from a branch"). This means:

- No `gh-pages` branch exists. Publishing is done by uploading a Pages artifact
  (`actions/upload-pages-artifact@v3`) and then calling
  `actions/deploy-pages@v4`.
- Each run re-deploys the full `site/` directory. This keeps the build
  deterministic and means `main` branch history is not mixed with a parallel
  `gh-pages` history.

### 6.2 URL resolution and NJIT domain mapping

The repo name `ap-prediction` becomes the URL path:

`github.com/njit-research/ap-prediction` (repo name `ap-prediction`)
→ project Pages URL path (`/ap-prediction/`).

GitHub Pages serves the content at the organization Pages host,
`https://njit-research.github.io/ap-prediction/`. The public-facing URL
`https://sites.njit.edu/ap-prediction/` is a domain mapping on top of that host
(an organization custom-domain / path mapping managed on the NJIT side), so both
URLs serve the same content:

- Primary: `https://sites.njit.edu/ap-prediction/`
- Host: `https://njit-research.github.io/ap-prediction/`

The static front end uses only relative paths (`./data/latest.json`), so it is
agnostic to which of these hosts it is served from.

### 6.3 Cache behavior

- JSON files (`latest.json`, `status.json`) are fetched with
  `cache: "no-store"` in `main.js`, so browsers always request a fresh copy.
- HTML and JS files (`index.html`, `main.js`) use GitHub Pages' default cache
  headers. The browser may cache them aggressively — if the page visibly lags
  behind, a hard refresh (`Cmd+Shift+R` / `Ctrl+F5`) forces a fresh pull.

---

## 7. Files & responsibilities

| Path | Purpose |
|------|---------|
| [`.github/workflows/forecast.yml`](../.github/workflows/forecast.yml) | Cron-triggered build+deploy pipeline |
| [`configs/realtime.ci.yaml`](../configs/realtime.ci.yaml) | Active profile + path overrides (checkpoint, stats, event_dir, results_dir relative to the engine root) |
| [`scripts/update_site_data.py`](../scripts/update_site_data.py) | Post-process: read latest forecast JSON, embed recent observed ap30 history, write `site/data/latest.json` + `status.json`, and append the per-anchor forecast archives (`forecast_history.json` / `.csv`) with a `status` (ok / imputed / failed) under the don't-downgrade rule |
| [`site/index.html`](../site/index.html) | Static page shell. Inline CSS. Loads Chart.js v4 + date-fns adapter from jsDelivr CDN |
| [`site/main.js`](../site/main.js) | Fetches `latest.json` + `status.json`; paints banner; renders red observed history, blue forecast with shaded MCD band, and a full-height "now" divider; UTC-formatted x-axis ticks; tooltips in UTC / KST |
| [`site/data/latest.json`](../site/data/latest.json) | Most recent forecast payload (auto-committed by the workflow) |
| [`site/data/status.json`](../site/data/status.json) | Pipeline health (auto-committed by the workflow) |
| [`site/data/forecast_history.json`](../site/data/forecast_history.json) | Rolling first-horizon (+30 min) forecast archive (ap30 + MCD `lower`/`upper` + `status`). Maintained for future re-exposure; not currently plotted (auto-committed) |
| [`site/data/forecast_history.csv`](../site/data/forecast_history.csv) | Rolling 30-day wide ap30 archive (`anchor_timestamp_utc, status, m_30 … m_720`). Maintained but not currently linked for download (auto-committed) |
| [`vendor/realtime-regression-sw/`](../vendor/realtime-regression-sw) | Inlined inference engine (fetch / pipeline / inference / output) + committed checkpoint |

### 7.1 `latest.json` schema

```json
{
  "run_timestamp_utc":    "2026-05-08T01:30:07Z",
  "anchor_timestamp_utc": "2026-05-08T01:30:00Z",
  "model": {
    "profile":          "in6h_out6h_gnn_transformer",
    "checkpoint_path":  "checkpoint/model_best.pth",
    "checkpoint_sha256":"...",
    "val_loss_at_train": 0.0,
    "val_mae_at_train":  6.826,                // raw ap30, 2022-2025 validation split
    "val_rmse_at_train": 12.85
  },
  "input": {
    "event_csv": "/.../dataset/events/20260508013000.csv",
    "sources": {
      "noaa_plasma_url": "...",
      "noaa_mag_url":    "...",
      "gfz_hpo_url":     "..."
    },
    "missing_data_filled_fraction": 0.017
  },
  "forecast": [                                // 12 entries = 6 hours
    {"horizon_steps":1, "horizon_minutes":30, "target_timestamp_utc":"...", "ap30":7.2},
    ...
  ],
  "analysis": {                                // prediction interval
    "mcd": {
      "n_std": 1.96, "noise_variance": 156.888489,
      "mean":  [...], "mc_std": [...],         // MC-dropout mean and sample std
      "std":   [...],                          // sqrt(mc_std^2 + noise_variance)
      "lower": [...], "upper": [...]           // mean -/+ n_std * std, aligned to horizon index
    }
  },
  "history": [                                 // recent observed ap30 (added by update_site_data.py)
    {"timestamp_utc":"...", "ap30":9.0},
    ...
  ]
}
```

`update_site_data.py` embeds up to `HISTORY_STEPS` (96, i.e. 48 hours) of
observed ap30 rows from the event window; with the current 6-hour input profile
the event window provides 12 steps.

### 7.2 `status.json` schema

```json
{
  "status":            "ok" | "warn" | "error",
  "last_success_utc":  "2026-05-08T01:30:07Z",
  "last_attempt_utc":  "2026-05-08T01:30:07Z",
  "last_error": null | {
    "code":    <int>,
    "message": "..."
  }
}
```

---

## 8. Cost and quota

- GitHub Actions Linux runner minutes are **unlimited and free for public
  repos**. The 10-min cron runs ~4,320 times per month; cost is $0.
- GitHub Pages bandwidth: 100 GB/month soft limit. The static site is a few
  hundred KB; nowhere near the limit.
- NOAA and GFZ feeds are unauthenticated public JSON/text; no API key or quota.

---

## 9. Known limitations

1. **Scheduler drift** — GitHub Actions cron is best-effort. A run scheduled for
   14:38 UTC may actually start anywhere from 14:38 to 15:00+. The anchor
   computation handles this by always aligning to the most recent 30-min
   boundary, but the "last updated" timestamp on the page reflects the actual
   run time, not the slot time.
2. **Public weights exposure** — `model_best.pth` is committed in a public repo.
   Anyone can clone and reuse the weights. Acceptable for this project
   (academic); if sensitivity ever changes, move the repo private or externalize
   the checkpoint.
3. **Single-point failure on stats-checkpoint pairing** — if `model_best.pth`
   and `table_stats.pkl` come from different training runs, the model silently
   produces miscalibrated outputs. There is no runtime check that the two match.
4. **No historical archive on the page** — `latest.json` is the only data the
   page shows. Past forecasts are not accessible from the UI (they still exist in
   the git history of `site/data/latest.json`).

---

## 10. Extending the dashboard

Candidate next steps, in rough order of effort:

1. **Historical accuracy view** — archive each run's `latest.json` to
   `site/data/history/YYYYMMDD.json`, plus a rolling `history.json` index. The
   page adds a secondary chart: "forecast-vs-realized MAE over the last 7 days".
2. **hp30 as a second target** — currently only ap30 is on the page. The engine
   also has variants predicting hp30 directly. Add a second line with a toggle.
3. **Attention heatmap** — attention extraction exists in the engine but emits
   PNG. For interactive use, serialize attention weights to JSON and render with
   a canvas heatmap library.

Each of these would be additive — none require restructuring the current
pipeline.
