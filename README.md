# ap-prediction

Public dashboard for 6-hour ap30 geomagnetic index forecasts with a 95%
prediction interval.

- Deployed site: https://sites.njit.edu/ap-prediction/
  (also at https://njit-research.github.io/ap-prediction/)
- Inference engine + model weights: bundled in-tree under `vendor/realtime-regression-sw/`
  (engine developed in [eunsu-park/geoindex-realtime](https://github.com/eunsu-park/geoindex-realtime))
- Update cadence: a new anchor every 30 min; the cron fires every 10 min with
  three attempts per anchor (cron `8,18,28,38,48,58 * * * *`, a backup against
  transient upstream outages)
- Architecture details: [docs/architecture.md](docs/architecture.md)

This repo is **self-contained**: the engine is inlined and the checkpoint
(`model_best.pth` + `table_stats.pkl`) is committed in-tree, so a run needs only
a checkout — no submodule, no GitHub Release download.

## How It Works

1. `.github/workflows/forecast.yml` runs on a 10-min cron (three attempts per
   30-min anchor; a later attempt only overwrites an earlier one at equal-or-
   better status, so a transient failure never clobbers a good forecast).
2. It checks out this repo — the inference engine and the model checkpoint are
   committed in-tree — and runs `scripts/run_realtime.py`. If an upstream feed
   is unreachable the run exits with a "data gap" warning (exit 2) instead of
   failing hard.
3. `scripts/update_site_data.py` copies the newest forecast JSON into
   `site/data/latest.json`, refreshes `site/data/status.json`, and appends to
   the past-forecast archives (`forecast_history.json` / `.csv`).
4. The `site/` directory is published as a GitHub Pages artifact.
5. `site/index.html` fetches `data/latest.json` (+ `forecast_history.json`) on
   load and renders a Chart.js plot of the 12-step (6-hour) ap30 forecast, the
   observed history, and the past-forecast line, with a `forecast_history.csv`
   download link.

## Repository Layout

```
ap-prediction/
├── .github/workflows/forecast.yml   cron-triggered pipeline
├── vendor/realtime-regression-sw/   inlined inference engine + committed checkpoint
├── configs/realtime.ci.yaml         CI path overrides
├── scripts/update_site_data.py      post-process inference output
├── site/
│   ├── index.html                   page shell
│   ├── main.js                      Chart.js render + metadata
│   └── data/
│       ├── latest.json              most recent forecast (committed each run)
│       └── status.json              pipeline status for the banner
└── README.md
```

## One-Time Setup

Enable GitHub Pages: Settings → Pages → Build and deployment → Source:
**GitHub Actions**.

That is the only setup step: the inference engine and the model checkpoint
(`model_best.pth` + `table_stats.pkl`) are committed in-tree, so the workflow
runs from a plain checkout with no asset download or submodule step.

## Updating the Model

Because the engine and weights are vendored in-tree, an upgrade is a payload
refresh, not a submodule bump:

1. Develop and validate in
   [eunsu-park/geoindex-realtime](https://github.com/eunsu-park/geoindex-realtime).
2. Re-inline the engine (`vendor/realtime-regression-sw/`) and replace the
   matched checkpoint pair:

   ```
   cp <new>/model_best.pth  vendor/realtime-regression-sw/checkpoint/
   cp <new>/table_stats.pkl vendor/realtime-regression-sw/checkpoint/
   git add vendor/realtime-regression-sw/checkpoint/model_best.pth \
           vendor/realtime-regression-sw/checkpoint/table_stats.pkl
   git commit -m "Update checkpoint to <training-run-id>"
   ```

3. Matched-pair invariant: `model_best.pth` and `table_stats.pkl` must come from
   the same training run (no runtime validation). See
   [docs/architecture.md](docs/architecture.md) §5 for handling engine-source /
   config changes alongside the weights.

## Trigger a Run Manually

Actions tab → "Forecast" workflow → "Run workflow".
Optionally provide an ISO8601 `now` to replay a specific anchor.

## Failure Handling

`run_realtime.py` exit codes mapped by `scripts/update_site_data.py`:

- `0` → `status.json.status = "ok"`, `latest.json` updated
- `2` → `status.json.status = "warn"` (InsufficientDataError —
  upstream data gap), `latest.json` preserved
- other → `status.json.status = "error"`, `latest.json` preserved

The workflow itself always succeeds (the Actions badge stays green); the page
banner is the true health indicator.

## License

MIT License. See [LICENSE](LICENSE).
