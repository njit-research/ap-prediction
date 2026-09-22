"""Realtime ap30 inference — single-run CLI.

Example:
    python scripts/run_realtime.py
    python scripts/run_realtime.py --now 2026-04-19T12:00:00 --device cpu --verbose
    python scripts/run_realtime.py --dry-run        # use tests/fixtures/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make `src` importable when running this script directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.attention import extract_gnn_attention  # noqa: E402
from src.analysis.mcd import mcd_forecast  # noqa: E402
from src.analysis.plotting import plot_adjacency, plot_attention, plot_forecast  # noqa: E402
from src.fetch.gfz_hpo import fetch_hpo  # noqa: E402
from src.fetch.noaa_swpc import fetch_swpc  # noqa: E402
from src.inference.config_loader import load_config  # noqa: E402
from src.inference.model_loader import build_and_load_model, sha256_of  # noqa: E402
from src.inference.predictor import assemble_input_tensor, predict  # noqa: E402
from src.inference.stats_loader import load_stats  # noqa: E402
from src.output.writer import write_forecast  # noqa: E402
from src.pipeline.aggregate import aggregate_30min  # noqa: E402
from src.pipeline.align import InsufficientDataError, align  # noqa: E402
from src.pipeline.event_builder import build_event_csv  # noqa: E402
from src._vendor.normalizer import Normalizer  # noqa: E402

logger = logging.getLogger("realtime")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Realtime ap30 inference")
    parser.add_argument("--config", type=Path,
                        default=_PROJECT_ROOT / "configs" / "realtime.yaml",
                        help="Path to runtime config YAML")
    parser.add_argument("--now", type=str, default=None,
                        help="Override reference time (ISO 8601, UTC).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use tests/fixtures/ instead of live network.")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Override device (default: from config).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Install a console logger with a timestamp format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_now(cli_now: str | None) -> datetime:
    """Parse `--now` override or default to UTC now."""
    if cli_now is None:
        return datetime.now(tz=timezone.utc).replace(tzinfo=None)
    ts = datetime.fromisoformat(cli_now.replace("Z", "+00:00"))
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _fetch_live(cfg, cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download NOAA + GFZ feeds."""
    run_day = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    cache_dir = cache_root / run_day

    logger.info("Fetching NOAA SWPC (plasma + mag)...")
    swpc = fetch_swpc(
        plasma_url=cfg.sources.noaa_plasma_url,
        mag_url=cfg.sources.noaa_mag_url,
        timeout=cfg.sources.download_timeout,
        max_retries=cfg.sources.max_retries,
        cache_dir=cache_dir,
    )

    logger.info("Fetching GFZ Hp30/ap30 nowcast...")
    hpo = fetch_hpo(
        url=cfg.sources.gfz_hpo_url,
        timeout=cfg.sources.download_timeout,
        max_retries=cfg.sources.max_retries,
        cache_dir=cache_dir,
    )

    return swpc, hpo


def _load_fixtures(fixtures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dry-run fallback — load NOAA + GFZ fixtures from disk.

    Expects the following files in `fixtures_dir`:
      - `plasma.json`, `mag.json` (NOAA SWPC format)
      - `hpo.txt` (GFZ Hp30/ap30 nowcast text)

    Args:
        fixtures_dir: Directory containing fixture files.

    Returns:
        Tuple of (swpc_1min, hpo_30min) DataFrames.
    """
    from src.fetch.noaa_swpc import _rows_to_dataframe, _MAG_RENAME, _PLASMA_RENAME, _numeric
    from src._vendor.parse_hpo import HP30, parse_hpo

    plasma_payload = json.loads((fixtures_dir / "plasma.json").read_text())
    mag_payload = json.loads((fixtures_dir / "mag.json").read_text())
    hpo_text = (fixtures_dir / "hpo.txt").read_text()

    plasma = _rows_to_dataframe(plasma_payload).rename(columns=_PLASMA_RENAME)
    plasma = _numeric(plasma, ["np", "v", "t"])
    plasma = plasma[["datetime", "np", "v", "t"]].sort_values("datetime").reset_index(drop=True)

    mag = _rows_to_dataframe(mag_payload).rename(columns=_MAG_RENAME)
    mag = _numeric(mag, ["bx", "by", "bz", "bt"])
    mag = mag[["datetime", "bx", "by", "bz", "bt"]].sort_values("datetime").reset_index(drop=True)

    swpc = plasma.merge(mag, on="datetime", how="outer").sort_values("datetime").reset_index(drop=True)
    hpo = parse_hpo(hpo_text, HP30).rename(columns={"Hp30": "hp30", "ap30": "ap30"})
    hpo = hpo.dropna(subset=["datetime"])[["datetime", "hp30", "ap30"]].reset_index(drop=True)
    return swpc, hpo


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    configure_logging(args.verbose)

    cfg = load_config(args.config)
    device_name = args.device or cfg.runtime.device
    now = resolve_now(args.now)
    logger.info("Reference time (UTC): %s", now.isoformat())

    cache_root = _PROJECT_ROOT / cfg.paths.cache_dir
    event_dir = _PROJECT_ROOT / cfg.paths.event_dir
    results_dir = _PROJECT_ROOT / cfg.paths.results_dir

    if args.dry_run:
        fixtures = _PROJECT_ROOT / "tests" / "fixtures"
        logger.info("--dry-run enabled: using fixtures at %s", fixtures)
        swpc_1min, hpo = _load_fixtures(fixtures)
    else:
        try:
            swpc_1min, hpo = _fetch_live(cfg, cache_root)
        except Exception as exc:  # noqa: BLE001
            # An upstream feed (NOAA/GFZ) being unreachable is a transient data
            # gap, not a pipeline error. Exit 2 (same as InsufficientDataError)
            # so the site shows a "waiting for next cycle" warning instead of a
            # hard error, and the previous forecast is preserved.
            logger.error("Upstream fetch failed (treating as data gap): %s", exc)
            return 2

    # Aggregate 1-min → 30-min, covering a window that comfortably spans the lookback.
    # Floor the window boundaries to :00/:30 so the aggregation grid matches the
    # resample output and doesn't produce spurious all-NaN reindexed rows.
    lookback_steps = int(cfg.window.lookback_steps)
    now_ts = pd.Timestamp(now).floor("30min")
    agg_start = now_ts - pd.Timedelta(minutes=30 * (lookback_steps + 6))
    agg_end = now_ts + pd.Timedelta(minutes=30)
    sw_30min = aggregate_30min(swpc_1min, start=agg_start.to_pydatetime(),
                               end=agg_end.to_pydatetime())

    # Align onto the lookback grid and enforce missing-data policy.
    md = cfg.runtime.missing_data
    try:
        aligned = align(
            sw_30min=sw_30min,
            hpo=hpo,
            now=now,
            lookback_steps=lookback_steps,
            boundary_offset_minutes=int(cfg.window.boundary_offset_minutes),
            max_gap_fraction=float(md.max_gap_fraction),
            ffill_limit_steps=int(md.ffill_limit_steps),
            require_recent_steps_present=int(md.require_recent_steps_present),
            anchor_rollback_max_attempts=int(md.anchor_rollback_max_attempts),
        )
    except InsufficientDataError as exc:
        logger.error("%s", exc)
        return 2

    # Materialize the event CSV with the training-schema column order.
    input_vars = list(cfg.data.timeseries.input_variables)
    event_csv = build_event_csv(aligned.frame, aligned.t_end,
                                out_dir=event_dir,
                                input_variables=input_vars)

    # Sanity: row / column counts.
    df = pd.read_csv(event_csv)
    assert len(df) == lookback_steps, f"Event CSV row count mismatch: {len(df)} != {lookback_steps}"

    # Load stats + build model + predict.
    required_vars = input_vars + list(cfg.data.timeseries.target_variables)
    stats = load_stats(Path(cfg.paths.stats_file), required_variables=required_vars)
    normalizer = Normalizer(stat_dict=stats,
                            method_config=cfg.data.timeseries.normalization)

    checkpoint_path = Path(cfg.paths.checkpoint)
    model, device = build_and_load_model(cfg, checkpoint_path, device_name)

    forecast = predict(cfg, model, normalizer, event_csv, device)
    assert forecast.shape == (int(cfg.window.forecast_steps),), \
        f"Unexpected forecast shape: {forecast.shape}"

    # Analysis (MCD + attention). Build the input tensor once and reuse.
    event_df = pd.read_csv(event_csv)
    input_vars_list = list(cfg.data.timeseries.input_variables)
    target_var = list(cfg.data.timeseries.target_variables)[0]
    input_tensor = assemble_input_tensor(event_df, input_vars_list, normalizer).to(device)

    analysis_cfg = cfg.analysis
    mcd_result = None
    if analysis_cfg.mcd.enable:
        logger.info("Running MCD (num_samples=%d)...", analysis_cfg.mcd.num_samples)
        mcd_result = mcd_forecast(
            model=model,
            input_tensor=input_tensor,
            normalizer=normalizer,
            target_variable=target_var,
            num_samples=int(analysis_cfg.mcd.num_samples),
            n_std=float(analysis_cfg.mcd.n_std),
            # σ_pred² = σ_MC² + σ_residual²; absent / 0 keeps the raw MC band.
            noise_variance=float(analysis_cfg.mcd.get("noise_variance", 0.0) or 0.0),
        )
        logger.info("MCD mean=[%.2f..%.2f], mean σ_MC=%.3f, mean σ_pred=%.3f (noise var %.3f)",
                    mcd_result.mean.min(), mcd_result.mean.max(),
                    float(mcd_result.mc_std.mean()), float(mcd_result.std.mean()),
                    mcd_result.noise_variance)

    attn_result = None
    if analysis_cfg.attention.enable:
        logger.info("Extracting attention + GNN adjacency...")
        node_labels = list(cfg.data.timeseries.gnn_variable_groups.keys())
        attn_result = extract_gnn_attention(
            model=model,
            input_tensor=input_tensor,
            node_labels=node_labels,
        )
        logger.info("Captured %d attention layer(s); adjacency shape %s",
                    len(attn_result.attention_per_layer),
                    attn_result.adjacency.shape)

    # Render plots + optional NPZ dump.
    anchor_stem = aligned.t_end.strftime("%Y%m%d%H%M%S")
    results_day_dir = Path(results_dir) / aligned.t_end.strftime("%Y%m%d")
    results_day_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = analysis_cfg.plot
    plot_paths = {}
    if plot_cfg.enable:
        plot_paths["forecast"] = plot_forecast(
            event_df=event_df,
            t_end=aligned.t_end,
            forecast=forecast,
            mcd=mcd_result,
            save_path=results_day_dir / f"{anchor_stem}_forecast.png",
            target_variable=target_var,
            history_steps=int(plot_cfg.history_steps),
            interval_minutes=30,
            dpi=int(plot_cfg.dpi),
        )
        if attn_result is not None:
            plot_paths["attention"] = plot_attention(
                attention=attn_result,
                t_end=aligned.t_end,
                save_path=results_day_dir / f"{anchor_stem}_attention.png",
                interval_minutes=30,
                dpi=int(plot_cfg.dpi),
            )
            plot_paths["adjacency"] = plot_adjacency(
                attention=attn_result,
                t_end=aligned.t_end,
                save_path=results_day_dir / f"{anchor_stem}_adjacency.png",
                dpi=int(plot_cfg.dpi),
            )

    npz_path = None
    if (mcd_result is not None or attn_result is not None) and analysis_cfg.attention.save_npz:
        npz_path = results_day_dir / f"{anchor_stem}_analysis.npz"
        npz_payload = {}
        if mcd_result is not None:
            npz_payload["mcd_samples"] = mcd_result.samples
            npz_payload["mcd_mean"] = mcd_result.mean
            npz_payload["mcd_std"] = mcd_result.std
            npz_payload["mcd_lower"] = mcd_result.lower
            npz_payload["mcd_upper"] = mcd_result.upper
        if attn_result is not None:
            npz_payload["adjacency"] = attn_result.adjacency
            npz_payload["node_labels"] = np.array(attn_result.node_labels)
            for i, attn in enumerate(attn_result.attention_per_layer):
                npz_payload[f"attention_layer_{i}"] = attn
                npz_payload[f"temporal_importance_layer_{i}"] = \
                    attn_result.temporal_importance_per_layer[i]
        np.savez_compressed(npz_path, **npz_payload)
        logger.info("Saved analysis NPZ: %s", npz_path)

    # Provenance metadata.
    ckpt_sha = sha256_of(checkpoint_path)
    model_meta = {
        "profile": cfg.profile.name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha[:12],
        "val_loss_at_train": float(cfg.model_provenance.val_loss_at_train),
        "val_mae_at_train": float(cfg.model_provenance.val_mae_at_train),
        "val_rmse_at_train": float(cfg.model_provenance.val_rmse_at_train),
    }
    source_urls = {
        "noaa_plasma_url": str(cfg.sources.noaa_plasma_url),
        "noaa_mag_url": str(cfg.sources.noaa_mag_url),
        "gfz_hpo_url": str(cfg.sources.gfz_hpo_url),
    }

    analysis_meta = {}
    if mcd_result is not None:
        analysis_meta["mcd"] = {
            "num_samples": int(len(mcd_result.samples)),
            "n_std": float(mcd_result.n_std),
            "noise_variance": float(mcd_result.noise_variance),
            "mean": [float(v) for v in mcd_result.mean],
            "std": [float(v) for v in mcd_result.std],
            "mc_std": [float(v) for v in mcd_result.mc_std],
            "lower": [float(v) for v in mcd_result.lower],
            "upper": [float(v) for v in mcd_result.upper],
        }
    if attn_result is not None:
        analysis_meta["attention"] = {
            "num_layers": len(attn_result.attention_per_layer),
            "num_heads": int(attn_result.attention_per_layer[0].shape[0]),
            "seq_len": int(attn_result.attention_per_layer[0].shape[-1]),
            "node_labels": attn_result.node_labels,
        }
    if plot_paths:
        analysis_meta["plots"] = {k: str(v) for k, v in plot_paths.items()}
    if npz_path is not None:
        analysis_meta["analysis_npz"] = str(npz_path)

    artifacts = write_forecast(
        forecast=forecast,
        t_end=aligned.t_end,
        event_csv=event_csv,
        results_dir=results_dir,
        model_meta=model_meta,
        source_urls=source_urls,
        missing_data_filled_fraction=aligned.filled_fraction,
        analysis=analysis_meta or None,
    )

    print(f"\nForecast written: {artifacts.json_path}")
    for label, path in plot_paths.items():
        print(f"  {label:>10}: {path}")
    if npz_path is not None:
        print(f"  {'analysis':>10}: {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
