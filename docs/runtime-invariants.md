# Runtime Invariants

Safety-critical constraints for the inference engine
(`vendor/realtime-regression-sw/`). Violating any of these will produce wrong
forecasts or runtime errors. Read this before modifying inference code.

## 1. Input Tensor Shape

The model receives exactly `(1, L, 22)`, where `L` is `window.lookback_steps`
from `configs/realtime.ci.yaml` (12 for the active `in6h_out6h_gnn_transformer`
profile; it was 24 under the previous `in12h` profile):

- **1** — batch size (single live sample).
- **L** — 30-min steps of input history (12 = 6 hours for `in6h`).
- **22** — 21 solar-wind parameters + `ap30`, in the exact order of
  `input_variables` in
  `vendor/realtime-regression-sw/configs/profile/base.yaml`.

The variable ordering is a hard invariant: the 22 columns must be supplied in
the exact order the model was trained on, or the forecast is silently wrong.

## 2. Event CSV Schema

The generated event CSV carries **all 23 training columns**
(`datetime` + 21 SW + `ap30` + `hp30`), even though `hp30` is not fed to the
model. This preserves training-schema compatibility; do not drop `hp30` from the
CSV even if it feels redundant.

## 3. Normalization Coupling

Normalization is performed with `table_stats.pkl`, produced during training and
committed alongside the weights under
`vendor/realtime-regression-sw/checkpoint/`.

- The checkpoint (`model_best.pth`) and the stats file (`table_stats.pkl`) must
  be a **matched pair from the same training run**. Replacing the model requires
  replacing both together (see [architecture.md §5](architecture.md#5-model-assets)).
- Using a mismatched stats file will silently produce miscalibrated forecasts —
  there is no runtime check that can detect this mismatch.

## 4. Anchor Time Rollback

The anchor time `t_end` is computed as:

1. Floor the current wall clock to the most recent completed 30-min boundary.
2. Subtract `boundary_offset_minutes` to account for publisher delay.

If the last few steps of the input window are NaN even after forward-fill,
`t_end` rolls back one 30-min step. This rollback happens **up to 2 times**; if
the window is still insufficient, the run exits with `InsufficientDataError`
(exit code 2), and the page shows a "data gap" warning banner while preserving
the last successful forecast.
