/* Render the latest ap30 forecast. Fetches site/data/latest.json and
 * site/data/status.json, fills the metadata block, paints the status banner,
 * and draws a Chart.js line plot of the 12-step (6-hour) forecast with its
 * 95% prediction interval (analysis.mcd.lower/upper).
 */
(async () => {
  const LATEST_URL = "./data/latest.json";
  const STATUS_URL = "./data/status.json";
  const STALE_MS = 2 * 60 * 60 * 1000;  // 2 hours

  const $ = (id) => document.getElementById(id);
  const fmtUTC = (iso) => {
    const d = new Date(iso);
    return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, "Z");
  };
  const fmtKST = (iso) => {
    const d = new Date(iso);
    const parts = new Intl.DateTimeFormat("en-CA", {
      timeZone: "Asia/Seoul",
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).formatToParts(d).reduce((acc, p) => (acc[p.type] = p.value, acc), {});
    return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute} KST`;
  };
  const fmtDual = (iso) => iso ? `${fmtUTC(iso)} / ${fmtKST(iso)}` : "—";

  // Compact UTC formatter for x-axis tick labels (e.g., "Apr 24 13:00").
  // Chart.js picks tick positions in browser-local time by default; this
  // callback re-formats the tick value in UTC so axis labels match the
  // axis title.
  const fmtTickUTC = (ms) => {
    const d = new Date(ms);
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "UTC",
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", hour12: false,
    }).formatToParts(d).reduce((acc, p) => (acc[p.type] = p.value, acc), {});
    return `${parts.month} ${parts.day} ${parts.hour}:${parts.minute}`;
  };

  const showBanner = (kind, text) => {
    const el = $("status-banner");
    el.className = `status-banner ${kind}`;
    el.textContent = text;
  };

  const fetchJSON = async (url) => {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${url} → ${res.status}`);
    return res.json();
  };

  // Status.json controls the banner. Missing or "error" → red. "warn" → yellow.
  // "ok" + stale data → yellow override.
  let status = null;
  try {
    status = await fetchJSON(STATUS_URL);
  } catch {
    showBanner("error", "Status file unavailable — pipeline may not have run yet.");
  }

  let latest;
  try {
    latest = await fetchJSON(LATEST_URL);
  } catch {
    showBanner("error", "Forecast data unavailable. The pipeline may not have produced any output yet.");
    return;
  }

  // Read input fill fraction for banner logic (metadata panel removed from UI).
  const filled = latest.input?.missing_data_filled_fraction;
  $("last-fetched").textContent = fmtUTC(new Date().toISOString());

  // Banner precedence: explicit status error/warn > staleness > data-quality.
  const runAgeMs = Date.now() - new Date(latest.run_timestamp_utc).getTime();
  if (status?.status === "error") {
    showBanner("error",
      `Pipeline error: ${status.last_error?.message ?? "unknown"}. Showing last successful forecast.`);
  } else if (status?.status === "warn") {
    showBanner("warn",
      `${status.last_error?.message ?? "Pipeline warning."} Showing last successful forecast.`);
  } else if (runAgeMs > STALE_MS) {
    const hours = (runAgeMs / 3_600_000).toFixed(1);
    showBanner("warn", `Data is stale: last successful run was ${hours} hours ago.`);
  } else if (filled != null && filled > 0.05) {
    showBanner("warn",
      `${(filled * 100).toFixed(1)}% of input data was filled from upstream gaps.`);
  } else {
    showBanner("ok", "Forecast is current.");
  }

  // Build chart datasets. Historical observations (gray, no markers) are drawn
  // before the anchor; the forecast (blue, markers) starts at the anchor.
  const forecast = latest.forecast ?? [];
  const history = latest.history ?? [];
  const forecastPoints = forecast.map((e) => ({
    x: new Date(e.target_timestamp_utc),
    y: e.ap30,
  }));
  const historyPoints = history.map((e) => ({
    x: new Date(e.timestamp_utc),
    y: e.ap30,
  }));

  const anchorMs = new Date(latest.anchor_timestamp_utc).getTime();

  // Prediction-interval band (μ ± n_std·σ_pred, σ_pred² = σ_MC² + σ_residual²).
  // The realtime pipeline writes parallel arrays under analysis.mcd aligned to
  // forecast horizon index; absent on older payloads.
  const mcd = latest.analysis?.mcd;
  const lowerArr = mcd?.lower ?? [];
  const upperArr = mcd?.upper ?? [];
  const hasBand =
    lowerArr.length === forecastPoints.length &&
    upperArr.length === forecastPoints.length &&
    forecastPoints.length > 0;
  const uncertaintyLower = hasBand
    ? forecastPoints.map((p, i) => ({ x: p.x, y: lowerArr[i] }))
    : [];
  const uncertaintyUpper = hasBand
    ? forecastPoints.map((p, i) => ({ x: p.x, y: upperArr[i] }))
    : [];

  // Bridge the last observed point to the first forecast point so the two
  // segments visually connect at the anchor.
  const bridge = historyPoints.length > 0 && forecastPoints.length > 0
    ? [historyPoints[historyPoints.length - 1], forecastPoints[0]]
    : [];

  // Vertical "now" divider at the anchor, drawn by a plugin so it spans the full
  // plot height (chartArea top→bottom) regardless of the y-axis scale.
  const nowLinePlugin = {
    id: "nowLine",
    afterDatasetsDraw(chart) {
      const px = chart.scales.x.getPixelForValue(anchorMs);
      if (px == null || Number.isNaN(px)) return;
      const { top, bottom } = chart.chartArea;
      const c = chart.ctx;
      c.save();
      c.beginPath();
      c.setLineDash([2, 2]);
      c.lineWidth = 1;
      c.strokeStyle = "#9ca3af";
      c.moveTo(px, top);
      c.lineTo(px, bottom);
      c.stroke();
      c.restore();
    },
  };

  const ctx = $("forecast-chart").getContext("2d");
  // eslint-disable-next-line no-undef
  new Chart(ctx, {
    type: "line",
    plugins: [nowLinePlugin],
    data: {
      datasets: [
        {
          label: "observed ap30",
          data: historyPoints,
          borderColor: "#dc2626",
          backgroundColor: "#dc2626",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        },
        {
          label: "uncertainty lower",
          data: uncertaintyLower,
          borderColor: "rgba(107, 114, 128, 0)",
          pointRadius: 0,
          fill: false,
          tension: 0.15,
        },
        {
          label: "uncertainty",
          data: uncertaintyUpper,
          borderColor: "rgba(107, 114, 128, 0)",
          backgroundColor: "rgba(107, 114, 128, 0.2)",
          pointRadius: 0,
          fill: "-1",
          tension: 0.15,
        },
        {
          label: "predicted ap30",
          data: forecastPoints,
          borderColor: "#2563eb",
          backgroundColor: "#2563eb",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.15,
        },
        {
          label: "bridge",
          data: bridge,
          borderColor: "#6b7280",
          borderWidth: 1.5,
          borderDash: [4, 3],
          pointRadius: 0,
          showLine: true,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false, axis: "x" },
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            filter: (item) =>
              item.text !== "bridge" && item.text !== "uncertainty lower",
            // Display legend in this fixed order regardless of dataset
            // array order (which is constrained by draw order + fill refs).
            sort: (a, b) => {
              const order = [
                "observed ap30",
                "predicted ap30",
                "uncertainty",
              ];
              return order.indexOf(a.text) - order.indexOf(b.text);
            },
          },
        },
        tooltip: {
          filter: (item) =>
            item.dataset.label !== "bridge" &&
            item.dataset.label !== "uncertainty lower",
          callbacks: {
            title: (items) => {
              const d = items[0].parsed.x;
              const iso = new Date(d).toISOString();
              return `${fmtUTC(iso)}\n${fmtKST(iso)}`;
            },
            label: (item) => `${item.dataset.label.replace(/\s*\(.*\)$/, "")} = ${item.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "hour" },
          title: { display: true, text: "UTC" },
          ticks: {
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 10,
            callback: (value) => fmtTickUTC(value),
          },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: "ap30" },
        },
      },
    },
  });
})();
