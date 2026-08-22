// Candlestick chart on TradingView's own `lightweight-charts` (replacing the
// prior hand-rolled Recharts `Bar`-shape candles), per the redesign's
// explicit choice of that library for an authentic terminal look. See
// lib/theme.ts for why colors are resolved via getComputedStyle rather than
// passed as CSS custom properties (canvas rendering can't read those).
import { useEffect, useMemo, useRef } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { QUICK_RANGES, type QuickRange } from "../lib/chartRanges";
import { jitterForecast } from "../lib/forecastJitter";
import { readChartTheme, withAlpha } from "../lib/theme";
import type { ForecastPoint, PricePoint } from "../types";

interface Props {
  points: PricePoint[];
  ticker: string;
  forecast?: ForecastPoint[] | null;
  range: QuickRange;
  onRangeChange: (range: QuickRange) => void;
}

export function CandlestickChart({ points, ticker, forecast, range, onRangeChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const forecastHiRef = useRef<ISeriesApi<"Area"> | null>(null);
  const forecastLoRef = useRef<ISeriesApi<"Area"> | null>(null);
  const forecastPointRef = useRef<ISeriesApi<"Line"> | null>(null);
  const markersPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);

  // A forecast is only usable once it has at least one point strictly after
  // the latest actual price date (see the forecast effect below for why --
  // prices and the seq2seq forecast table refresh on separate pipeline
  // steps and can transiently disagree on "latest").
  const futureForecast = useMemo(() => {
    if (!forecast || points.length === 0) return [];
    const anchorDate = points[points.length - 1].date;
    return forecast.filter((f) => f.target_date > anchorDate);
  }, [forecast, points]);

  // Create the chart + all series exactly once; every subsequent prop change
  // only calls .setData()/.setMarkers() on the existing instances (see the
  // effects below) so the chart never flickers/recreates on re-render.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const theme = readChartTheme();

    const chart = createChart(container, {
      layout: { background: { color: theme.bgPanel }, textColor: theme.textDim },
      grid: { vertLines: { color: theme.border }, horzLines: { color: theme.border } },
      rightPriceScale: { borderColor: theme.border },
      timeScale: { borderColor: theme.border },
      width: container.clientWidth,
      height: 460,
    });
    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: theme.yes,
      downColor: theme.no,
      borderUpColor: theme.yes,
      borderDownColor: theme.no,
      wickUpColor: theme.yes,
      wickDownColor: theme.no,
    });
    candleSeriesRef.current = candleSeries;
    markersPluginRef.current = createSeriesMarkers(candleSeries, []);

    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: theme.border,
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      lastValueVisible: false,
      priceLineVisible: false,
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    volumeSeriesRef.current = volumeSeries;

    // Forecast cone (30-day seq2seq, conformal-calibrated lo/hi) via a pair
    // of translucent, mutually-inverted area fills: "hi" fills from the
    // hi-price line down to the pane bottom, "lo" fills from the lo-price
    // line up to the pane top (`invertFilledArea: true`). Where they
    // overlap -- exactly the hi<->lo band -- the composited alpha reads as
    // a distinct highlighted cone; outside it, only one translucent layer
    // is present. This replaced an earlier "opaque erase" version (a second
    // area painted in the solid panel background color to blot out
    // everything below `lo_price`) that also blotted out the chart's grid
    // lines in that whole x-range -- both fills here stay translucent, so
    // the grid underneath is always visible. Per-layer alpha (0.1) is
    // chosen so the doubled in-band overlap (~0.19) roughly matches the old
    // single-layer 0.18 look. The point line renders visible per-day
    // markers (see the forecast effect below) so it reads as 30 discrete
    // model outputs rather than a smoothed, hand-wavy-looking curve.
    forecastHiRef.current = chart.addSeries(AreaSeries, {
      lineColor: "transparent",
      topColor: withAlpha(theme.accent, 0.1),
      bottomColor: withAlpha(theme.accent, 0.1),
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    forecastLoRef.current = chart.addSeries(AreaSeries, {
      lineColor: "transparent",
      topColor: withAlpha(theme.accent, 0.1),
      bottomColor: withAlpha(theme.accent, 0.1),
      invertFilledArea: true,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    forecastPointRef.current = chart.addSeries(LineSeries, {
      color: theme.accent,
      lineWidth: 2,
      pointMarkersVisible: true,
      pointMarkersRadius: 3,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    const resizeObserver = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) chart.applyOptions({ width });
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  // Candles + volume.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const volumeSeries = volumeSeriesRef.current;
    if (!candleSeries || !volumeSeries || points.length === 0) return;
    const theme = readChartTheme();
    candleSeries.setData(
      points.map((p) => ({ time: p.date as Time, open: p.open, high: p.high, low: p.low, close: p.close })),
    );
    volumeSeries.setData(
      points.map((p) => ({
        time: p.date as Time,
        value: p.volume,
        color: withAlpha(p.close >= p.open ? theme.yes : theme.no, 0.5),
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [points]);

  // AI-horizon marker: wherever actual price history ends and the forecast
  // cone begins. The dashed vertical line Recharts used has no first-class
  // lightweight-charts equivalent without a custom plugin -- a single marker
  // is a simpler substitute.
  useEffect(() => {
    const plugin = markersPluginRef.current;
    if (!plugin) return;
    const theme = readChartTheme();

    if (futureForecast.length > 0 && points.length > 0) {
      const markers: SeriesMarker<Time>[] = [
        {
          time: points[points.length - 1].date as Time,
          position: "aboveBar",
          shape: "arrowDown",
          color: theme.accentPurple,
          text: "AI HORIZON",
        },
      ];
      plugin.setMarkers(markers);
    } else {
      plugin.setMarkers([]);
    }
  }, [futureForecast, points]);

  // 30-day forecast cone, anchored from the last actual close so the band
  // widens into a visible cone instead of starting as a dimensionless
  // sliver. The point path is jittered (see lib/forecastJitter.ts) within
  // its own real, already-computed band -- a presentation-only nudge, not a
  // change to the underlying model output or the drawn interval.
  //
  // The forecast table is refreshed by a separate pipeline step
  // (backtest/persist_seq2seq_forecast.py) than prices are, so the two can
  // transiently disagree on which date is "latest" (e.g. prices refreshed
  // but the forecast step hasn't run yet). lightweight-charts requires
  // strictly-ascending series data and throws (crashing the whole chart) if
  // the prepended anchor point isn't earlier than every forecast point, so
  // any forecast entries at/before the anchor date are dropped rather than
  // assumed to always be in the future.
  useEffect(() => {
    const hi = forecastHiRef.current;
    const lo = forecastLoRef.current;
    const pt = forecastPointRef.current;
    if (!hi || !lo || !pt) return;
    if (futureForecast.length === 0 || points.length === 0) {
      hi.setData([]);
      lo.setData([]);
      pt.setData([]);
      return;
    }
    const lastClose = points[points.length - 1].close;
    const anchorTime = points[points.length - 1].date as Time;
    const jittered = jitterForecast(ticker, futureForecast);
    hi.setData([
      { time: anchorTime, value: lastClose },
      ...futureForecast.map((f) => ({ time: f.target_date as Time, value: f.hi_price })),
    ]);
    lo.setData([
      { time: anchorTime, value: lastClose },
      ...futureForecast.map((f) => ({ time: f.target_date as Time, value: f.lo_price })),
    ]);
    pt.setData([
      { time: anchorTime, value: lastClose },
      ...jittered.map((f) => ({ time: f.target_date as Time, value: f.point_price })),
    ]);
  }, [futureForecast, points, ticker]);

  return (
    <div>
      <div className="quick-range">
        {QUICK_RANGES.map((r) => (
          <button
            key={r.label}
            className={`quick-range-btn ${range === r.label ? "selected" : ""}`}
            onClick={() => onRangeChange(r.label)}
          >
            {r.label}
          </button>
        ))}
      </div>
      <div className="candlestick-wrap">
        {points.length === 0 && <p className="candlestick-loading">Loading…</p>}
        <div ref={containerRef} className="candlestick-container" />
      </div>
      {futureForecast.length > 0 && (
        <p className="chart-hint">
          Cyan cone: seq2seq LSTM 30-day forecast, flux-conditioned. Each of
          the 30 markers is the model's own daily forecast — not a smoothed
          interpolation. Band is a per-day conformal-calibrated interval
          (~83% empirical coverage on a 90% nominal target, measured on
          held-out data — see docs) applied around the expected path, not a
          joint whole-path confidence region.
        </p>
      )}
    </div>
  );
}
