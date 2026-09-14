"use strict";

const refreshMs = Math.max(Number(document.documentElement.dataset.refreshMs) || 1000, 250);
const byId = (id) => document.getElementById(id);
const charts = new Map();
let latest = null;
let strategyQuery = "";

function node(tag, className = "", value = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (value !== "") item.textContent = String(value);
  return item;
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function fmt(value, digits = 2) {
  const parsed = finite(value);
  return parsed === null ? "—" : parsed.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function signed(value, digits = 2, suffix = "") {
  const parsed = finite(value);
  if (parsed === null) return "—";
  return `${parsed >= 0 ? "+" : ""}${fmt(parsed, digits)}${suffix}`;
}

function percent(value, digits = 0) {
  const parsed = finite(value);
  return parsed === null ? "—" : `${(parsed * 100).toFixed(digits)}%`;
}

function directionClass(value) {
  const text = String(value || "").toLowerCase();
  if (["up", "buy", "long", "active"].includes(text)) return "up";
  if (["down", "sell", "short", "error"].includes(text)) return "down";
  return "hold";
}

function age(timestamp) {
  if (!timestamp) return "awaiting ticks";
  const seconds = Math.max((Date.now() - Date.parse(timestamp)) / 1000, 0);
  if (!Number.isFinite(seconds)) return "unknown";
  if (seconds < 90) return `${Math.round(seconds)}s ago`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

function legList(payload) {
  if (payload.kind === "board") return payload.legs || [];
  if (payload.kind === "market" && payload.market) {
    return [{ label: "INDEX", kind: "INDEX", strike: 0, greeks: {}, verdict: null, market: payload.market }];
  }
  return [];
}

function spotLeg(payload) {
  const legs = legList(payload);
  return legs.find((leg) => leg.label === "INDEX") || legs[0] || null;
}

function metric(label, value, sub = "", tone = "") {
  const card = node("article", "metric");
  card.append(node("div", "label", label));
  card.append(node("div", `value ${tone}`, value));
  card.append(node("div", "sub", sub));
  return card;
}

function renderOverview(payload) {
  const root = byId("overview");
  const spot = spotLeg(payload);
  const market = spot?.market || {};
  const chain = payload.chain || {};
  const change = finite(market.change) || 0;
  const items = [
    metric("instrument", payload.symbol || market.symbol || "—", `source ${market.source || "—"}`, "cyan"),
    metric("spot / last", fmt(payload.spot ?? market.last_price), `${signed(market.change)} · ${signed(market.change_pct, 2, "%")}`, change >= 0 ? "up" : "down"),
    metric("market regime", String(market.regime || "unknown").toUpperCase(), `conviction ${signed(market.conviction, 2)}`),
    metric("option chain", chain.expiry || "—", `PCR ${fmt(chain.pcr)} · ATM ${fmt(chain.atm_strike, 0)}`),
    metric("feed state", String(payload.status || "waiting").toUpperCase(), `tick ${age(market.last_tick_ts)}`, age(market.last_tick_ts).startsWith("0") ? "up" : ""),
    metric("research state", market.research?.ai ? "AUTO-LEARNING" : "WARMING", `${market.research?.workers ?? 1} CPU · ${fmt(market.research?.compute_ms, 1)}ms/bar`, "prediction"),
  ];
  root.replaceChildren(...items);
}

function readout(label, value, tone = "") {
  const item = node("div", "readout");
  item.append(node("label", "", label));
  item.append(node("strong", tone, value));
  return item;
}

function signalFor(market, horizon = 1) {
  const signals = market?.research?.ai?.signals || {};
  return signals[horizon] || signals[String(horizon)] || null;
}

function projectionItem(projected) {
  const item = node("div", "projection");
  item.append(node("span", "", `+${projected.horizon ?? "?"} BAR`));
  item.append(node("strong", "", `${fmt(projected.close)} · ${signed(projected.expected_move_bps ?? projected.change_bps, 1, "bp")}`));
  return item;
}

function renderLegs(payload) {
  const root = byId("leg-grid");
  charts.clear();
  const cards = legList(payload).map((leg, index) => {
    const market = leg.market || {};
    const ai = signalFor(market);
    const card = node("article", "leg-card");
    const head = node("div", "leg-head");
    const identity = node("div");
    identity.append(node("div", "leg-name", leg.strike ? `${leg.label} ${fmt(leg.strike, 0)}` : leg.label));
    identity.append(node("div", "leg-symbol", market.symbol || market.instrument_key || "—"));
    const quote = node("div");
    quote.append(node("div", "leg-price", fmt(market.last_price)));
    quote.append(node("div", `leg-change ${finite(market.change) >= 0 ? "up" : "down"}`, `${signed(market.change)} (${signed(market.change_pct, 2, "%")})`));
    head.append(identity, quote);

    const wrap = node("div", "chart-wrap");
    const canvas = node("canvas", "market-chart");
    canvas.setAttribute("aria-label", `${leg.label} actual and projected candlestick chart`);
    wrap.append(canvas);

    const reads = node("div", "leg-readouts");
    reads.append(readout("AUTO-AI +1", ai ? `${ai.action} · P↑ ${fmt(ai.p_up, 2)}` : "WARMING", ai ? directionClass(ai.action) : "muted"));
    if (leg.kind === "CE" || leg.kind === "PE") {
      reads.append(readout("DELTA / IV", `${signed(leg.greeks?.delta, 2)} · ${percent(leg.greeks?.iv, 1)}`));
      reads.append(readout("VERDICT / EDGE", leg.verdict ? `${leg.verdict.action} · ${signed(leg.verdict.edge_bps, 1, "bp")}` : "—", directionClass(leg.verdict?.action)));
    } else {
      reads.append(readout("TRUST / CONF", ai ? `${percent(ai.trust_score)} · ${percent(ai.confidence)}` : "—"));
      reads.append(readout("REGIME / VIEW", `${market.regime || "unknown"} · ${signed(market.conviction, 2)}`, finite(market.conviction) >= 0 ? "up" : "down"));
    }

    const strip = node("div", "projection-strip");
    const projections = (market.projections || []).slice(0, 3);
    if (projections.length) projections.forEach((item) => strip.append(projectionItem(item)));
    else strip.append(node("div", "empty", "WAITING FOR CURRENT-BAR PROJECTION"));

    card.append(head, wrap, reads, strip);
    charts.set(canvas, market);
    window.requestAnimationFrame(() => drawChart(canvas, market));
    card.style.setProperty("--leg-index", index);
    return card;
  });
  root.replaceChildren(...cards);
  if (!cards.length) root.append(node("div", "empty", "NO INSTRUMENT SNAPSHOT YET"));
}

function drawChart(canvas, market) {
  const width = Math.max(canvas.clientWidth, 240);
  const height = Math.max(canvas.clientHeight, 180);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const actual = (market.candles || []).filter(validCandle);
  const projected = (market.projections || []).filter(validCandle);
  const usableActual = actual.slice(-Math.max(Math.floor(width / 7) - projected.length, 24));
  const all = [...usableActual, ...projected];
  if (!all.length) {
    ctx.fillStyle = "#718078";
    ctx.font = "10px monospace";
    ctx.fillText("AWAITING CANDLES", 14, 28);
    return;
  }

  const left = 8;
  const right = 58;
  const top = 12;
  const bottom = 19;
  const plotWidth = Math.max(width - left - right, 50);
  const plotHeight = Math.max(height - top - bottom, 50);
  let low = Math.min(...all.map((item) => Number(item.low)));
  let high = Math.max(...all.map((item) => Number(item.high)));
  const padding = Math.max((high - low) * 0.08, Math.abs(high || 1) * 0.0002);
  low -= padding;
  high += padding;
  const range = high - low || 1;
  const y = (price) => top + ((high - Number(price)) / range) * plotHeight;
  const step = plotWidth / Math.max(all.length, 1);
  const bodyWidth = Math.max(Math.min(step * 0.62, 8), 2);

  ctx.font = "9px monospace";
  ctx.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const lineY = top + (plotHeight * line) / 4;
    const price = high - (range * line) / 4;
    ctx.strokeStyle = "#142018";
    ctx.beginPath();
    ctx.moveTo(left, lineY + 0.5);
    ctx.lineTo(left + plotWidth, lineY + 0.5);
    ctx.stroke();
    ctx.fillStyle = "#718078";
    ctx.fillText(price.toLocaleString("en-IN", { maximumFractionDigits: 2 }), left + plotWidth + 5, lineY + 3);
  }

  if (projected.length && usableActual.length) {
    const separator = left + usableActual.length * step;
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = "#5588ff";
    ctx.beginPath();
    ctx.moveTo(separator, top);
    ctx.lineTo(separator, top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#7698ff";
    ctx.fillText("FWD", Math.min(separator + 4, width - right - 24), top + 10);
  }

  all.forEach((candle, index) => {
    const isProjection = index >= usableActual.length;
    const rising = Number(candle.close) >= Number(candle.open);
    const color = isProjection ? "#5588ff" : rising ? "#44ff88" : "#ff5263";
    const center = left + (index + 0.5) * step;
    const openY = y(candle.open);
    const closeY = y(candle.close);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = isProjection ? Math.max(0.45, 1 - (index - usableActual.length) * 0.15) : 0.88;
    ctx.beginPath();
    ctx.moveTo(center, y(candle.high));
    ctx.lineTo(center, y(candle.low));
    ctx.stroke();
    const bodyTop = Math.min(openY, closeY);
    const bodyHeight = Math.max(Math.abs(closeY - openY), 1);
    if (rising && !isProjection) {
      ctx.strokeRect(center - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
    } else {
      ctx.fillRect(center - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
    }
  });
  ctx.globalAlpha = 1;
}

function validCandle(item) {
  return item && [item.open, item.high, item.low, item.close].every((value) => finite(value) !== null);
}

function table(headers, rows, classes = []) {
  if (!rows.length) return node("div", "empty", "WAITING FOR MEASURED DATA");
  const result = node("table");
  const head = node("thead");
  const headRow = node("tr");
  headers.forEach((name) => headRow.append(node("th", "", name)));
  head.append(headRow);
  const body = node("tbody");
  rows.forEach((values) => {
    const row = node("tr");
    values.forEach((value, index) => {
      const cell = node("td", classes[index] || "");
      if (value instanceof Node) cell.append(value);
      else cell.textContent = String(value ?? "—");
      row.append(cell);
    });
    body.append(row);
  });
  result.append(head, body);
  return result;
}

function tag(value) {
  return node("span", `tag ${directionClass(value)}`, value || "WAIT");
}

function renderAi(payload) {
  const cards = spotLeg(payload)?.market?.research?.ai?.scorecards || [];
  const rows = cards.map((card) => [
    card.algorithm,
    card.samples ?? 0,
    `${card.hits ?? 0}/${card.misses ?? 0}`,
    card.samples ? percent(card.accuracy, 1) : "—",
    card.samples ? percent(card.rolling_accuracy, 1) : "—",
    signed(card.net_pnl_bps, 1, "bp"),
    signed(card.drawdown_bps, 1, "bp"),
    percent(card.trust_score),
  ]);
  byId("ai-scorecards").replaceChildren(table(
    ["algorithm", "n", "hit/miss", "accuracy", "rolling", "net", "drawdown", "trust"],
    rows,
    ["cyan"],
  ));
}

function renderPredictions(payload) {
  const rows = [];
  legList(payload).forEach((leg) => {
    const ai = leg.market?.research?.ai?.signals || {};
    Object.entries(ai).sort(([a], [b]) => Number(a) - Number(b)).forEach(([horizon, item]) => {
      const members = item.algorithms || [];
      const leader = [...members].sort((a, b) => (b.trust_score || 0) - (a.trust_score || 0))[0];
      rows.push([
        leg.label,
        `+${horizon}`,
        tag(item.action),
        fmt(item.p_up, 3),
        percent(item.confidence),
        percent(item.trust_score),
        leader ? `${leader.name} ${fmt(leader.p_up, 2)}` : "—",
        members.length,
      ]);
    });
  });
  byId("live-predictions").replaceChildren(table(
    ["leg", "bar", "action", "P(up)", "confidence", "trust", "best member", "models"],
    rows,
  ));
}

function renderStrategies(payload) {
  const legs = legList(payload);
  const spot = spotLeg(payload);
  const signals = spot?.market?.signals || [];
  const normalized = strategyQuery.trim().toLowerCase();
  const filtered = signals.filter((signal) => !normalized || `${signal.strategy} ${signal.reason}`.toLowerCase().includes(normalized));
  const rows = filtered.map((signal) => {
    const states = legs.map((leg) => {
      const match = (leg.market?.signals || []).find((candidate) => candidate.strategy === signal.strategy);
      const state = match?.meta?.state || "WAIT";
      return state === "ACTIVE" ? tag(match.direction) : tag(state);
    });
    const meta = signal.meta || {};
    const measured = meta.scored ? ` · H/M ${meta.hits || 0}/${Math.max((meta.scored || 0) - (meta.hits || 0), 0)} · net ${signed(meta.net_pnl_bps, 1, "bp")} · trust ${percent(meta.trust_score)}` : " · warming scorecard";
    return [signal.strategy, ...states, `${signal.reason || "no setup"}${measured}`];
  });
  const headers = ["strategy", ...legs.map((leg) => leg.label), "condition / measured result"];
  const classes = ["cyan", ...legs.map(() => ""), "reason"];
  byId("strategies").replaceChildren(table(headers, rows, classes));
}

function humanIndicator(name, value) {
  const number = finite(value);
  if (number === null) return "—";
  if (name === "rsi_14") return number >= 70 ? "overbought" : number <= 30 ? "oversold" : "neutral";
  if (name === "adx_14") return number >= 25 ? "strong trend" : "weak trend";
  if (name === "bb_pct_b") return number > 1 ? "above upper band" : number < 0 ? "below lower band" : "inside bands";
  if (name.includes("entropy") || name.includes("ratio") || name.includes("autocorr")) return number >= 0 ? "positive" : "negative";
  return number >= 0 ? "positive" : "negative";
}

function indicatorLabel(name) {
  return String(name).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderIndicators(payload) {
  const indicators = spotLeg(payload)?.market?.indicators || {};
  const items = Object.entries(indicators).map(([name, value]) => {
    const item = node("div", "indicator");
    item.append(node("label", "", indicatorLabel(name)));
    item.append(node("strong", finite(value) >= 0 ? "up" : "down", fmt(value, 4)));
    item.append(node("small", "muted", humanIndicator(name, value)));
    return item;
  });
  byId("indicators").replaceChildren(...items);
  if (!items.length) byId("indicators").append(node("div", "empty", "INDICATORS ARE WARMING"));
}

function renderForward(payload) {
  const rows = [];
  legList(payload).forEach((leg) => {
    const horizons = leg.market?.research?.horizons || {};
    Object.entries(horizons).sort(([a], [b]) => Number(a) - Number(b)).forEach(([horizon, stat]) => {
      const scored = stat.scored || 0;
      rows.push([
        leg.label,
        `+${horizon}`,
        scored,
        `${stat.hits || 0}/${stat.misses ?? Math.max(scored - (stat.hits || 0), 0)}`,
        scored ? percent(stat.hit_rate, 1) : "—",
        scored ? `${fmt(stat.mean_abs_error_bps, 2)}bp` : "—",
      ]);
    });
  });
  byId("forward-scores").replaceChildren(table(["leg", "bar", "scored", "hit/miss", "hit rate", "MAE"], rows));
}

function systemItem(label, value, tone = "") {
  const item = node("div", "system-item");
  item.append(node("label", "", label));
  item.append(node("strong", tone, value));
  return item;
}

function renderSystem(payload) {
  const legs = legList(payload);
  const spot = spotLeg(payload)?.market || {};
  const research = spot.research || {};
  const projectionPending = legs.reduce((total, leg) => total + Number(leg.market?.research?.pending || 0), 0);
  const missing = legs.reduce((total, leg) => total + Number(leg.market?.research?.missing || 0), 0);
  const aiPending = legs.reduce((total, leg) => total + Number(leg.market?.research?.ai?.pending || 0), 0);
  const expired = legs.reduce((total, leg) => total + Number(leg.market?.research?.ai?.expired || 0), 0);
  const items = [
    systemItem("data source", String(spot.source || "—").toUpperCase(), "cyan"),
    systemItem("latest tick", age(spot.last_tick_ts), age(spot.last_tick_ts).endsWith("s ago") ? "up" : "hold"),
    systemItem("accepted / rejected", `${research.ticks || 0} / ${research.rejected_ticks || 0}`),
    systemItem("compute / workers", `${fmt(research.compute_ms, 1)}ms / ${research.workers || 1}`),
    systemItem("projection pending", `${projectionPending} · missing ${missing}`, missing ? "hold" : "up"),
    systemItem("AI pending / expired", `${aiPending} / ${expired}`, expired ? "hold" : "prediction"),
    systemItem("spread", `${fmt(research.spread, 3)} per unit`),
    systemItem("cost assumptions", `₹${fmt(research.fixed_cost_rupees, 0)}/lot + ${fmt(research.variable_cost_bps, 1)}bp`),
  ];
  byId("system-state").replaceChildren(...items);
}

function render(payload) {
  latest = payload;
  renderOverview(payload);
  renderLegs(payload);
  renderAi(payload);
  renderPredictions(payload);
  renderStrategies(payload);
  renderIndicators(payload);
  renderForward(payload);
  renderSystem(payload);
  byId("loading").classList.add("hidden");
  byId("last-update").textContent = `SNAPSHOT ${payload.updated_at ? new Date(payload.updated_at).toLocaleTimeString("en-IN", { hour12: false }) : "—"} · ${payload.ts || "—"}`;
}

function connection(mode, text) {
  byId("connection-dot").className = `dot ${mode}`;
  byId("connection-text").textContent = text;
}

async function poll() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (payload.ready) {
      render(payload);
      connection("live", "LIVE / POLLING");
    } else {
      connection("waiting", "WARMING STATE");
    }
  } catch (error) {
    connection("error", "CONNECTION LOST");
  } finally {
    window.setTimeout(poll, refreshMs);
  }
}

byId("strategy-filter").addEventListener("input", (event) => {
  strategyQuery = event.target.value;
  if (latest) renderStrategies(latest);
});

window.addEventListener("resize", () => {
  charts.forEach((market, canvas) => drawChart(canvas, market));
});

window.setInterval(() => {
  byId("clock").textContent = new Date().toLocaleTimeString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" }) + " IST";
}, 1000);

poll();
