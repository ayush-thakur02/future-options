"use strict";

const refreshMs = Math.max(Number(document.documentElement.dataset.refreshMs) || 1000, 250);
const byId = (id) => document.getElementById(id);
const charts = new Map();
const SVG_NS = "http://www.w3.org/2000/svg";
let latest = null;
let strategyQuery = "";
let resizeFrame = null;

function node(tag, className = "", value = "") {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (value !== "") item.textContent = String(value);
  return item;
}

function svgNode(tag, attributes = {}, value = "") {
  const item = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([name, content]) => item.setAttribute(name, String(content)));
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

function probabilityLabel(leg, compact = false) {
  if (leg.kind === "CE" || leg.kind === "PE") return compact ? "P(PREM ↑)" : "P(PREMIUM UP)";
  return compact ? "P(INDEX ↑)" : "P(INDEX UP)";
}

function legDirectionContext(leg) {
  if (leg.kind === "CE") return "CALL PREMIUM ↑ ≈ INDEX ↑";
  if (leg.kind === "PE") return "PUT PREMIUM ↑ ≈ INDEX ↓";
  return "INDEX PRICE DIRECTION";
}

function actionMeaning(leg, action) {
  if (action === "BUY") {
    if (leg.kind === "CE") return "LONG CALL · BULLISH INDEX";
    if (leg.kind === "PE") return "LONG PUT · BEARISH INDEX";
    return "LONG INDEX · BULLISH";
  }
  if (action === "SELL") {
    if (leg.kind === "CE") return "EXIT / SHORT CALL · BEARISH INDEX";
    if (leg.kind === "PE") return "EXIT / SHORT PUT · BULLISH INDEX";
    return "SHORT INDEX · BEARISH";
  }
  return leg.kind === "CE" || leg.kind === "PE" ? `NO ${leg.label} TRADE` : "NO INDEX TRADE";
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
  const today = legList(payload).reduce((total, leg) => {
    const summary = leg.market?.research?.ai?.today || {};
    total.samples += Number(summary.samples || 0);
    total.hits += Number(summary.hits || 0);
    if (!total.date && summary.date) total.date = summary.date;
    return total;
  }, { samples: 0, hits: 0, date: "" });
  const todayAccuracy = today.samples ? today.hits / today.samples : null;
  const items = [
    metric("instrument", payload.symbol || market.symbol || "—", `source ${market.source || "—"}`, "cyan"),
    metric("spot / last", fmt(payload.spot ?? market.last_price), `${signed(market.change)} · ${signed(market.change_pct, 2, "%")}`, change >= 0 ? "up" : "down"),
    metric("market regime", String(market.regime || "unknown").toUpperCase(), `conviction ${signed(market.conviction, 2)}`),
    metric("option chain", chain.expiry || "—", `PCR ${fmt(chain.pcr)} · ATM ${fmt(chain.atm_strike, 0)}`),
    metric("feed state", String(payload.status || "waiting").toUpperCase(), `tick ${age(market.last_tick_ts)}`, age(market.last_tick_ts).startsWith("0") ? "up" : ""),
    metric("research state", market.research?.ai ? "AUTO-LEARNING" : "WARMING", `${market.research?.workers ?? 1} CPU · ${fmt(market.research?.compute_ms, 1)}ms/bar`, "prediction"),
    metric(
      "today's accuracy",
      today.samples ? percent(todayAccuracy, 1) : "WARMING",
      today.samples ? `${today.hits}/${today.samples} matured AI outcomes · all legs` : "no AI outcomes matured today",
      today.samples ? (todayAccuracy >= 0.5 ? "up" : "down") : "hold",
    ),
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

function projectionFor(market, horizon = 1) {
  return (market?.projections || []).find((item) => Number(item.horizon) === Number(horizon)) || null;
}

function timeOnly(timestamp) {
  if (!timestamp) return "—";
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) return String(timestamp).slice(11, 16) || "—";
  return parsed.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Kolkata",
  });
}

function projectionItem(projected) {
  const item = node("div", "projection");
  item.append(node("span", "", `+${projected.horizon ?? "?"} · ${timeOnly(projected.ts)} IST`));
  item.append(node("strong", "", `O ${fmt(projected.open)} → C ${fmt(projected.close)}`));
  item.append(node("small", "", `H ${fmt(projected.high)} · L ${fmt(projected.low)} · ${signed(projected.expected_move_bps ?? projected.change_bps, 1, "bp")} · conf ${percent(projected.confidence)}`));
  return item;
}

function decisionStat(label, value, tone = "") {
  const item = node("div", "decision-stat");
  item.append(node("label", "", label));
  item.append(node("strong", tone, value));
  return item;
}

function evaluateDecision(leg) {
  const market = leg.market || {};
  const ai = signalFor(market, 1);
  const projected = projectionFor(market, 1);
  const policy = market.research?.ai?.policy || {};
  const verdict = leg.verdict;
  const candidate = ai?.action || "HOLD";
  const cost = finite(policy.cost_bps) || 0;
  const expectedMove = Math.abs(finite(projected?.expected_move_bps) || 0);
  const isOption = leg.kind === "CE" || leg.kind === "PE";
  const verdictAction = verdict?.action || null;
  const premiumShort = isOption && verdictAction === "SHORT";
  const verdictPass = verdict
    ? Boolean(verdict.is_trade) && (premiumShort || (finite(verdict.edge_bps) || 0) > 0)
    : expectedMove > cost;
  const directionAgrees = !verdict || (
    (candidate === "BUY" && verdictAction === "LONG")
    || (candidate === "SELL" && verdictAction === "SHORT")
  );

  let action = candidate;
  let reason = "The realtime learners are still warming; wait for a measured signal.";
  if (!ai) {
    action = "HOLD";
  } else if (candidate === "HOLD") {
    action = "HOLD";
    if ((finite(ai.trust_score) || 0) < (finite(policy.min_trust_for_action) || 0)) {
      reason = `HOLD: trust ${percent(ai.trust_score)} is below the ${percent(policy.min_trust_for_action)} action floor.`;
    } else {
      reason = `HOLD: ${probabilityLabel(leg)} ${fmt(ai.p_up, 3)} is inside the neutral ${fmt(policy.sell_probability, 2)}–${fmt(policy.buy_probability, 2)} zone.`;
    }
  } else if (!verdictPass) {
    action = "HOLD";
    reason = verdict
      ? `HOLD: ${verdict.reason}; the projected move has no positive breakeven edge.`
      : `HOLD: expected ${fmt(expectedMove, 1)}bp does not clear estimated ${fmt(cost, 1)}bp costs.`;
  } else if (!directionAgrees) {
    action = "HOLD";
    reason = `HOLD: AI says ${candidate}, but the cost/breakeven verdict says ${verdictAction}; wait for agreement.`;
  } else {
    const verb = candidate === "BUY" ? "BUY" : isOption ? "SELL / EXIT" : "SELL";
    reason = `${verb} setup (${actionMeaning(leg, candidate)}): realtime AI and the cost-gated ${verdictAction || "move"} verdict agree. Re-check on every one-second update.`;
  }

  const edge = verdict
    ? premiumShort ? null : finite(verdict.edge_bps)
    : expectedMove - cost;
  const invalidation = !projected
    ? "—"
    : action === "BUY"
      ? `< ${fmt(projected.low)}`
      : action === "SELL"
        ? `> ${fmt(projected.high)}`
        : `${fmt(projected.low)}–${fmt(projected.high)}`;
  return {
    action,
    ai,
    projected,
    policy,
    verdict,
    edge,
    edgeText: premiumShort ? "IV-RICH PASS" : signed(edge, 1, "bp"),
    invalidation,
    reason,
  };
}

function renderDecisions(payload) {
  const root = byId("decision-grid");
  const cards = legList(payload).map((leg) => {
    const result = evaluateDecision(leg);
    const { ai, projected } = result;
    const actionTone = directionClass(result.action);
    const card = node("article", `decision-card action-${result.action.toLowerCase()}`);
    const head = node("div", "decision-head");
    const identity = node("div");
    identity.append(node("div", "decision-instrument", leg.strike ? `${leg.label} ${fmt(leg.strike, 0)}` : leg.label));
    identity.append(node("div", "decision-symbol", leg.market?.symbol || "—"));
    const actionBadge = node("div", `decision-action ${actionTone}`);
    actionBadge.append(
      node("strong", "", result.action),
      node("small", "", actionMeaning(leg, result.action)),
    );
    actionBadge.title = `${legDirectionContext(leg)}. Actions refer to the instrument or premium shown.`;
    head.append(identity, actionBadge);

    const stats = node("div", "decision-stats");
    stats.append(
      decisionStat(probabilityLabel(leg), fmt(ai?.p_up, 3)),
      decisionStat("TRUST", percent(ai?.trust_score)),
      decisionStat("EDGE / GATE", result.edgeText, result.edgeText === "IV-RICH PASS" || finite(result.edge) > 0 ? "up" : "hold"),
      decisionStat("TARGET / INVALID", `${fmt(projected?.close)} / ${result.invalidation}`),
    );

    card.append(head, node("div", "decision-context", legDirectionContext(leg)), node("div", "decision-summary", result.reason), stats);
    return card;
  });
  root.replaceChildren(...cards);
  if (!cards.length) root.append(node("div", "empty", "DECISION STATE IS WARMING"));
}

function renderLegs(payload) {
  const root = byId("leg-grid");
  if (chartObserver) chartObserver.disconnect();
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
    const chart = svgNode("svg", {
      class: "market-chart",
      role: "img",
      "aria-label": `${leg.label} actual and projected candlestick chart`,
      preserveAspectRatio: "none",
    });
    wrap.append(chart);

    const reads = node("div", "leg-readouts");
    reads.append(readout("AUTO-AI +1", ai ? `${ai.action} · ${probabilityLabel(leg, true)} ${fmt(ai.p_up, 2)}` : "WARMING", ai ? directionClass(ai.action) : "muted"));
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
    charts.set(chart, market);
    if (chartObserver) chartObserver.observe(chart);
    window.requestAnimationFrame(() => drawChart(chart, market));
    card.style.setProperty("--leg-index", index);
    return card;
  });
  root.replaceChildren(...cards);
  if (!cards.length) root.append(node("div", "empty", "NO INSTRUMENT SNAPSHOT YET"));
}

function drawChart(chart, market) {
  const width = Math.max(Math.round(chart.clientWidth), 240);
  const height = Math.max(Math.round(chart.clientHeight), 180);
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  chart.replaceChildren();
  const actual = (market.candles || []).filter(validCandle);
  const projected = (market.projections || []).filter(validCandle);
  const usableActual = actual.slice(-Math.max(Math.floor(width / 7) - projected.length, 24));
  const all = [...usableActual, ...projected];
  if (!all.length) {
    chart.append(svgNode("text", { x: 14, y: 28, fill: "#64748b", "font-size": 10 }, "AWAITING CANDLES"));
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

  for (let line = 0; line <= 4; line += 1) {
    const lineY = top + (plotHeight * line) / 4;
    const price = high - (range * line) / 4;
    chart.append(svgNode("line", {
      x1: left,
      y1: lineY,
      x2: left + plotWidth,
      y2: lineY,
      stroke: "#e2e8f0",
      "vector-effect": "non-scaling-stroke",
    }));
    chart.append(svgNode("text", {
      x: left + plotWidth + 5,
      y: lineY + 3,
      fill: "#64748b",
      "font-size": 9,
    }, price.toLocaleString("en-IN", { maximumFractionDigits: 2 })));
  }

  if (projected.length && usableActual.length) {
    const separator = left + usableActual.length * step;
    chart.append(svgNode("rect", {
      x: separator,
      y: top,
      width: Math.max(left + plotWidth - separator, 0),
      height: plotHeight,
      fill: "#2457d6",
      "fill-opacity": 0.045,
    }));
    chart.append(svgNode("line", {
      x1: separator,
      y1: top,
      x2: separator,
      y2: top + plotHeight,
      stroke: "#2457d6",
      "stroke-opacity": 0.55,
      "vector-effect": "non-scaling-stroke",
    }));
    chart.append(svgNode("text", {
      x: Math.min(separator + 4, width - right - 24),
      y: top + 10,
      fill: "#1d4ed8",
      "font-size": 9,
      "font-weight": 700,
    }, "PROJECTED"));

  }

  all.forEach((candle, index) => {
    const isProjection = index >= usableActual.length;
    const rising = Number(candle.close) >= Number(candle.open);
    const color = isProjection ? "#2457d6" : rising ? "#087a4f" : "#c4324a";
    const center = left + (index + 0.5) * step;
    const openY = y(candle.open);
    const closeY = y(candle.close);
    const opacity = isProjection ? Math.max(0.5, 1 - (index - usableActual.length) * 0.14) : 0.9;
    const group = svgNode("g", { opacity });
    const title = svgNode("title", {}, `${isProjection ? `PROJECTED +${candle.horizon}` : "PRINTED"} · O ${fmt(candle.open)} · H ${fmt(candle.high)} · L ${fmt(candle.low)} · C ${fmt(candle.close)}${isProjection ? ` · confidence ${percent(candle.confidence)}` : ""}`);
    group.append(title);
    group.append(svgNode("line", {
      x1: center,
      y1: y(candle.high),
      x2: center,
      y2: y(candle.low),
      stroke: color,
      "stroke-width": isProjection ? 1.6 : 1,
      "vector-effect": "non-scaling-stroke",
    }));
    const bodyTop = Math.min(openY, closeY);
    const bodyHeight = Math.max(Math.abs(closeY - openY), 1);
    group.append(svgNode("rect", {
      x: center - bodyWidth / 2,
      y: bodyTop,
      width: bodyWidth,
      height: bodyHeight,
      fill: rising || isProjection ? "#ffffff" : color,
      stroke: color,
      "stroke-width": isProjection ? 1.8 : 1,
      "vector-effect": "non-scaling-stroke",
    }));
    if (isProjection) {
      group.append(svgNode("circle", { cx: center, cy: closeY, r: 2.2, fill: color }));
      group.append(svgNode("text", {
        x: center,
        y: top + plotHeight + 13,
        fill: color,
        "font-size": 9,
        "font-weight": 700,
        "text-anchor": "middle",
      }, `+${candle.horizon}`));
    }
    chart.append(group);
  });
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

function calculationMetric(label, value, tone = "") {
  const item = node("div", "calculation-metric");
  item.append(node("label", "", label), node("strong", tone, value));
  return item;
}

function calculationSection(title, content, className = "") {
  const section = node("section", "calculation-section");
  section.append(node("h3", "", title));
  if (className) section.append(node("pre", className, content));
  else section.append(content);
  return section;
}

function showCalculation({ kicker, title, lead, metrics, formula, details }) {
  byId("calculation-kicker").textContent = kicker;
  byId("calculation-title").textContent = title;
  const content = byId("calculation-content");
  const parts = [node("p", "calculation-lead", lead)];
  if (metrics?.length) {
    const metricGrid = node("div", "calculation-metrics");
    metrics.forEach(([label, value, tone]) => metricGrid.append(calculationMetric(label, value, tone)));
    parts.push(metricGrid);
  }
  if (formula) parts.push(calculationSection("FORMULA USED", formula, "calculation-formula"));
  if (details?.length) {
    const list = node("ul", "calculation-list");
    details.forEach((detail) => list.append(node("li", "", detail)));
    parts.push(calculationSection("LIVE INPUT / INTERPRETATION", list));
  }
  content.replaceChildren(...parts);
  const modal = byId("calculation-modal");
  if (!modal.open) modal.showModal();
}

function algorithmFormula(name) {
  const formulas = {
    online_logistic: "margin = bias + Σ(weightᵢ × causal_z(featureᵢ))\nP(up) = sigmoid(margin)\nweights ← L2-shrunk SGD update after the target bar matures",
    passive_aggressive: "margin = bias + Σ(weightᵢ × causal_z(featureᵢ))\nP(up) = sigmoid(margin / max(1, ||weights||))\nupdate only when max(0, 1 − target × margin) > 0",
    gaussian_nb: "log_score(class) = log(class prior) + Σ Gaussian log-likelihood(featureᵢ | class)\nP(up) = sigmoid(log_score(up) − log_score(down))",
    ftrl_proximal: "weightᵢ = proximal(zᵢ, nᵢ, α, β, L1, L2)\nmargin = bias + Σ(weightᵢ × causal_z(featureᵢ))\nP(up) = sigmoid(margin)",
    adaptive_knn: "distance = RMS(query_z − neighbour_z)\nweight = exp(−ln(2) × age / half_life) / max(distance, 0.05)\nP(up) = weighted neighbour outcomes with Beta prior",
    strategy_combinations: "expert = active signed strategy singles/pairs/triples\nP(up | expert) = (decayed_up + prior/2) / (support + prior)\nP(up) = support-weighted mean of eligible experts",
  };
  return formulas[name] || "P(up) = online learner probability from causally available features\nparameters update only after the exact target bar closes";
}

function showAiCalculation(card, market) {
  const oneBar = signalFor(market, 1);
  const live = (oneBar?.algorithms || []).find((item) => item.name === card.algorithm);
  showCalculation({
    kicker: "REALTIME AI CALCULATION",
    title: card.algorithm,
    lead: "This learner predicts before the target closes, is scored at the exact target, and only then updates. The figures below are the current immutable prediction and matured scorecard.",
    metrics: [
      ["LIVE P(UP)", fmt(live?.p_up, 4), directionClass(live?.action)],
      ["ACTION", live?.action || "WARMING", directionClass(live?.action)],
      ["HIT / MISS", `${card.hits || 0} / ${card.misses || 0}`, ""],
      ["TRUST", percent(card.trust_score), "cyan"],
    ],
    formula: algorithmFormula(card.algorithm),
    details: [
      `Accuracy ${percent(card.accuracy, 1)} from ${card.samples || 0} matured predictions; rolling accuracy ${percent(card.rolling_accuracy, 1)}.`,
      `Net hypothetical result ${signed(card.net_pnl_bps, 1, "bp")}; maximum drawdown ${signed(card.drawdown_bps, 1, "bp")}.`,
      "Trust is evidence-gated and combines conservative accuracy skill, Brier skill, calibration quality, and a positive post-cost result. It is not the same as probability.",
    ],
  });
}

function showStrategyCalculation(signal) {
  const meta = signal.meta || {};
  const raw = finite(meta.raw_score) || 0;
  const threshold = finite(meta.active_threshold) || 0.15;
  const samples = Number(meta.scored || 0);
  const hits = Number(meta.hits || 0);
  showCalculation({
    kicker: `${String(meta.category || "STRATEGY").toUpperCase()} PLUGIN CALCULATION`,
    title: signal.strategy,
    lead: meta.description || signal.reason || "Plugin strategy evaluated on the latest causally available feature row.",
    metrics: [
      ["SIGNED SCORE", signed(raw, 4), raw > 0 ? "up" : raw < 0 ? "down" : "hold"],
      ["CONFIDENCE", percent(Math.abs(raw)), "prediction"],
      ["STATE", meta.state || "WAIT", directionClass(meta.state === "ACTIVE" ? signal.direction : "HOLD")],
      ["TRUST", percent(meta.trust_score), "cyan"],
    ],
    formula: `raw_score = plugin.score(latest bars, latest indicators)\nconfidence = |raw_score| = ${Math.abs(raw).toFixed(4)}\ndirection = sign(raw_score)\nstate = ACTIVE when |raw_score| ≥ ${threshold.toFixed(2)}, otherwise WAIT\ncurrent: |${raw.toFixed(4)}| ${Math.abs(raw) >= threshold ? "≥" : "<"} ${threshold.toFixed(2)} → ${meta.state || "WAIT"}`,
    details: [
      signal.reason || "No live condition detail is available.",
      samples ? `Measured outcomes: ${hits} hits / ${Math.max(samples - hits, 0)} misses; accuracy ${percent(hits / samples, 1)}; net ${signed(meta.net_pnl_bps, 1, "bp")}.` : "No exact-target outcome has matured for this strategy yet.",
      `Trust = evidence(min(1, n/${meta.trust_min_samples || 50})) × [75% conservative accuracy skill + 25% positive post-cost quality].`,
    ],
  });
}

function showPredictionCalculation(leg, horizon, item, projected, stats) {
  const members = item.algorithms || [];
  const weights = members.map((member) => Math.max(finite(member.trust_score) || 0, 0.05));
  const numerator = members.reduce((sum, member, index) => sum + (finite(member.p_up) || 0.5) * weights[index], 0);
  const denominator = weights.reduce((sum, value) => sum + value, 0);
  const policy = leg.market?.research?.ai?.policy || {};
  const move = Math.abs(finite(projected?.expected_move_bps) || 0);
  const cost = finite(policy.cost_bps) || 0;
  showCalculation({
    kicker: `${leg.label} +${horizon} CONSENSUS`,
    title: `${item.action} · ${probabilityLabel(leg)} ${fmt(item.p_up, 4)}`,
    lead: `The cell predicts ${leg.kind === "CE" || leg.kind === "PE" ? "this option premium" : "the index"}, combines all online learners with a minimum 5% vote weight, applies the configured probability/trust policy, then compares the projected move with estimated round-trip cost. ${actionMeaning(leg, item.action)}.`,
    metrics: [
      ["TARGET", `${timeOnly(item.target_at)} IST`, ""],
      ["CONFIDENCE", percent(item.confidence), "prediction"],
      ["TRUST", percent(item.trust_score), "cyan"],
      ["MOVE − COST", signed(move - cost, 2, "bp"), move > cost ? "up" : "hold"],
    ],
    formula: `direction basis = ${legDirectionContext(leg)}\nweightᵢ = max(learner_trustᵢ, 0.05)\nP(up) = Σ(Pᵢ × weightᵢ) / Σ(weightᵢ)\n      = ${numerator.toFixed(4)} / ${denominator.toFixed(4)} = ${denominator ? (numerator / denominator).toFixed(4) : "0.5000"}\nensemble_trust = mean(learner trust) = ${fmt(item.trust_score, 4)}\nBUY if P(up) ≥ ${fmt(policy.buy_probability, 2)} and trust ≥ ${fmt(policy.min_trust_for_action, 2)}\nSELL if P(up) ≤ ${fmt(policy.sell_probability, 2)} and trust ≥ ${fmt(policy.min_trust_for_action, 2)}\npost_cost_edge = |${fmt(projected?.expected_move_bps, 2)}| − ${fmt(cost, 2)} = ${signed(move - cost, 2, "bp")}`,
    details: [
      ...members.map((member) => `${member.name}: P(up) ${fmt(member.p_up, 4)}, action ${member.action}, trust ${percent(member.trust_score)}.`),
      projected ? `Projected OHLC: ${fmt(projected.open)} / ${fmt(projected.high)} / ${fmt(projected.low)} / ${fmt(projected.close)} at confidence ${percent(projected.confidence)}.` : "The projected candle is still warming.",
      stats.scored ? `Projected-candle history: ${stats.hits || 0}/${stats.scored} directional hits (${percent(stats.hit_rate, 1)}), MAE ${fmt(stats.mean_abs_error_bps, 2)}bp.` : "No projected candle at this horizon has matured yet.",
    ],
  });
}

function renderAi(payload) {
  const market = spotLeg(payload)?.market || {};
  const cards = market.research?.ai?.scorecards || [];
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
  const scoreTable = table(
    ["algorithm", "n", "hit/miss", "accuracy", "rolling", "net", "drawdown", "trust"],
    rows,
    ["cyan"],
  );
  if (scoreTable.tagName === "TABLE") {
    scoreTable.querySelectorAll("tbody tr").forEach((row, index) => {
      row.classList.add("inspectable");
      row.title = "Open live AI calculation";
      row.addEventListener("click", () => showAiCalculation(cards[index], market));
    });
  }
  byId("ai-scorecards").replaceChildren(scoreTable);
}

function probabilityBar(value) {
  const bar = node("span", "probability-bar");
  const fill = node("i");
  fill.style.width = `${Math.max(0, Math.min((finite(value) || 0.5) * 100, 100))}%`;
  bar.append(fill);
  return bar;
}

function predictionCell(leg, horizon) {
  const market = leg.market || {};
  const item = signalFor(market, horizon);
  const projected = projectionFor(market, horizon);
  const policy = market.research?.ai?.policy || {};
  const stats = market.research?.horizons?.[horizon]
    || market.research?.horizons?.[String(horizon)]
    || {};
  const cell = node("div", "prediction-cell");
  if (!item) {
    cell.append(node("div", "empty", "AI HORIZON WARMING"));
    return cell;
  }
  cell.classList.add("inspectable");
  cell.title = "Open prediction calculation";
  cell.addEventListener("click", () => showPredictionCalculation(leg, horizon, item, projected, stats));

  const head = node("div", "prediction-cell-head");
  head.append(tag(item.action), node("time", "", `${timeOnly(item.target_at)} IST`));
  const probability = node("div", "prediction-prob");
  probability.append(node("span", "", probabilityLabel(leg, true)), probabilityBar(item.p_up), node("b", directionClass(item.action), fmt(item.p_up, 2)));

  const detail = node("div", "prediction-detail");
  detail.append(
    node("span", "", `TRUST ${percent(item.trust_score)}`),
    node("span", "", projected ? `C ${fmt(projected.close)} · ${signed(projected.expected_move_bps, 1, "bp")}` : "PATH —"),
  );
  const expected = Math.abs(finite(projected?.expected_move_bps) || 0);
  const cost = finite(policy.cost_bps) || 0;
  let gateText = "HOLD · AI NEUTRAL";
  let gateTone = "hold";
  if (item.action !== "HOLD" && !projected) {
    gateText = "HOLD · PATH WARMING";
  } else if (item.action !== "HOLD" && expected > cost) {
    gateText = `POST-COST CANDIDATE · +${fmt(expected - cost, 1)}bp`;
    gateTone = directionClass(item.action);
  } else if (item.action !== "HOLD") {
    gateText = `HOLD · ${fmt(expected, 1)}bp < ${fmt(cost, 1)}bp COST`;
  }
  if (stats.scored) gateText += ` · HIT ${percent(stats.hit_rate)}`;

  cell.append(head, probability, detail, node("div", `prediction-gate ${gateTone}`, gateText));
  return cell;
}

function renderPredictions(payload) {
  const root = byId("live-predictions");
  const legs = legList(payload);
  const horizonSet = new Set();
  legs.forEach((leg) => {
    Object.keys(leg.market?.research?.ai?.signals || {}).forEach((value) => horizonSet.add(Number(value)));
    (leg.market?.projections || []).forEach((item) => horizonSet.add(Number(item.horizon)));
  });
  const horizons = [...horizonSet].filter(Number.isFinite).sort((a, b) => a - b);
  if (!horizons.length) horizons.push(1, 2, 3);

  const rows = [];
  const header = node("div", "prediction-row matrix-head");
  header.style.setProperty("--horizons", horizons.length);
  header.append(node("div", "prediction-leg", "INSTRUMENT"));
  horizons.forEach((horizon) => header.append(node("div", "prediction-cell", `+${horizon} TARGET BAR`)));
  rows.push(header);

  legs.forEach((leg) => {
    const row = node("div", "prediction-row");
    row.style.setProperty("--horizons", horizons.length);
    const label = node("div", "prediction-leg", leg.strike ? `${leg.label} ${fmt(leg.strike, 0)}` : leg.label);
    label.append(node("small", "", leg.market?.symbol || "—"));
    row.append(label);
    horizons.forEach((horizon) => row.append(predictionCell(leg, horizon)));
    rows.push(row);
  });
  root.replaceChildren(...rows);
}

function renderStrategies(payload) {
  const legs = legList(payload);
  const signalMap = new Map();
  legs.forEach((leg) => {
    (leg.market?.signals || []).forEach((signal) => {
      if (!signalMap.has(signal.strategy) || leg.label === "INDEX") signalMap.set(signal.strategy, signal);
    });
  });
  const signals = [...signalMap.values()];
  const normalized = strategyQuery.trim().toLowerCase();
  const filtered = signals.filter((signal) => !normalized || signal.strategy.toLowerCase().includes(normalized));
  const summaryRows = legs.map((leg) => {
    const legSignals = leg.market?.signals || [];
    const counts = { UP: 0, DOWN: 0, HOLD: 0 };
    legSignals.forEach((signal) => {
      const active = signal.meta?.state === "ACTIVE";
      const bucket = active && signal.direction === "UP"
        ? "UP"
        : active && signal.direction === "DOWN"
          ? "DOWN"
          : "HOLD";
      counts[bucket] += 1;
    });
    const sampleCount = legSignals.length;
    const divisor = sampleCount || 1;
    const averageConfidence = legSignals.reduce((sum, signal) => sum + (finite(signal.strength) || 0), 0) / divisor;
    const averageTrust = legSignals.reduce((sum, signal) => sum + (finite(signal.meta?.trust_score) || 0), 0) / divisor;
    const netView = (counts.UP - counts.DOWN) / divisor;
    const directionLabel = leg.kind === "CE" || leg.kind === "PE" ? "PREMIUM" : "INDEX";
    const values = [
      ["LEG / DIRECTION", legDirectionContext(leg), "cyan"],
      [`${directionLabel} UP`, `${percent(counts.UP / divisor)} · ${counts.UP}/${sampleCount}`, "up"],
      [`${directionLabel} DOWN`, `${percent(counts.DOWN / divisor)} · ${counts.DOWN}/${sampleCount}`, "down"],
      ["HOLD / WAIT", `${percent(counts.HOLD / divisor)} · ${counts.HOLD}/${sampleCount}`, "hold"],
      ["AVG CONFIDENCE", percent(averageConfidence), "prediction"],
      ["AVG TRUST", percent(averageTrust), "cyan"],
      ["NET BREADTH", signed(netView, 2), netView > 0 ? "up" : netView < 0 ? "down" : "hold"],
    ];
    const row = node("div", "strategy-summary-row");
    row.title = `${legDirectionContext(leg)}. UP and DOWN refer to the displayed instrument's price.`;
    values.forEach(([label, value, tone]) => {
      const item = node("div", "strategy-summary-item");
      item.append(node("label", "", label), node("strong", tone, value));
      row.append(item);
    });
    return row;
  });
  byId("strategy-summary").replaceChildren(...summaryRows);

  const rows = filtered.map((signal) => {
    const states = legs.map((leg) => {
      const match = (leg.market?.signals || []).find((candidate) => candidate.strategy === signal.strategy);
      const state = match?.meta?.state || "WAIT";
      return state === "ACTIVE" ? tag(match.direction) : tag(state);
    });
    const meta = signal.meta || {};
    return [
      signal.strategy,
      ...states,
      percent(signal.strength),
      percent(meta.trust_score),
      meta.scored ? `${meta.hits || 0}/${Math.max((meta.scored || 0) - (meta.hits || 0), 0)}` : "—",
      meta.scored ? signed(meta.net_pnl_bps, 1, "bp") : "—",
    ];
  });
  const headers = ["strategy", ...legs.map((leg) => leg.label), "confidence", "trust", "hit/miss", "net"];
  const classes = ["cyan", ...legs.map(() => "")];
  const strategyTable = table(headers, rows, classes);
  if (strategyTable.tagName === "TABLE") {
    strategyTable.querySelectorAll("tbody tr").forEach((row, index) => {
      row.classList.add("inspectable");
      row.title = "Open strategy calculation";
      row.addEventListener("click", () => showStrategyCalculation(filtered[index]));
    });
  }
  byId("strategies").replaceChildren(strategyTable);
}

function humanIndicator(name, value, price) {
  const number = finite(value);
  if (number === null) return "—";
  if (/^ema_\d+$/.test(name)) return finite(price) >= number ? "price above EMA" : "price below EMA";
  if (name === "vwap") return finite(price) >= number ? "price above VWAP" : "price below VWAP";
  if (name === "supertrend") return finite(price) >= number ? "above supertrend" : "below supertrend";
  if (name.includes("ema_") && name.includes("spread")) return number >= 0 ? "bullish EMA stack" : "bearish EMA stack";
  if (name === "supertrend_dir") return number > 0 ? "bullish" : number < 0 ? "bearish" : "flat";
  if (name.startsWith("roc_")) return number >= 0 ? "positive momentum" : "negative momentum";
  if (name === "rsi_14") return number >= 70 ? "overbought" : number <= 30 ? "oversold" : "neutral";
  if (name === "adx_14") return number >= 25 ? "strong trend" : "weak trend";
  if (name === "macd_hist" || name === "ppo_hist") return number >= 0 ? "bullish momentum" : "bearish momentum";
  if (name === "stoch_k" || name === "stoch_d") return number >= 80 ? "overbought" : number <= 20 ? "oversold" : "neutral";
  if (name === "mfi_14") return number >= 80 ? "money flow high" : number <= 20 ? "money flow low" : "balanced flow";
  if (name === "williams_r") return number >= -20 ? "overbought" : number <= -80 ? "oversold" : "neutral";
  if (name === "bb_pct_b") return number > 1 ? "above upper band" : number < 0 ? "below lower band" : "inside bands";
  if (name === "choppiness_14") return number >= 61.8 ? "range-bound" : number <= 38.2 ? "trending" : "mixed";
  if (name === "efficiency_ratio_10") return number >= 0.4 ? "directional" : "noisy";
  if (name === "hurst_100") return number > 0.55 ? "persistent" : number < 0.45 ? "mean-reverting" : "random-like";
  if (name.includes("entropy") || name.includes("ratio") || name.includes("autocorr")) return number >= 0 ? "positive" : "negative";
  return number >= 0 ? "positive" : "negative";
}

function indicatorLabel(name) {
  return String(name).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function indicatorValue(name, value) {
  const number = finite(value);
  if (number === null) return "—";
  if (/^(ema_\d+|supertrend|vwap)$/.test(name)) return fmt(number, 2);
  if (name.endsWith("_dir") || name.endsWith("_cross")) return fmt(number, 0);
  if (name.includes("_dist") || name.includes("_spread") || name.startsWith("roc_") || ["atr_norm", "macd_line", "macd_signal", "macd_hist", "cmf_20"].includes(name)) {
    return signed(number * 100, 3, "%");
  }
  return fmt(number, 3);
}

function renderIndicators(payload) {
  const market = spotLeg(payload)?.market || {};
  const indicators = market.indicators || {};
  const priority = ["ema_9", "ema_21", "ema_50", "ema_200", "vwap", "supertrend"];
  const entries = Object.entries(indicators).sort(([left], [right]) => {
    const leftIndex = priority.indexOf(left);
    const rightIndex = priority.indexOf(right);
    if (leftIndex >= 0 || rightIndex >= 0) return (leftIndex < 0 ? priority.length : leftIndex) - (rightIndex < 0 ? priority.length : rightIndex);
    return left.localeCompare(right);
  });
  const items = entries.map(([name, value]) => {
    const item = node("div", "indicator");
    item.title = "Open indicator calculation";
    item.append(node("label", "", indicatorLabel(name)));
    item.append(node("strong", finite(value) >= 0 ? "up" : "down", indicatorValue(name, value)));
    item.append(node("small", "muted", humanIndicator(name, value, market.last_price)));
    item.addEventListener("click", () => showCalculation({
      kicker: "LATEST CLOSED-BAR INDICATOR",
      title: indicatorLabel(name),
      lead: "This causal indicator is computed from bars available at the latest close and is supplied to the strategy plugins and online learners.",
      metrics: [
        ["VALUE", indicatorValue(name, value), finite(value) >= 0 ? "up" : "down"],
        ["READ", humanIndicator(name, value, market.last_price), "cyan"],
        ["LAST PRICE", fmt(market.last_price), ""],
        ["BAR TIME", timeOnly(market.ts), ""],
      ],
      formula: `${indicatorLabel(name)} = technical_pipeline(latest closed OHLCV history)\ncurrent raw value = ${fmt(value, 8)}\ndisplay value = ${indicatorValue(name, value)}`,
      details: [
        "No future bar or target outcome is used in this value.",
        `Current interpretation: ${humanIndicator(name, value, market.last_price)}.`,
      ],
    }));
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
  const warmup = payload.warmup || {};
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
  // The warm-up is why the strategy and AI panels already have a record on the
  // first frame of the day. Without it on screen, a trust score that is not zero
  // before the market opens looks like it came from nowhere.
  items.push(
    systemItem(
      "warm-up",
      warmup.headline ? String(warmup.headline).toUpperCase() : "NOT WARMED",
      warmup.ran ? (warmup.cold ? "prediction" : "up") : "muted",
    ),
    systemItem(
      "warm-up origin",
      warmup.ran ? (warmup.cold ? "COLD START — FULL REPLAY" : "RESUMED FROM CHECKPOINT") : "—",
      warmup.ran && !warmup.cold ? "up" : "",
    ),
    systemItem(
      "bars replayed",
      warmup.ran ? `${Number(warmup.bars || 0).toLocaleString("en-IN")} · ${fmt(warmup.seconds, 1)}s` : "—",
    ),
    systemItem(
      "per instrument",
      (warmup.instruments || [])
        .map((item) => `${item.label} ${item.skipped ? "—" : Number(item.bars || 0).toLocaleString("en-IN")}`)
        .join(" · ") || "—",
      "cyan",
    ),
  );
  byId("system-state").replaceChildren(...items);
}

function rupee(value, digits = 0) {
  const parsed = finite(value);
  if (parsed === null) return "—";
  return `₹${parsed.toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

function signedRupee(value, digits = 0) {
  const parsed = finite(value);
  if (parsed === null) return "—";
  const body = Math.abs(parsed).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return `${parsed >= 0 ? "+" : "−"}₹${body}`;
}

function moneyTone(value) {
  const parsed = finite(value);
  if (parsed === null || parsed === 0) return "hold";
  return parsed > 0 ? "up" : "down";
}

function renderMoneySummary(money) {
  const reserve = money.reserve || {};
  const policy = money.policy || {};
  const net = finite(money.net_pnl) || 0;
  const items = [
    metric("paper equity", rupee(money.equity), `${rupee(money.cash)} cash · ${signedRupee(money.unrealized)} open`, "cyan"),
    metric("net result", signedRupee(net), `on ${rupee(money.capital_committed)} committed`, moneyTone(net)),
    metric(
      "reserve left",
      rupee(reserve.remaining),
      `${rupee(reserve.deployed)} drawn · ${reserve.calls || 0} top-up${reserve.calls === 1 ? "" : "s"}`,
      reserve.calls ? "hold" : "",
    ),
    metric(
      "lot size",
      money.lots ? `${money.lots} units` : "—",
      `one lot per leg · decides every ${fmt(policy.decision_seconds, 0)}s`,
      "prediction",
    ),
  ];
  byId("money-summary").replaceChildren(...items);
}

function moneyViewLine(wallet, policy) {
  const view = wallet.view;
  const line = node("div", "money-view inspectable");
  if (!view) {
    line.append(node("span", "muted", "THE VIEW IS WARMING — NO BLENDED READING YET"));
    return line;
  }
  const tone = moneyTone(view.edge_bps);
  line.append(node("span", "cyan", `VIEW ${signed(view.view, 2)}`));
  line.append(node("span", "", `EDGE ${signed(view.edge_bps, 1, "bp")} · ${signedRupee(view.edge_rupees)}`));
  line.append(node("span", "muted", `PROJECTS ${fmt(view.expected_move_bps, 1)}bp vs NEEDS ${fmt(view.required_move_bps, 1)}bp`));
  line.append(node("span", tone, `${view.agreement || "0/0"} AGREE`));
  line.title = "Click to inspect which source carried the view";
  line.addEventListener("click", () => showMoneyCalculation(wallet, policy));
  return line;
}

function renderMoneyWallets(money) {
  const root = byId("money-wallets");
  const wallets = money.wallets || [];
  const policy = money.policy || {};
  const cards = wallets.map((wallet) => {
    const position = wallet.position;
    const pnl = finite(wallet.total_pnl) || 0;
    const card = node("article", `money-card tone-${moneyTone(pnl)}`);

    const head = node("div", "money-head");
    const identity = node("div");
    identity.append(node("div", "money-leg", wallet.label));
    identity.append(node("div", "money-sub", `opening ${rupee(wallet.opening)} · ${wallet.trades || 0} trades`));
    const equity = node("div", "money-equity");
    equity.append(node("div", `money-value ${moneyTone(pnl)}`, rupee(wallet.equity)));
    equity.append(node("div", `money-delta ${moneyTone(pnl)}`, `${signedRupee(pnl)} · ${signed(wallet.return_pct, 2, "%")}`));
    head.append(identity, equity);

    const lot = node("div", "money-lot");
    if (position) {
      lot.append(tag(position.side));
      lot.append(node("span", "money-detail", `${position.units} × ${fmt(position.entry_price)} → ${fmt(position.price)}`));
      lot.append(node("span", `money-detail ${moneyTone(position.unrealized)}`, `${signedRupee(position.unrealized)} open`));
      lot.append(node("span", "money-detail muted", `stop ${fmt(position.stop_price)} · target ${fmt(position.target_price)}`));
    } else {
      lot.append(tag("FLAT"));
      lot.append(node("span", "money-detail muted", wallet.unrealized ? signedRupee(wallet.unrealized) : "no open lot"));
    }

    const stats = node("div", "money-stats");
    stats.append(
      readout("WIN RATE", wallet.trades ? `${percent(wallet.win_rate, 0)} · ${wallet.wins}/${wallet.trades}` : "—"),
      readout("COSTS PAID", rupee(wallet.costs_paid), wallet.costs_paid > Math.abs(wallet.total_pnl) ? "down" : ""),
      readout("RESERVE USED", rupee(wallet.reserve_drawn), wallet.reserve_drawn > 0 ? "hold" : ""),
      readout("DRAWDOWN", rupee(wallet.drawdown)),
    );

    card.append(head, lot, moneyViewLine(wallet, policy), stats);
    return card;
  });
  root.replaceChildren(...cards);
  if (!cards.length) root.append(node("div", "empty", "THE PAPER BOOK IS WARMING"));
}

function renderMoney(payload) {
  const money = payload.simulation || {};
  if (!money.wallets) {
    // A market-only frame has no board, so there is no leg to fund. Say so
    // rather than rendering three empty wallets.
    byId("money-summary").replaceChildren();
    byId("money-wallets").replaceChildren(node("div", "empty", "MONEY SIMULATOR NEEDS THE INDEX / CALL / PUT BOARD"));
    byId("money-decisions").replaceChildren();
    byId("money-trades").replaceChildren();
    return;
  }
  renderMoneySummary(money);
  renderMoneyWallets(money);

  const decisions = (money.decisions || []).slice(0, 30).map((decision) => [
    timeOnly(decision.at),
    decision.leg,
    tag(decision.action),
    decision.pnl ? signedRupee(decision.pnl) : "—",
    decision.reason,
  ]);
  byId("money-decisions").replaceChildren(
    table(["time", "leg", "action", "result", "why"], decisions, ["cyan", "cyan", "", "", "muted"]),
  );

  const trades = (money.trades || []).slice(0, 30).map((trade) => [
    timeOnly(trade.entry_ts),
    trade.leg,
    trade.side,
    trade.units,
    fmt(trade.entry_price),
    fmt(trade.exit_price),
    `${signedRupee(trade.gross)}`,
    `${rupee(trade.costs)}`,
    signedRupee(trade.net),
    trade.exit_reason,
  ]);
  byId("money-trades").replaceChildren(
    table(
      ["entry", "leg", "side", "units", "in", "out", "gross", "costs", "net", "exit"],
      trades,
      ["cyan", "cyan", "", "", "", "", "", "muted", "", "muted"],
    ),
  );
}

function showMoneyCalculation(wallet, policy = {}) {
  const view = wallet.view || {};
  const position = wallet.position;
  const sources = (view.sources || []).map((item) => ({
    source: item.source,
    value: finite(item.value) || 0,
    weight: finite(item.weight) || 0,
    weighted: finite(item.weighted) || 0,
    detail: item.detail || "",
  }));
  const sourceList = node("ul", "calculation-list");
  sources.forEach((item) => {
    sourceList.append(
      node(
        "li",
        "",
        `${item.source}: value ${signed(item.value, 3)} × weight ${fmt(item.weight, 2)} = ${signed(item.weighted, 3)} — ${item.detail}`,
      ),
    );
  });
  const weightTotal = sources.filter((item) => item.weight > 0).reduce((total, item) => total + item.weight, 0);
  showCalculation({
    kicker: "MONEY SIMULATOR CALCULATION",
    title: `${wallet.label} — the view behind the lot`,
    lead: "Every source the platform computes is blended into one signed view for this leg, then tested against what the round trip actually costs. The blend is renormalised over the sources that are present, so an untrained model does not dilute the rest.",
    metrics: [
      ["BLENDED VIEW", signed(view.view, 3), moneyTone(view.view)],
      ["EDGE / LOT", `${signed(view.edge_bps, 1, "bp")} · ${signedRupee(view.edge_rupees)}`, moneyTone(view.edge_bps)],
      ["PROJECTED MOVE", `${fmt(view.expected_move_bps, 1)}bp`, "prediction"],
      ["ROUND TRIP COST", `${fmt(view.required_move_bps, 1)}bp`, "muted"],
      ["OPEN LOT", position ? `${position.side} ${position.units} @ ${fmt(position.entry_price)}` : "FLAT", position ? directionClass(position.side) : "hold"],
    ],
    formula: [
      "view = Σ(sourceᵢ × weightᵢ) ÷ Σ(weightᵢ)   over the sources present",
      `expected_move_bps = ${fmt(view.expected_move_bps, 2)}   (the projection, restated in this leg's own price)`,
      `required_move_bps  = ${fmt(view.required_move_bps, 2)}   (round trip + the decay paid while holding)`,
      `edge_bps = |expected| − required = ${signed(view.edge_bps, 2)}`,
      `an entry needs |view| ≥ ${fmt(policy.entry_view, 2)} and edge ≥ ${fmt(policy.min_edge_bps, 2)}bp`,
      `positions are one lot of ${policy.lot_size || "—"} units; stops sit at ${fmt(policy.stop_atr, 2)}×ATR and targets at ${fmt(policy.target_atr, 2)}×ATR`,
    ].join("\n"),
    details: [
      sources.length
        ? `Sources present: ${sources.length} · total weight ${fmt(weightTotal, 2)}.`
        : "No source has produced a reading yet.",
      `Walking rupees: equity ${rupee(wallet.equity)}, cash ${rupee(wallet.cash)}, realised ${signedRupee(wallet.realized)}, costs paid ${rupee(wallet.costs_paid)}.`,
      `Reserve drawn ${rupee(wallet.reserve_drawn)} across ${wallet.top_ups || 0} top-up(s); repaid from profits before they count as this leg's own.`,
    ],
  });
  const content = byId("calculation-content");
  if (sources.length) content.append(calculationSection("SOURCE CONTRIBUTIONS", sourceList));
}

function render(payload) {
  latest = payload;
  renderOverview(payload);
  renderDecisions(payload);
  renderMoney(payload);
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

byId("calculation-close").addEventListener("click", () => byId("calculation-modal").close());
byId("calculation-modal").addEventListener("click", (event) => {
  if (event.target === byId("calculation-modal")) byId("calculation-modal").close();
});

function redrawCharts() {
  if (resizeFrame !== null) window.cancelAnimationFrame(resizeFrame);
  resizeFrame = window.requestAnimationFrame(() => {
    charts.forEach((market, chart) => drawChart(chart, market));
    resizeFrame = null;
  });
}

const chartObserver = typeof ResizeObserver === "undefined"
  ? null
  : new ResizeObserver(redrawCharts);

window.addEventListener("resize", redrawCharts);
if (window.visualViewport) window.visualViewport.addEventListener("resize", redrawCharts);

window.setInterval(() => {
  byId("clock").textContent = new Date().toLocaleTimeString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" }) + " IST";
}, 1000);

poll();
