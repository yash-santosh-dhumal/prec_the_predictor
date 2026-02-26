const fileInput = document.getElementById("chartFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const previewContainer = document.getElementById("previewContainer");
const resultContainer = document.getElementById("resultContainer");
const analysisCanvas = document.getElementById("analysisCanvas");
const assetTypeSelect = document.getElementById("assetType");
const symbolInput = document.getElementById("symbolInput");
const timeframeSelect = document.getElementById("timeframe");

let uploadedImage = null;

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) {
    uploadedImage = null;
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      uploadedImage = img;
      previewContainer.innerHTML = "";
      previewContainer.appendChild(img);
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  const market = assetTypeSelect.value;
  const symbol = symbolInput.value.trim();
  const timeframe = timeframeSelect.value;

  if (!symbol) {
    resultContainer.textContent = "Please enter a symbol.";
    return;
  }

  analyzeBtn.disabled = true;
  resultContainer.innerHTML = "Fetching market data and training model...";

  try {
    const candles = await fetchCandles(market, symbol, timeframe);
    if (!candles || candles.length < 120) {
      throw new Error("Not enough candle data returned for this symbol/timeframe.");
    }

    const dataset = buildDataset(candles);
    if (dataset.samples.length < 60) {
      throw new Error("Unable to build enough training samples from the fetched data.");
    }

    const trained = trainLogisticModel(dataset.samples, dataset.labels, 350, 0.08);
    const latestFeatures = dataset.samples[dataset.samples.length - 1];
    const probabilityUp = sigmoid(dot(trained.weights, latestFeatures) + trained.bias);

    const metrics = uploadedImage ? extractImageMetrics(uploadedImage) : null;
    const plan = buildTradePlan({
      candles,
      market,
      symbol,
      timeframe,
      probabilityUp,
      modelAccuracy: trained.accuracy,
      imageMetrics: metrics,
    });

    renderPlan(plan);
  } catch (error) {
    resultContainer.innerHTML = `<strong>Analysis error:</strong> ${error.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
});

async function fetchCandles(market, symbol, timeframe) {
  if (market === "Crypto") {
    const interval = mapToBinanceInterval(timeframe);
    const endpoint = `https://api.binance.com/api/v3/klines?symbol=${encodeURIComponent(
      symbol.toUpperCase()
    )}&interval=${interval}&limit=500`;

    const response = await fetch(endpoint);
    if (!response.ok) {
      throw new Error("Binance data fetch failed. Check symbol (example: BTCUSDT). ");
    }

    const rows = await response.json();
    return rows.map((row) => ({
      openTime: Number(row[0]),
      open: Number(row[1]),
      high: Number(row[2]),
      low: Number(row[3]),
      close: Number(row[4]),
      volume: Number(row[5]),
    }));
  }

  const stooqSymbol = normalizeStooqSymbol(symbol, market);
  const url = `https://stooq.com/q/d/l/?s=${encodeURIComponent(stooqSymbol)}&i=d`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Stooq data fetch failed. Check symbol (example: AAPL.US, EURUSD, XAUUSD). ");
  }

  const csv = await response.text();
  const candles = parseStooqCsv(csv);
  return timeframe === "1d" ? candles : resampleCandles(candles, timeframeToDays(timeframe));
}

function normalizeStooqSymbol(symbol, market) {
  const raw = symbol.trim().toLowerCase();
  if (market === "Stock" || market === "Index") {
    return raw.includes(".") ? raw : `${raw}.us`;
  }
  return raw;
}

function parseStooqCsv(csv) {
  const lines = csv.trim().split("\n");
  if (lines.length <= 1 || lines[0].includes("N/D")) {
    return [];
  }

  const data = [];
  for (let i = 1; i < lines.length; i++) {
    const [date, open, high, low, close, volume] = lines[i].split(",");
    if ([open, high, low, close].some((value) => value === "N/D")) {
      continue;
    }

    data.push({
      openTime: Date.parse(date),
      open: Number(open),
      high: Number(high),
      low: Number(low),
      close: Number(close),
      volume: Number(volume || 0),
    });
  }
  return data;
}

function resampleCandles(candles, bucketDays) {
  if (bucketDays <= 1) {
    return candles;
  }
  const grouped = [];

  for (let i = 0; i < candles.length; i += bucketDays) {
    const slice = candles.slice(i, i + bucketDays);
    if (!slice.length) {
      continue;
    }

    grouped.push({
      openTime: slice[0].openTime,
      open: slice[0].open,
      high: Math.max(...slice.map((c) => c.high)),
      low: Math.min(...slice.map((c) => c.low)),
      close: slice[slice.length - 1].close,
      volume: slice.reduce((sum, candle) => sum + candle.volume, 0),
    });
  }
  return grouped;
}

function buildDataset(candles) {
  const closes = candles.map((c) => c.close);
  const highs = candles.map((c) => c.high);
  const lows = candles.map((c) => c.low);
  const volumes = candles.map((c) => c.volume);

  const rsi = calcRsi(closes, 14);
  const sma20 = calcSma(closes, 20);
  const sma50 = calcSma(closes, 50);
  const ema12 = calcEma(closes, 12);
  const ema26 = calcEma(closes, 26);
  const atr14 = calcAtr(highs, lows, closes, 14);

  const samples = [];
  const labels = [];

  for (let i = 55; i < candles.length - 2; i++) {
    const close = closes[i];
    const nextClose = closes[i + 2];
    const ret2 = (nextClose - close) / close;

    const featureRow = [
      (rsi[i] - 50) / 50,
      (close - sma20[i]) / close,
      (close - sma50[i]) / close,
      (ema12[i] - ema26[i]) / close,
      atr14[i] / close,
      (close - closes[i - 5]) / closes[i - 5],
      volumeZScore(volumes, i, 20),
    ].map((value) => (Number.isFinite(value) ? value : 0));

    samples.push(featureRow);
    labels.push(ret2 > 0 ? 1 : 0);
  }

  const normalizedSamples = normalizeColumns(samples);
  return { samples: normalizedSamples, labels };
}

function trainLogisticModel(samples, labels, epochs, lr) {
  const featureCount = samples[0].length;
  const split = Math.floor(samples.length * 0.8);

  const trainX = samples.slice(0, split);
  const trainY = labels.slice(0, split);
  const testX = samples.slice(split);
  const testY = labels.slice(split);

  const weights = new Array(featureCount).fill(0);
  let bias = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let gradB = 0;
    const gradW = new Array(featureCount).fill(0);

    for (let i = 0; i < trainX.length; i++) {
      const pred = sigmoid(dot(weights, trainX[i]) + bias);
      const err = pred - trainY[i];

      gradB += err;
      for (let j = 0; j < featureCount; j++) {
        gradW[j] += err * trainX[i][j];
      }
    }

    const scale = 1 / trainX.length;
    bias -= lr * gradB * scale;
    for (let j = 0; j < featureCount; j++) {
      weights[j] -= lr * gradW[j] * scale;
    }
  }

  let correct = 0;
  for (let i = 0; i < testX.length; i++) {
    const pred = sigmoid(dot(weights, testX[i]) + bias) >= 0.5 ? 1 : 0;
    if (pred === testY[i]) {
      correct++;
    }
  }
  const accuracy = testX.length ? correct / testX.length : 0.5;

  return { weights, bias, accuracy };
}

function buildTradePlan({ candles, market, symbol, timeframe, probabilityUp, modelAccuracy, imageMetrics }) {
  const last = candles[candles.length - 1];
  const entry = last.close;

  const highs = candles.map((c) => c.high);
  const lows = candles.map((c) => c.low);
  const closes = candles.map((c) => c.close);
  const atr = calcAtr(highs, lows, closes, 14);
  const atrNow = atr[atr.length - 1] || entry * 0.01;

  let hybridProb = probabilityUp;
  if (imageMetrics) {
    const imageDirectionProb = clamp(0.5 + imageMetrics.trendStrength * 0.9, 0.05, 0.95);
    hybridProb = 0.75 * probabilityUp + 0.25 * imageDirectionProb;
  }

  const buyBias = hybridProb >= 0.5;
  const confidence = Math.round((0.55 * Math.abs(hybridProb - 0.5) * 200 + 0.45 * modelAccuracy * 100));

  const stopDistance = Math.max(atrNow * 1.2, entry * 0.006);
  const rr = clamp(1.4 + Math.abs(hybridProb - 0.5) * 3.4, 1.4, 3.2);

  const stop = buyBias ? entry - stopDistance : entry + stopDistance;
  const target = buyBias ? entry + stopDistance * rr : entry - stopDistance * rr;

  return {
    market,
    symbol,
    timeframe,
    action: buyBias ? "BUY" : "SELL",
    confidence: `${confidence}%`,
    modelProbability: `${(hybridProb * 100).toFixed(1)}% ${buyBias ? "upside" : "downside"}`,
    modelAccuracy: `${(modelAccuracy * 100).toFixed(1)}%`,
    entry: entry.toFixed(4),
    stop: stop.toFixed(4),
    target: target.toFixed(4),
    rr: `1 : ${rr.toFixed(2)}`,
    narrative: buildNarrative({ market, symbol, hybridProb, modelAccuracy, timeframe, imageMetrics }),
  };
}

function buildNarrative({ market, symbol, hybridProb, modelAccuracy, timeframe, imageMetrics }) {
  const edge = Math.abs(hybridProb - 0.5);
  const edgeLabel = edge > 0.18 ? "high" : edge > 0.1 ? "moderate" : "low";
  const direction = hybridProb >= 0.5 ? "bullish" : "bearish";

  let extra = "";
  if (imageMetrics) {
    const chartStructure = imageMetrics.turbulence > 0.8 ? "choppy" : "cleaner";
    extra = ` Uploaded image suggests ${chartStructure} structure with volatility score ${imageMetrics.volatility.toFixed(2)}.`;
  }

  return `${market} ${symbol.toUpperCase()} ${timeframe} model shows ${direction} probability with ${edgeLabel} statistical edge. Validation split accuracy is ${
    (modelAccuracy * 100).toFixed(1)
  }%, so treat this as probabilistic guidance and confirm with your execution setup.${extra}`;
}

function renderPlan(plan) {
  const actionClass = plan.action === "BUY" ? "signal-buy" : "signal-sell";

  resultContainer.innerHTML = `
    <span class="signal-pill ${actionClass}">${plan.action} BIAS</span>
    <div class="grid">
      <div class="metric"><h4>Market</h4><p>${plan.market}</p></div>
      <div class="metric"><h4>Symbol</h4><p>${plan.symbol.toUpperCase()}</p></div>
      <div class="metric"><h4>Timeframe</h4><p>${plan.timeframe}</p></div>
      <div class="metric"><h4>Confidence</h4><p>${plan.confidence}</p></div>
      <div class="metric"><h4>Model Direction Prob.</h4><p>${plan.modelProbability}</p></div>
      <div class="metric"><h4>Validation Accuracy</h4><p>${plan.modelAccuracy}</p></div>
      <div class="metric"><h4>Entry</h4><p>${plan.entry}</p></div>
      <div class="metric"><h4>Target</h4><p>${plan.target}</p></div>
      <div class="metric"><h4>Stop Loss</h4><p>${plan.stop}</p></div>
      <div class="metric"><h4>Risk / Reward</h4><p>${plan.rr}</p></div>
    </div>
    <p>${plan.narrative}</p>
  `;
}

function extractImageMetrics(image) {
  const ctx = analysisCanvas.getContext("2d", { willReadFrequently: true });
  const width = 420;
  const height = Math.max(220, Math.round((image.height / image.width) * width));

  analysisCanvas.width = width;
  analysisCanvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);

  const pixels = ctx.getImageData(0, 0, width, height).data;
  const brightnessByX = new Array(width).fill(0);
  const contrastByX = new Array(width).fill(0);

  for (let x = 0; x < width; x++) {
    let sum = 0;
    let sumSq = 0;
    for (let y = 0; y < height; y++) {
      const i = (y * width + x) * 4;
      const lum = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      sum += lum;
      sumSq += lum * lum;
    }

    const mean = sum / height;
    brightnessByX[x] = mean;
    contrastByX[x] = Math.sqrt(Math.max(sumSq / height - mean * mean, 0));
  }

  const firstWindow = brightnessByX.slice(0, Math.floor(width * 0.2));
  const lastWindow = brightnessByX.slice(Math.floor(width * 0.8));

  return {
    trendStrength: (average(lastWindow) - average(firstWindow)) / 255,
    volatility: clamp(average(contrastByX) / 80, 0.2, 2),
    turbulence: stdDev(brightnessByX) / 64,
  };
}

function calcSma(values, period) {
  const out = new Array(values.length).fill(values[0]);
  let running = 0;
  for (let i = 0; i < values.length; i++) {
    running += values[i];
    if (i >= period) {
      running -= values[i - period];
    }
    out[i] = running / Math.min(i + 1, period);
  }
  return out;
}

function calcEma(values, period) {
  const out = new Array(values.length).fill(values[0]);
  const alpha = 2 / (period + 1);
  out[0] = values[0];
  for (let i = 1; i < values.length; i++) {
    out[i] = alpha * values[i] + (1 - alpha) * out[i - 1];
  }
  return out;
}

function calcRsi(closes, period) {
  const rsi = new Array(closes.length).fill(50);
  let gain = 0;
  let loss = 0;

  for (let i = 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    gain += Math.max(diff, 0);
    loss += Math.max(-diff, 0);

    if (i > period) {
      const oldDiff = closes[i - period + 1] - closes[i - period];
      gain -= Math.max(oldDiff, 0);
      loss -= Math.max(-oldDiff, 0);
    }

    if (i >= period) {
      const avgGain = gain / period;
      const avgLoss = loss / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi[i] = 100 - 100 / (1 + rs);
    }
  }
  return rsi;
}

function calcAtr(highs, lows, closes, period) {
  const tr = new Array(closes.length).fill(0);
  for (let i = 1; i < closes.length; i++) {
    const highLow = highs[i] - lows[i];
    const highClose = Math.abs(highs[i] - closes[i - 1]);
    const lowClose = Math.abs(lows[i] - closes[i - 1]);
    tr[i] = Math.max(highLow, highClose, lowClose);
  }
  return calcEma(tr, period);
}

function normalizeColumns(samples) {
  if (!samples.length) {
    return samples;
  }
  const cols = samples[0].length;
  const means = new Array(cols).fill(0);
  const stds = new Array(cols).fill(1);

  for (const row of samples) {
    for (let j = 0; j < cols; j++) {
      means[j] += row[j];
    }
  }
  for (let j = 0; j < cols; j++) {
    means[j] /= samples.length;
  }

  for (const row of samples) {
    for (let j = 0; j < cols; j++) {
      stds[j] += (row[j] - means[j]) ** 2;
    }
  }
  for (let j = 0; j < cols; j++) {
    stds[j] = Math.sqrt(stds[j] / samples.length) || 1;
  }

  return samples.map((row) => row.map((value, j) => (value - means[j]) / stds[j]));
}

function volumeZScore(volumes, idx, lookback) {
  const start = Math.max(0, idx - lookback + 1);
  const window = volumes.slice(start, idx + 1);
  const mean = average(window);
  const sd = stdDev(window) || 1;
  return (volumes[idx] - mean) / sd;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function stdDev(values) {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function mapToBinanceInterval(timeframe) {
  if (timeframe === "15m") {
    return "15m";
  }
  if (timeframe === "4h") {
    return "4h";
  }
  if (timeframe === "1d") {
    return "1d";
  }
  return "1h";
}

function timeframeToDays(timeframe) {
  if (timeframe === "15m") {
    return 1;
  }
  if (timeframe === "1h") {
    return 1;
  }
  if (timeframe === "4h") {
    return 4;
  }
  return 1;
}
