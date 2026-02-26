const fileInput = document.getElementById("chartFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const previewContainer = document.getElementById("previewContainer");
const resultContainer = document.getElementById("resultContainer");
const analysisCanvas = document.getElementById("analysisCanvas");
const assetTypeSelect = document.getElementById("assetType");

let uploadedImage = null;

fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) {
    analyzeBtn.disabled = true;
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      uploadedImage = img;
      previewContainer.innerHTML = "";
      previewContainer.appendChild(img);
      analyzeBtn.disabled = false;
      resultContainer.innerHTML = "Ready. Click analyze chart.";
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", () => {
  if (!uploadedImage) {
    return;
  }

  const metrics = extractImageMetrics(uploadedImage);
  const plan = buildTradePlan(metrics, assetTypeSelect.value);
  renderPlan(plan, metrics);
});

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
      const luminance = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      sum += luminance;
      sumSq += luminance * luminance;
    }

    const mean = sum / height;
    const variance = Math.max(sumSq / height - mean * mean, 0);
    brightnessByX[x] = mean;
    contrastByX[x] = Math.sqrt(variance);
  }

  const firstWindow = brightnessByX.slice(0, Math.floor(width * 0.2));
  const lastWindow = brightnessByX.slice(Math.floor(width * 0.8));

  const meanFirst = average(firstWindow);
  const meanLast = average(lastWindow);
  const trendStrength = (meanLast - meanFirst) / 255;

  const avgContrast = average(contrastByX);
  const volatility = clamp(avgContrast / 80, 0.2, 2);

  const turbulence = stdDev(brightnessByX) / 64;

  return {
    trendStrength,
    volatility,
    turbulence,
    width,
    height,
  };
}

function buildTradePlan(metrics, market) {
  const { trendStrength, volatility, turbulence } = metrics;
  const bullish = trendStrength >= 0;
  const absoluteTrend = Math.abs(trendStrength);
  const confidence = clamp((absoluteTrend * 120 + (1 - Math.min(turbulence, 1)) * 35), 35, 93);

  const entry = 100;
  const riskBuffer = clamp(1.1 + volatility * 0.9 + turbulence * 0.6, 1.1, 3.2);
  const rewardMultiple = clamp(1.6 + absoluteTrend * 4, 1.4, 3.4);

  const stop = bullish ? entry - riskBuffer : entry + riskBuffer;
  const target = bullish
    ? entry + riskBuffer * rewardMultiple
    : entry - riskBuffer * rewardMultiple;

  const minutes = Math.round(clamp(45 + volatility * 90 + turbulence * 100, 45, 360));

  return {
    market,
    action: bullish ? "BUY" : "SELL",
    confidence: confidence.toFixed(1) + "%",
    entry: entry.toFixed(2),
    stop: stop.toFixed(2),
    target: target.toFixed(2),
    rr: `1 : ${(Math.abs(target - entry) / Math.abs(entry - stop)).toFixed(2)}`,
    timeframe: formatTimeframe(minutes),
    context: buildNarrative(market, bullish, volatility, turbulence, absoluteTrend),
  };
}

function buildNarrative(market, bullish, volatility, turbulence, absoluteTrend) {
  const momentum = absoluteTrend > 0.09 ? "strong" : absoluteTrend > 0.05 ? "moderate" : "weak";
  const volLabel = volatility > 1.2 ? "high" : volatility > 0.8 ? "normal" : "compressed";
  const structure = turbulence > 0.8 ? "choppy" : turbulence > 0.45 ? "mixed" : "clean";

  return `${market} chart shows ${momentum} ${bullish ? "upside" : "downside"} momentum with ${volLabel} volatility and ${structure} structure. Watch for confirmation candle near entry before executing. Scale position size to risk no more than 1-2% of capital.`;
}

function renderPlan(plan, metrics) {
  const actionClass = plan.action === "BUY" ? "signal-buy" : "signal-sell";

  resultContainer.innerHTML = `
    <span class="signal-pill ${actionClass}">${plan.action} BIAS</span>
    <div class="grid">
      <div class="metric"><h4>Market</h4><p>${plan.market}</p></div>
      <div class="metric"><h4>Confidence</h4><p>${plan.confidence}</p></div>
      <div class="metric"><h4>Entry</h4><p>${plan.entry}</p></div>
      <div class="metric"><h4>Target</h4><p>${plan.target}</p></div>
      <div class="metric"><h4>Stop Loss</h4><p>${plan.stop}</p></div>
      <div class="metric"><h4>Risk / Reward</h4><p>${plan.rr}</p></div>
      <div class="metric"><h4>Suggested timeframe</h4><p>${plan.timeframe}</p></div>
      <div class="metric"><h4>Chart volatility score</h4><p>${metrics.volatility.toFixed(2)}</p></div>
    </div>
    <p>${plan.context}</p>
  `;
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

function formatTimeframe(totalMinutes) {
  if (totalMinutes < 60) {
    return `${totalMinutes} minutes`;
  }

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes === 0 ? `${hours}h` : `${hours}h ${minutes}m`;
}
