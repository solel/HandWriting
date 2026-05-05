const drawCanvas = document.querySelector("#drawCanvas");
const previewCanvas = document.querySelector("#previewCanvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewCtx = previewCanvas.getContext("2d");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const brushSize = document.querySelector("#brushSize");
const canvasHint = document.querySelector("#canvasHint");
const inputStatus = document.querySelector("#inputStatus");
const predictionBadge = document.querySelector("#predictionBadge");
const resultCopy = document.querySelector("#resultCopy");
const probList = document.querySelector("#probList");
const pixelGrid = document.querySelector("#pixelGrid");
const vectorStrip = document.querySelector("#vectorStrip");
const inkAmount = document.querySelector("#inkAmount");
const normalRange = document.querySelector("#normalRange");
const missionBox = document.querySelector("#missionBox");
const missionButtons = document.querySelectorAll("[data-mission]");

const SIZE = 28;
const TEMPLATE_SIZE = SIZE * SIZE;
const TFJS_MODEL_URLS = [
  "https://cdn.jsdelivr.net/gh/dar5hak/offline-mnist@master/static/models/model.json",
  "https://dar5hak.github.io/offline-mnist/static/models/model.json",
  "https://dar5hak.github.io/offline-mnist/models/model.json",
];
let isDrawing = false;
let hasInk = false;
let lastPoint = null;
let currentPixels = new Array(TEMPLATE_SIZE).fill(0);
let templates = [];
let mnistModel = null;
let mnistModelReady = false;
let predictionRun = 0;

function setupCanvas() {
  drawCtx.fillStyle = "#0f172a";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = "#ffffff";
  drawCtx.shadowColor = "rgba(255,255,255,0.35)";
  drawCtx.shadowBlur = 2;
  clearPreview();
}

function clearPreview() {
  previewCtx.fillStyle = "#0f172a";
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
}

function pointerPosition(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((event.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function beginDraw(event) {
  isDrawing = true;
  hasInk = true;
  canvasHint.classList.add("hidden");
  inputStatus.textContent = "입력 중";
  lastPoint = pointerPosition(event);
  drawDot(lastPoint);
}

function moveDraw(event) {
  if (!isDrawing) return;
  const point = pointerPosition(event);
  drawCtx.lineWidth = Number(brushSize.value);
  drawCtx.beginPath();
  drawCtx.moveTo(lastPoint.x, lastPoint.y);
  drawCtx.lineTo(point.x, point.y);
  drawCtx.stroke();
  lastPoint = point;
  updatePixels(false);
}

function endDraw() {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  inputStatus.textContent = "예측 가능";
  updatePixels(false);
}

function drawDot(point) {
  drawCtx.fillStyle = "#ffffff";
  drawCtx.beginPath();
  drawCtx.arc(point.x, point.y, Number(brushSize.value) / 2, 0, Math.PI * 2);
  drawCtx.fill();
}

function resetAll() {
  hasInk = false;
  canvasHint.classList.remove("hidden");
  inputStatus.textContent = "입력 대기";
  predictionBadge.textContent = "?";
  resultCopy.textContent = "예측하기를 누르면 출력층 10개 노드가 계산한 0부터 9까지의 가능성이 표시됩니다.";
  setupCanvas();
  currentPixels = new Array(TEMPLATE_SIZE).fill(0);
  renderPixels(currentPixels);
  renderVector(currentPixels);
  renderProbabilities(new Array(10).fill(0), -1);
  updateStats(currentPixels);
}

function updatePixels(runPrediction = true) {
  const pixels = canvasTo28x28();
  currentPixels = pixels;
  renderPreview();
  renderPixels(pixels);
  renderVector(pixels);
  updateStats(pixels);
  if (runPrediction) predict();
}

function canvasTo28x28() {
  const source = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  const bounds = findInkBounds(source);
  if (!bounds) return new Array(TEMPLATE_SIZE).fill(0);

  const margin = 24;
  const side = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) + margin * 2;
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const sampleMinX = cx - side / 2;
  const sampleMinY = cy - side / 2;
  const cell = side / SIZE;
  const pixels = [];

  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      let total = 0;
      let samples = 0;
      for (let sy = 0; sy < 3; sy += 1) {
        for (let sx = 0; sx < 3; sx += 1) {
          const x = Math.round(sampleMinX + (col + (sx + 0.5) / 3) * cell);
          const y = Math.round(sampleMinY + (row + (sy + 0.5) / 3) * cell);
          total += sampleBrightness(source, x, y);
          samples += 1;
        }
      }
      pixels.push(total / samples);
    }
  }

  return soften(pixels);
}

function sampleBrightness(imageData, x, y) {
  if (x < 0 || x >= imageData.width || y < 0 || y >= imageData.height) return 0;
  const index = (y * imageData.width + x) * 4;
  return imageData.data[index] / 255;
}

function findInkBounds(imageData) {
  let minX = imageData.width;
  let minY = imageData.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < imageData.height; y += 1) {
    for (let x = 0; x < imageData.width; x += 1) {
      if (sampleBrightness(imageData, x, y) > 0.12) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  return maxX < 0 ? null : { minX, minY, maxX, maxY };
}

function soften(pixels) {
  return pixels.map((value, index) => {
    const row = Math.floor(index / SIZE);
    const col = index % SIZE;
    let total = value * 4;
    let count = 4;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) continue;
        const nr = row + dy;
        const nc = col + dx;
        if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
          total += pixels[nr * SIZE + nc];
          count += 1;
        }
      }
    }
    return Math.min(1, total / count);
  });
}

function renderPreview() {
  clearPreview();
  previewCtx.drawImage(drawCanvas, 0, 0, previewCanvas.width, previewCanvas.height);
}

function renderPixels(pixels) {
  pixelGrid.innerHTML = "";
  pixels.forEach((value) => {
    const cell = document.createElement("div");
    cell.className = "pixel-cell";
    const tone = Math.round(15 + value * 240);
    cell.style.backgroundColor = `rgb(${tone}, ${tone}, ${tone})`;
    cell.title = value.toFixed(2);
    pixelGrid.appendChild(cell);
  });
}

function renderVector(pixels) {
  vectorStrip.innerHTML = "";
  const sampleIndexes = [
    0, 1, 2, 3, 4, 5, 6, 7, 92, 93, 94, 95, 96, 97, 210, 211, 212, 213, 214, 215,
    406, 407, 408, 409, 410, 411, 574, 575, 576, 577, 578, 579, 776, 777, 778, 779,
    780, 781, 782, 783,
  ];
  sampleIndexes.forEach((index) => {
    const cell = document.createElement("div");
    cell.className = "vector-cell";
    const value = pixels[index] || 0;
    cell.textContent = value.toFixed(2);
    cell.style.background = `rgba(23, 107, 135, ${0.08 + value * 0.62})`;
    vectorStrip.appendChild(cell);
  });
}

function updateStats(pixels) {
  const active = pixels.filter((value) => value > 0.08);
  const ink = Math.round((active.length / pixels.length) * 100);
  const max = Math.max(...pixels);
  inkAmount.textContent = `잉크 ${ink}%`;
  normalRange.textContent = `0.00~${max.toFixed(2)}`;
}

function renderProbabilities(probabilities, winner) {
  probList.innerHTML = "";
  probabilities.forEach((probability, digit) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="digit-label">${digit}</div>
      <div class="prob-track"><div class="prob-fill"></div></div>
      <div class="prob-value">${Math.round(probability * 100)}%</div>
    `;
    const fill = row.querySelector(".prob-fill");
    fill.style.width = `${Math.round(probability * 100)}%`;
    if (digit === winner) {
      row.querySelector(".digit-label").style.background = "var(--yellow)";
      fill.style.background = "linear-gradient(90deg, #176b87, #f2c94c)";
    }
    probList.appendChild(row);
  });
}

async function predict() {
  if (!hasInk) {
    resultCopy.textContent = "먼저 캔버스에 숫자를 그려주세요.";
    return;
  }

  const runId = (predictionRun += 1);
  if (mnistModelReady) {
    const modelProbabilities = await predictWithMnistModel(currentPixels);
    if (runId !== predictionRun) return;
    if (modelProbabilities) {
      showPrediction(modelProbabilities, "실제 MNIST로 학습된 모델이 0부터 9까지의 가능성을 계산했습니다.");
      return;
    }
  }

  const rawScores = templates.map((digitTemplates) =>
    Math.max(...digitTemplates.map((template) => similarity(currentPixels, template.pixels)))
  );
  const features = extractFeatures(currentPixels);
  const adjustedScores = rawScores.map((score, digit) => score + featureBonus(digit, features));
  const probabilities = softmax(adjustedScores, 13);
  showPrediction(probabilities, "가벼운 내장 예측 모델이 0부터 9까지의 가능성을 계산했습니다.");
}

function showPrediction(probabilities, sourceMessage) {
  const winner = probabilities.indexOf(Math.max(...probabilities));
  predictionBadge.textContent = winner;
  renderProbabilities(probabilities, winner);
  resultCopy.textContent = `AI의 예측: ${winner}. ${sourceMessage}`;
  inputStatus.textContent = "예측 완료";
}

async function predictWithMnistModel(pixels) {
  if (!window.tf || !mnistModel) return null;

  try {
    const probabilities = window.tf.tidy(() => {
      const input = window.tf.tensor4d(pixels, [1, SIZE, SIZE, 1]);
      const output = mnistModel.predict(input);
      return Array.from(output.dataSync());
    });
    if (probabilities.length !== 10 || probabilities.some((value) => !Number.isFinite(value))) {
      return null;
    }
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    return total > 0 ? probabilities.map((value) => value / total) : null;
  } catch (error) {
    mnistModelReady = false;
    return null;
  }
}

function similarity(input, template) {
  let dot = 0;
  let inputNorm = 0;
  let templateNorm = 0;
  for (let i = 0; i < input.length; i += 1) {
    dot += input[i] * template[i];
    inputNorm += input[i] * input[i];
    templateNorm += template[i] * template[i];
  }
  if (inputNorm === 0 || templateNorm === 0) return -1;
  return dot / Math.sqrt(inputNorm * templateNorm);
}

function softmax(scores, temperature) {
  const max = Math.max(...scores);
  const exps = scores.map((score) => Math.exp((score - max) * temperature));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function extractFeatures(pixels) {
  let total = 0;
  let cx = 0;
  let cy = 0;
  let top = 0;
  let bottom = 0;
  let left = 0;
  let right = 0;
  let center = 0;

  pixels.forEach((value, index) => {
    const row = Math.floor(index / SIZE);
    const col = index % SIZE;
    total += value;
    cx += col * value;
    cy += row * value;
    if (row < 9) top += value;
    if (row > 18) bottom += value;
    if (col < 9) left += value;
    if (col > 18) right += value;
    if (row >= 9 && row <= 18 && col >= 9 && col <= 18) center += value;
  });

  return {
    total,
    cx: total ? cx / total : 14,
    cy: total ? cy / total : 14,
    topRatio: total ? top / total : 0,
    bottomRatio: total ? bottom / total : 0,
    leftRatio: total ? left / total : 0,
    rightRatio: total ? right / total : 0,
    centerRatio: total ? center / total : 0,
  };
}

function featureBonus(digit, f) {
  let bonus = 0;
  if (digit === 1 && f.centerRatio > 0.24) bonus += 0.06;
  if (digit === 7 && f.topRatio > 0.28 && f.bottomRatio < 0.24) bonus += 0.08;
  if (digit === 0 && f.centerRatio < 0.2 && f.total > 55) bonus += 0.07;
  if (digit === 8 && f.centerRatio > 0.2 && f.total > 72) bonus += 0.05;
  if (digit === 6 && f.leftRatio > f.rightRatio + 0.04) bonus += 0.04;
  if (digit === 9 && f.rightRatio > f.leftRatio - 0.02 && f.topRatio > 0.25) bonus += 0.04;
  if (digit === 4 && f.rightRatio > 0.26 && f.centerRatio > 0.17) bonus += 0.05;
  if (digit === 2 && f.topRatio > 0.22 && f.bottomRatio > 0.24) bonus += 0.03;
  if (digit === 3 && f.rightRatio > f.leftRatio + 0.04) bonus += 0.04;
  if (digit === 5 && f.leftRatio > 0.2 && f.topRatio > 0.2) bonus += 0.03;
  return bonus;
}

function createTemplates() {
  const definitions = [
    ["0", "M 14 4 C 6 4 4 10 4 14 C 4 22 8 25 14 25 C 21 25 24 20 24 14 C 24 8 21 4 14 4 Z"],
    ["1", "M 14 5 L 14 25 M 10 9 L 14 5 L 18 9 M 10 25 L 19 25"],
    ["2", "M 6 9 C 9 4 18 4 21 9 C 24 14 19 17 14 20 L 6 25 L 24 25"],
    ["3", "M 7 6 C 16 2 24 6 21 13 C 20 16 16 16 13 16 M 13 16 C 19 16 24 19 21 24 C 17 29 8 25 6 21"],
    ["4", "M 21 25 L 21 4 L 5 18 L 25 18"],
    ["5", "M 23 5 L 8 5 L 7 14 C 11 11 21 12 23 18 C 25 25 13 28 6 22"],
    ["6", "M 22 7 C 15 2 6 7 6 17 C 6 25 13 27 19 23 C 25 19 21 12 14 13 C 9 14 7 17 8 21"],
    ["7", "M 5 5 L 24 5 L 12 25"],
    ["8", "M 14 4 C 7 4 7 14 14 14 C 22 14 22 4 14 4 Z M 14 14 C 5 14 5 25 14 25 C 23 25 23 14 14 14 Z"],
    ["9", "M 22 14 C 20 19 14 19 10 17 C 3 13 7 4 15 5 C 23 6 24 15 20 21 C 18 24 14 26 10 25"],
  ];

  templates = Array.from({ length: 10 }, () => []);

  definitions.forEach(([digit, path]) => {
    const target = templates[Number(digit)];
    [
      { rotation: 0, dx: 0, dy: 0, scaleX: 1, scaleY: 1, lineWidth: 4.3 },
      { rotation: -0.12, dx: -0.3, dy: 0, scaleX: 0.98, scaleY: 1.04, lineWidth: 4.1 },
      { rotation: 0.12, dx: 0.3, dy: 0, scaleX: 1.02, scaleY: 0.98, lineWidth: 4.1 },
      { rotation: 0, dx: 0, dy: 0.4, scaleX: 0.92, scaleY: 1.08, lineWidth: 4.6 },
      { rotation: 0, dx: 0, dy: -0.4, scaleX: 1.08, scaleY: 0.92, lineWidth: 4.0 },
    ].forEach((variant) => {
      target.push({ pixels: rasterizePath(path, variant) });
    });
  });

  const textFonts = [
    "Arial",
    "Verdana",
    "Trebuchet MS",
    "Georgia",
    "Times New Roman",
    "Malgun Gothic",
  ];
  for (let digit = 0; digit <= 9; digit += 1) {
    textFonts.forEach((font) => {
      [
        { rotation: 0, dx: 0, dy: 0, scaleX: 1, scaleY: 1, fontSize: 25, weight: 800 },
        { rotation: -0.1, dx: -0.5, dy: 0.2, scaleX: 0.94, scaleY: 1.05, fontSize: 25, weight: 800 },
        { rotation: 0.1, dx: 0.5, dy: 0.2, scaleX: 1.04, scaleY: 0.96, fontSize: 25, weight: 800 },
        { rotation: 0, dx: 0, dy: 0.5, scaleX: 0.88, scaleY: 1.08, fontSize: 24, weight: 700 },
      ].forEach((variant) => {
        templates[digit].push({ pixels: rasterizeText(String(digit), font, variant) });
      });
    });
  }
}

function rasterizePath(path, variant = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.strokeStyle = "#fff";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = variant.lineWidth || 4.3;
  const p = new Path2D(path);
  ctx.save();
  ctx.translate(SIZE / 2 + (variant.dx || 0), SIZE / 2 + (variant.dy || 0));
  ctx.rotate(variant.rotation || 0);
  ctx.scale(variant.scaleX || 1, variant.scaleY || 1);
  ctx.translate(-SIZE / 2, -SIZE / 2);
  ctx.stroke(p);
  ctx.restore();
  return canvasToPixels(canvas);
}

function rasterizeText(digit, font, variant = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `${variant.weight || 800} ${variant.fontSize || 25}px ${font}`;
  ctx.save();
  ctx.translate(SIZE / 2 + (variant.dx || 0), SIZE / 2 + 1 + (variant.dy || 0));
  ctx.rotate(variant.rotation || 0);
  ctx.scale(variant.scaleX || 1, variant.scaleY || 1);
  ctx.fillText(digit, 0, 0);
  ctx.restore();
  return canvasToPixels(canvas);
}

function canvasToPixels(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  const pixels = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    pixels.push(imageData.data[i] / 255);
  }
  return soften(pixels);
}

async function loadMnistModel() {
  if (!window.tf) {
    inputStatus.textContent = "내장 모델";
    return;
  }

  inputStatus.textContent = "모델 준비 중";
  for (const url of TFJS_MODEL_URLS) {
    try {
      const model = await window.tf.loadLayersModel(url);
      const outputShape = model.outputs && model.outputs[0] && model.outputs[0].shape;
      if (!outputShape || outputShape[outputShape.length - 1] !== 10) {
        model.dispose();
        continue;
      }
      mnistModel = model;
      mnistModelReady = true;
      inputStatus.textContent = hasInk ? "예측 가능" : "실제 모델 준비";
      resultCopy.textContent = "예측하기를 누르면 실제 MNIST 학습 모델의 소프트맥스 확률이 표시됩니다.";
      return;
    } catch (error) {
      mnistModelReady = false;
    }
  }

  inputStatus.textContent = hasInk ? "예측 가능" : "내장 모델";
  resultCopy.textContent = "실제 모델을 불러오지 못하면 가벼운 내장 예측 모델로 바로 실행됩니다.";
}

drawCanvas.addEventListener("pointerdown", beginDraw);
drawCanvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);
predictBtn.addEventListener("click", () => {
  updatePixels(true);
});
clearBtn.addEventListener("click", resetAll);

missionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    missionButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    missionBox.textContent = button.dataset.mission;
  });
});

createTemplates();
resetAll();
loadMnistModel();
