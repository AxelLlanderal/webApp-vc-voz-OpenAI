import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const btnToggleSuspend = document.getElementById("btnToggleSuspend");

const statusText = document.getElementById("statusText");
const commandText = document.getElementById("commandText");
const debugText = document.getElementById("debugText");
const modeBadge = document.getElementById("modeBadge");

let running = false;
let suspended = false;

let handLandmarker = null;

let lastGoodCommandAt = 0;
const SUSPEND_AFTER_MS = 3500;

// Movimiento para despertar
let lastFrameSmall = null;
const MOTION_WAKE_THRESHOLD = 18;
const MOTION_SAMPLE_SIZE = 128;

// Suavizado
const STABLE_FRAMES = 3;
let lastRawCommand = null;
let sameCount = 0;
let currentStableCommand = "—";

// Cámara espejada
const MIRRORED_VIEW = true;

// ===================== Global event =====================
function emitGlobalCommand(command) {
  window.dispatchEvent(new CustomEvent("control:command", {
    detail: { source: "gestos", command }
  }));
}

// ===================== UI Helpers =====================
function setModeUI(isSuspended) {
  suspended = isSuspended;
  if (suspended) {
    modeBadge.className = "badge text-bg-warning";
    modeBadge.textContent = "SUSPENDIDO";
    btnToggleSuspend.textContent = "▶ Despertar";
  } else {
    modeBadge.className = "badge text-bg-success";
    modeBadge.textContent = "ACTIVO";
    btnToggleSuspend.textContent = "⏸ Suspender";
  }
}

function setStatus(msg) {
  statusText.textContent = msg;
}

function setCommand(msg) {
  commandText.textContent = msg;

  // ✅ actualiza orden global si es válida
  if (msg && msg !== "—") {
    emitGlobalCommand(msg);
  }
}

// ===================== Camera =====================
async function setupCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("getUserMedia no está disponible en este navegador/contexto.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((res) => {
    video.onloadedmetadata = () => res();
  });

  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ===================== MediaPipe Load =====================
async function loadHandLandmarker() {
  setStatus("Cargando MediaPipe (WASM + modelo) ...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm"
  );

  const modelUrl =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

  const resp = await fetch(modelUrl);
  if (!resp.ok) throw new Error("No se pudo descargar el modelo.");
  const buffer = await resp.arrayBuffer();
  const blobUrl = URL.createObjectURL(new Blob([buffer], { type: "application/octet-stream" }));

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: blobUrl,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 1
  });

  setStatus("Modelo listo. Presiona Iniciar.");
}

// ===================== Geometry / Fingers =====================
function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function fingerExtended(lm, tipIdx, pipIdx) {
  return lm[tipIdx].y < lm[pipIdx].y;
}

function thumbExtended(lm) {
  return dist(lm[4], lm[5]) > 0.12 && dist(lm[4], lm[0]) > 0.18;
}

function fingerFlags(lm) {
  const th = thumbExtended(lm);
  const idx = fingerExtended(lm, 8, 6);
  const mid = fingerExtended(lm, 12, 10);
  const ring = fingerExtended(lm, 16, 14);
  const pinky = fingerExtended(lm, 20, 18);
  return { th, idx, mid, ring, pinky };
}

function countExtendedFingers(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  return [th, idx, mid, ring, pinky].filter(Boolean).length;
}

function isOKSign(lm) {
  const okClose = dist(lm[4], lm[8]) < 0.06;
  const mid = fingerExtended(lm, 12, 10);
  const ring = fingerExtended(lm, 16, 14);
  const pinky = fingerExtended(lm, 20, 18);
  return okClose && (mid || ring || pinky);
}

function isOpenPalm(lm) {
  return countExtendedFingers(lm) === 5;
}

function isFist(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  return !th && !idx && !mid && !ring && !pinky;
}

function dirFromVx(vx) {
  if (Math.abs(vx) < 0.08) return null;
  const raw = vx > 0 ? "RIGHT" : "LEFT";
  if (!MIRRORED_VIEW) return raw;
  return raw === "RIGHT" ? "LEFT" : "RIGHT";
}

function indexPointingDir(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  if (!idx) return null;
  if (mid || ring || pinky) return null;
  if (th) return null;

  const vx = lm[8].x - lm[5].x;
  return dirFromVx(vx);
}

function thumbPointingDir(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  if (!th) return null;
  if (idx || mid || ring || pinky) return null;

  const vx = lm[4].x - lm[1].x;
  return dirFromVx(vx);
}

function isTwoFingers(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  return !th && idx && mid && !ring && !pinky;
}

function isThreeFingers(lm) {
  const { th, idx, mid, ring, pinky } = fingerFlags(lm);
  return !th && idx && mid && ring && !pinky;
}

function computeMotionScore() {
  const off = document.createElement("canvas");
  off.width = MOTION_SAMPLE_SIZE;
  off.height = MOTION_SAMPLE_SIZE;

  const octx = off.getContext("2d", { willReadFrequently: true });
  octx.drawImage(video, 0, 0, off.width, off.height);

  const img = octx.getImageData(0, 0, off.width, off.height);

  let score = 0;
  if (lastFrameSmall) {
    for (let i = 0; i < img.data.length; i += 16) {
      score += Math.abs(img.data[i] - lastFrameSmall.data[i]);
    }
    score /= (img.data.length / 16);
  }

  lastFrameSmall = img;
  return score;
}

function stabilizeCommand(rawCommand) {
  if (rawCommand === lastRawCommand) {
    sameCount++;
  } else {
    lastRawCommand = rawCommand;
    sameCount = 1;
  }

  if (sameCount >= STABLE_FRAMES) {
    currentStableCommand = rawCommand;
  }

  return currentStableCommand;
}

function recognizeCommand(lm) {
  if (isOpenPalm(lm)) return "Avanzar";
  if (isFist(lm)) return "Retroceder";
  if (isOKSign(lm)) return "Detener";

  const idxDir = indexPointingDir(lm);
  if (idxDir === "RIGHT") return "Vuelta derecha";
  if (idxDir === "LEFT") return "Vuelta izquierda";

  const thDir = thumbPointingDir(lm);
  if (thDir === "RIGHT") return "90° derecha";
  if (thDir === "LEFT") return "90° izquierda";

  if (isThreeFingers(lm)) return "360° derecha";
  if (isTwoFingers(lm)) return "360° izquierda";

  return "Orden no reconocida";
}

// ===================== Main Loop =====================
async function loop() {
  if (!running) return;

  const now = performance.now();

  if (suspended) {
    const motion = computeMotionScore();
    debugText.textContent = `Movimiento: ${motion.toFixed(1)} (umbral ${MOTION_WAKE_THRESHOLD})`;

    if (motion > MOTION_WAKE_THRESHOLD) {
      setModeUI(false);
      setStatus("Despierto: reanudando reconocimiento...");
      lastGoodCommandAt = now;
      setCommand("—");
      lastRawCommand = null;
      sameCount = 0;
      currentStableCommand = "—";
    }

    requestAnimationFrame(loop);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const result = handLandmarker.detectForVideo(video, now);

  if (result?.landmarks?.length) {
    const lm = result.landmarks[0];

    const rawCommand = recognizeCommand(lm);
    const stableCommand = stabilizeCommand(rawCommand);

    setCommand(stableCommand);

    const { th, idx, mid, ring, pinky } = fingerFlags(lm);
    debugText.textContent =
      `Dedos=${countExtendedFingers(lm)} | th=${+th} idx=${+idx} mid=${+mid} ring=${+ring} pinky=${+pinky} | Raw=${rawCommand} | Estable=${stableCommand}`;

    if (stableCommand !== "Orden no reconocida" && stableCommand !== "—") {
      lastGoodCommandAt = now;
      setStatus("Reconociendo gestos...");
    } else {
      setStatus("Gesto detectado pero no coherente.");
    }
  } else {
    debugText.textContent = "Sin mano detectada.";
  }

  if (now - lastGoodCommandAt > SUSPEND_AFTER_MS) {
    setModeUI(true);
    setStatus("Suspendido por inactividad (sin orden válida).");
  }

  requestAnimationFrame(loop);
}

// ===================== UI handlers =====================
btnStart.addEventListener("click", async () => {
  try {
    btnStart.disabled = true;
    setStatus("Solicitando cámara...");
    await setupCamera();

    if (!handLandmarker) {
      await loadHandLandmarker();
    }

    running = true;
    setModeUI(false);
    lastGoodCommandAt = performance.now();

    lastRawCommand = null;
    sameCount = 0;
    currentStableCommand = "—";

    btnStop.disabled = false;
    btnToggleSuspend.disabled = false;

    setStatus("Reconociendo gestos...");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus("Error: " + (err?.message || err));
    btnStart.disabled = false;
  }
});

btnStop.addEventListener("click", () => {
  running = false;
  setStatus("Detenido.");
  setCommand("—");
  debugText.textContent = "";

  btnStart.disabled = false;
  btnStop.disabled = true;
  btnToggleSuspend.disabled = true;
});

btnToggleSuspend.addEventListener("click", () => {
  if (!running) return;
  setModeUI(!suspended);
  setStatus(suspended ? "Suspendido manualmente." : "Reconociendo gestos...");
});

// ===================== Init =====================
setModeUI(false);
setCommand("—");
setStatus("Listo para iniciar.");