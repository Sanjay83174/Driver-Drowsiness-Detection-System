const tabLive = document.getElementById("tab-live");
const tabUpload = document.getElementById("tab-upload");
const liveSection = document.getElementById("live-section");
const uploadSection = document.getElementById("upload-section");

const videoEl = document.getElementById("live-video");
const canvasEl = document.getElementById("live-canvas");
const startBtn = document.getElementById("start-live");
const stopBtn = document.getElementById("stop-live");

const liveStateEl = document.getElementById("live-state");
const liveProbEl = document.getElementById("live-prob");
const liveDurationEl = document.getElementById("live-duration");
const liveWarningEl = document.getElementById("live-warning");

const fileInput = document.getElementById("file-input");
const analyzeBtn = document.getElementById("analyze-file");
const uploadStatusEl = document.getElementById("upload-status");
const uploadResultEl = document.getElementById("upload-result");
const uploadPreviewImg = document.getElementById("upload-preview-image");

let mediaStream = null;
let captureIntervalId = null;
let drowsyStartTime = null;
let lastState = "idle";

// ---- Tab switching ----
tabLive.addEventListener("click", () => {
  tabLive.classList.add("active");
  tabUpload.classList.remove("active");
  liveSection.classList.add("active");
  uploadSection.classList.remove("active");
});

tabUpload.addEventListener("click", () => {
  tabUpload.classList.add("active");
  tabLive.classList.remove("active");
  uploadSection.classList.add("active");
  liveSection.classList.remove("active");
});

// ---- Live camera handling ----
async function startCamera() {
  if (mediaStream) return;

  // Unlock alarm audio on this user gesture so beep can play when drowsy is detected later.
  if (!drowsyAlarmBeepUrl) drowsyAlarmBeepUrl = createBeepWavUrl();
  const alarmEl = document.getElementById("drowsy-alarm");
  if (alarmEl) {
    alarmEl.src = drowsyAlarmBeepUrl;
    // Prime the audio element silently so future .play() calls are allowed.
    alarmEl.volume = 0;
    alarmEl.load();
    const p = alarmEl.play();
    if (p && typeof p.then === "function") {
      p
        .then(() => {
          alarmEl.pause();
          alarmEl.currentTime = 0;
          alarmEl.volume = 0.8;
        })
        .catch(() => {
          // If play is blocked, just restore volume; later plays may still work.
          alarmEl.volume = 0.8;
        });
    } else {
      alarmEl.volume = 0.8;
    }
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    videoEl.srcObject = mediaStream;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    liveWarningEl.textContent = "Camera running. Stay centered in the frame.";
    liveStateEl.textContent = "Analyzing...";
    liveStateEl.className = "badge neutral";

    // Wait for video to be ready
    videoEl.onloadedmetadata = () => {
      videoEl.play();
      canvasEl.width = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
    };

    // Capture frame every 500ms
    captureIntervalId = setInterval(captureAndSendFrame, 500);
  } catch (err) {
    console.error(err);
    liveWarningEl.textContent = "Could not access camera. Check permissions.";
  }
}

function stopCamera() {
  if (captureIntervalId) {
    clearInterval(captureIntervalId);
    captureIntervalId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;
  liveStateEl.textContent = "Idle";
  liveStateEl.className = "badge neutral";
  liveProbEl.textContent = "-";
  liveDurationEl.textContent = "0.0 s";
  liveWarningEl.textContent = "Camera stopped.";
  lastState = "idle";
  drowsyStartTime = null;
  stopAlarm();
}

async function captureAndSendFrame() {
  if (!mediaStream || videoEl.readyState < 2) return;

  const ctx = canvasEl.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  canvasEl.toBlob(
    async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");

      try {
        const res = await fetch("/api/predict_image", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          console.error("Prediction error", await res.text());
          return;
        }

        const data = await res.json();
        const prob = data.drowsy_probability ?? 0;
        const state = data.state || "awake";

        liveProbEl.textContent = (prob * 100).toFixed(1) + "%";
        updateLiveStateFromProbability(prob, state);
      } catch (e) {
        console.error("Network error", e);
      }
    },
    "image/jpeg",
    0.7
  );
}

// ---- Alarm Logic ----

// Create a beep sound (0.5s beep + 0.5s silence) so we can loop it.
function createBeepWavUrl() {
  const sampleRate = 44100;
  const beepDesc = 0.5; // seconds
  const silenceDesc = 0.5; // seconds
  const duration = beepDesc + silenceDesc;
  const freq = 1000;
  const numSamples = Math.floor(sampleRate * duration);
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = numChannels * (bitsPerSample / 8);
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  const beepSamples = Math.floor(sampleRate * beepDesc);
  for (let i = 0; i < numSamples; i++) {
    let sample = 0;
    if (i < beepSamples) {
      const t = i / sampleRate;
      // Simple sine wave
      sample = Math.sin(2 * Math.PI * freq * t) * 0.5 * 32767;
    }
    view.setInt16(44 + i * 2, sample, true);
  }

  return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
}

let drowsyAlarmBeepUrl = null;
let isAlarmPlaying = false;

function startAlarm() {
  if (isAlarmPlaying) return;

  const audio = document.getElementById("drowsy-alarm");
  if (!audio) return;

  if (!drowsyAlarmBeepUrl) {
    drowsyAlarmBeepUrl = createBeepWavUrl();
  }

  audio.src = drowsyAlarmBeepUrl;
  audio.loop = true;
  audio.volume = 1.0;

  const p = audio.play();
  if (p && typeof p.catch === "function") {
    p.catch((e) => console.warn("Alarm play failed:", e));
  }
  isAlarmPlaying = true;
}

function stopAlarm() {
  if (!isAlarmPlaying) return;

  const audio = document.getElementById("drowsy-alarm");
  if (audio) {
    audio.pause();
    audio.currentTime = 0;
  }
  isAlarmPlaying = false;
}

function updateLiveStateFromProbability(prob, modelState) {
  const now = performance.now();
  // Use a slightly higher threshold for live camera so normal blinking
  // does not immediately trigger drowsiness.
  const isDrowsy = prob >= 0.8;

  if (isDrowsy) {
    if (drowsyStartTime === null) {
      drowsyStartTime = now;
    }
    const durationSec = (now - drowsyStartTime) / 1000;
    liveDurationEl.textContent = durationSec.toFixed(1) + " s";

    if (durationSec >= 4) {
      liveStateEl.textContent = "Drowsy";
      liveStateEl.className = "badge danger";
      liveWarningEl.textContent =
        "Eyes likely closed or yawning continuously for more than 3–4 seconds. Please take a break!";

      // TRIGGER ALARM CONTINUOUSLY
      startAlarm();

    } else if (durationSec >= 2) {
      liveStateEl.textContent = "Becoming drowsy";
      liveStateEl.className = "badge warn";
      liveWarningEl.textContent =
        "Signs of drowsiness detected. Watch for long blinks or yawning.";

      // Stop alarm if we dipped back from "Drowsy" to "Becoming drowsy" (unlikely but safe)
      stopAlarm();
    }
  } else {
    drowsyStartTime = null;
    liveDurationEl.textContent = "0.0 s";
    liveStateEl.textContent = "Awake";
    liveStateEl.className = "badge safe";
    liveWarningEl.textContent = "You appear awake. Eyes are open and facial expressions look normal.";

    // Stop alarm immediately
    stopAlarm();
  }

  lastState = isDrowsy ? "drowsy" : "awake";
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

// ---- Upload image / video ----
analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  uploadResultEl.textContent = "";

  if (!file) {
    uploadStatusEl.textContent = "Please choose an image or video file.";
    return;
  }

  const isVideo = file.type.startsWith("video/");
  const isImage = file.type.startsWith("image/");

  if (!isVideo && !isImage) {
    uploadStatusEl.textContent = "Unsupported file type. Use image/* or video/*.";
    return;
  }

  uploadStatusEl.textContent = "Analyzing...";

  // Show preview for images; hide for videos.
  if (isImage && uploadPreviewImg) {
    uploadPreviewImg.src = URL.createObjectURL(file);
    uploadPreviewImg.classList.remove("hidden");
  } else if (uploadPreviewImg) {
    uploadPreviewImg.src = "";
    uploadPreviewImg.classList.add("hidden");
  }

  try {
    const formData = new FormData();
    if (isImage) {
      formData.append("image", file);
      const res = await fetch("/api/predict_image", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        uploadStatusEl.textContent = "Error from server while analyzing image.";
        return;
      }
      const data = await res.json();
      const prob = data.drowsy_probability ?? 0;
      const state = data.state || "awake";

      uploadStatusEl.textContent = "Image analyzed.";

      let interpretation;
      if (state === "drowsy") {
        interpretation =
          "High drowsy probability: eyes may be closing or facial activity suggests fatigue/yawning.";
      } else {
        interpretation =
          "Low drowsy probability: eyes appear open and the face looks awake/attentive.";
      }

      const stateBadgeClass = state === "drowsy" ? "danger" : "safe";
      uploadResultEl.innerHTML =
        `State: <span class="badge ${stateBadgeClass}">${state.toUpperCase()}</span><br>` +
        `Drowsy probability (Accuracy): ${(prob * 100).toFixed(1)}%<br>` +
        `Interpretation: ${interpretation}`;
    } else if (isVideo) {
      formData.append("video", file);
      const res = await fetch("/api/predict_video", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        uploadStatusEl.textContent = "Error from server while analyzing video.";
        return;
      }
      const data = await res.json();
      uploadStatusEl.textContent = "Video analyzed.";

      const state = data.overall_state || "awake";
      const avgProb = data.average_drowsy_probability ?? 0;
      const segments = data.drowsy_segments || [];

      const stateBadgeClass = state === "drowsy" ? "danger" : "safe";
      let text =
        `Overall state: <span class="badge ${stateBadgeClass}">${state.toUpperCase()}</span><br>` +
        `Average drowsy probability: ${(avgProb * 100).toFixed(1)}%<br>`;

      if (segments.length === 0) {
        text += "No continuous drowsy segments longer than 3–4 seconds were detected.";
      } else {
        text += "Drowsy segments (approximate times in seconds):<br>";
        for (const seg of segments) {
          text += `&nbsp;&nbsp;- from ${seg.start.toFixed(1)}s to ${seg.end.toFixed(1)}s<br>`;
        }
      }

      uploadResultEl.innerHTML = text;
    }
  } catch (err) {
    console.error(err);
    uploadStatusEl.textContent = "Network or server error during analysis.";
  }
});

// Show preview immediately when a new image is selected (before Analyze).
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file || !uploadPreviewImg) return;

  const isImage = file.type.startsWith("image/");
  if (isImage) {
    uploadPreviewImg.src = URL.createObjectURL(file);
    uploadPreviewImg.classList.remove("hidden");
  } else {
    uploadPreviewImg.src = "";
    uploadPreviewImg.classList.add("hidden");
  }

  uploadStatusEl.textContent = "";
  uploadResultEl.textContent = "";
});

