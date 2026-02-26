document.addEventListener("DOMContentLoaded", () => {
  /* =====================================================
     🔑 API KEY DESDE MOCKAPI (NO HARDcode)
  ===================================================== */
  const MOCKAPI_URL = "https://698def67aded595c253090f9.mockapi.io/api/v1/apiKey";
  let OPENAI_API_KEY = "";
  let apiKeyPromise = null;
  /* ===================================================== */

  const OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses";

  // Wake Word
  const WAKE_WORD = "alfa";
  const IDLE_MS = 10000;

  // ÚNICAS salidas permitidas (validación)
  const ALLOWED_OUTPUTS = new Set([
    "avanzar",
    "retroceder",
    "detener",
    "vuelta derecha",
    "vuelta izquierda",
    "90° derecha",
    "90° izquierda",
    "360° derecha",
    "360° izquierda",
    "Orden no reconocida",
  ]);

  // UI
  const modePill = document.getElementById("modePill");
  const transcriptEl = document.getElementById("transcript");
  const commandEl = document.getElementById("command");
  const substatusEl = document.getElementById("substatus");

  function safeText(el, text) {
    if (el) el.textContent = text;
  }

  function setMode(text, cls) {
    if (!modePill) return;
    modePill.textContent = text;
    modePill.className = `pill ${cls}`;
  }

  function setSubstatus(text) {
    safeText(substatusEl, text);
  }

  function normalize(text) {
    return String(text || "")
      .toLowerCase()
      .trim()
      .replace(/\s+/g, " ");
  }

  function emitGlobalCommand(command) {
    window.dispatchEvent(new CustomEvent("control:command", {
      detail: { source: "voz", command }
    }));
  }

  /* =====================================================
     ✅ CARGA API KEY DESDE MOCKAPI
  ===================================================== */
  async function loadApiKeyFromMockAPI() {
    if (apiKeyPromise) return apiKeyPromise;

    apiKeyPromise = (async () => {
      try {
        setSubstatus("Cargando credenciales (MockAPI)…");

        const r = await fetch(MOCKAPI_URL, { method: "GET" });
        if (!r.ok) throw new Error(`MockAPI HTTP ${r.status}`);

        const data = await r.json();
        const first = Array.isArray(data) ? data[0] : data;

        const key = first?.apikey;
        if (!key || typeof key !== "string") {
          throw new Error("No se encontró 'apikey' en el primer registro.");
        }

        OPENAI_API_KEY = key.trim();
        setSubstatus("Listo. Escuchando órdenes…");
        return OPENAI_API_KEY;
      } catch (err) {
        setMode("Error", "pill-error");
        setSubstatus(`No pude cargar API Key desde MockAPI: ${err.message}`);
        OPENAI_API_KEY = "";
        return "";
      }
    })();

    return apiKeyPromise;
  }

  loadApiKeyFromMockAPI();

  /* =========================
     SPEECH RECOGNITION
  ========================= */
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    setMode("No compatible", "pill-error");
    setSubstatus("Tu navegador no soporta SpeechRecognition. Prueba en Chrome/Edge.");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = "es-MX";
  recognition.continuous = true;
  recognition.interimResults = false;

  let suspended = false;
  let idleTimer = null;

  function resetIdleTimer() {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      suspended = true;
      setMode("Suspendido", "pill-sleep");
      setSubstatus('Suspendido por inactividad. Di "Alpha" para despertar.');
      safeText(commandEl, "—");
    }, IDLE_MS);
  }

  function safeStart() {
    try { recognition.start(); } catch (_) {}
  }

  recognition.onstart = () => {
    setMode(suspended ? "Suspendido" : "Activo", suspended ? "pill-sleep" : "pill-active");
    setSubstatus(suspended ? 'Esperando "Alpha"...' : "Escuchando órdenes…");
    resetIdleTimer();
  };

  recognition.onerror = (e) => {
    setMode("Error", "pill-error");
    setSubstatus(`Error STT: ${e.error || "desconocido"}`);
  };

  recognition.onend = () => {
    safeStart();
  };

  recognition.onresult = async (event) => {
    const last = event.results[event.results.length - 1];
    const raw = last?.[0]?.transcript?.trim() || "";
    if (!raw) return;

    safeText(transcriptEl, raw);
    resetIdleTimer();

    const lower = normalize(raw);

    // Suspendido: solo wake word
    if (suspended) {
      if (lower.includes(WAKE_WORD)) {
        suspended = false;
        setMode("Activo", "pill-active");
        setSubstatus("Despierto. Escuchando órdenes…");
        resetIdleTimer();
      } else {
        setSubstatus('Suspendido. Di "Alpha" para despertar.');
      }
      return;
    }

    // Activo: si dice wake word, ignora
    if (lower.includes(WAKE_WORD)) {
      setSubstatus("Wake word detectada (activo).");
      return;
    }

    setSubstatus("Procesando con IA…");
    const key = OPENAI_API_KEY || (await loadApiKeyFromMockAPI());
    const cmd = await classifyWithOpenAI(raw, key);

    safeText(commandEl, cmd);
    setSubstatus(cmd === "Orden no reconocida" ? "No se reconoció una orden válida." : "Orden reconocida.");

    // ✅ actualiza la orden global
    emitGlobalCommand(cmd);
  };

  setMode("Activo", "pill-active");
  setSubstatus("Pide permisos del micrófono. Escuchando órdenes…");
  safeStart();

  /* =========================
     OpenAI: Clasificador
  ========================= */
  async function classifyWithOpenAI(text, apiKey) {
    if (!apiKey) {
      setMode("Sin API Key", "pill-error");
      setSubstatus("No hay API Key disponible (MockAPI falló o no respondió).");
      return "Orden no reconocida";
    }

    const system = `
Eres un intérprete de intención para un sistema de control por voz.
Tu misión es leer (o inferir desde una transcripción con errores) la intención del usuario y mapearla al comando de control MÁS ADECUADO.

Debes responder ÚNICAMENTE con EXACTAMENTE UNA de estas opciones (una sola línea y nada más):
avanzar
retroceder
detener
vuelta derecha
vuelta izquierda
90° derecha
90° izquierda
360° derecha
360° izquierda
Orden no reconocida

Criterio general:
- Comprende el significado completo del mensaje, aunque sea una frase larga o rara.
- Reconoce sinónimos, expresiones equivalentes, modismos, y palabras parecidas por errores del micrófono.
- Maneja negaciones y “lo contrario de…”.
  Ejemplo: “haz lo contrario de ir hacia atrás” ⇒ avanzar.
- Si el usuario pide un giro con ángulo, elige 90° o 360° según corresponda.
- Si pide girar sin ángulo específico, usa “vuelta derecha” o “vuelta izquierda”.
- Si pide parar, pausar, frenar o inmovilizar, usa “detener”.
- Si el mensaje contiene varias acciones, elige la acción PRINCIPAL o la primera orden clara.
- Si no hay intención clara o no encaja con el set, responde “Orden no reconocida”.

Prohibido:
- No expliques nada.
- No uses comillas.
- No agregues texto extra.
`.trim();

    try {
      const r = await fetch("https://api.openai.com/v1/responses", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: "gpt-4o-mini",
          input: [
            { role: "system", content: system },
            { role: "user", content: text }
          ],
          temperature: 0
        })
      });

      if (!r.ok) return "Orden no reconocida";

      const data = await r.json();
      const out =
        data?.output_text ||
        data?.output?.[0]?.content?.map(c => c?.text).filter(Boolean).join("") ||
        "";

      const result = String(out).trim();
      return ALLOWED_OUTPUTS.has(result) ? result : "Orden no reconocida";
    } catch {
      return "Orden no reconocida";
    }
  }

  /* =====================================================
     🔊 VOZ EXPLICATIVA DEL SISTEMA (ALFA)
  ===================================================== */
  const infoBtn = document.getElementById("infoVoiceBtn");

  function getBestSpanishVoice() {
    const voices = window.speechSynthesis.getVoices() || [];
    const es = voices.filter(v => (v.lang || "").toLowerCase().startsWith("es"));

    const score = (v) => {
      const n = (v.name || "").toLowerCase();
      let s = 0;
      if (n.includes("natural")) s += 6;
      if (n.includes("google")) s += 5;
      if (n.includes("microsoft")) s += 4;
      if (n.includes("mex") || n.includes("méx")) s += 3;
      if (n.includes("spanish") || n.includes("español")) s += 2;
      return s;
    };

    es.sort((a, b) => score(b) - score(a));
    return es[0] || null;
  }

  function waitForVoices(timeoutMs = 1500) {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis;

      const existing = synth.getVoices();
      if (existing && existing.length) return resolve(existing);

      let done = false;
      const timer = setTimeout(() => {
        if (done) return;
        done = true;
        synth.onvoiceschanged = null;
        resolve(synth.getVoices() || []);
      }, timeoutMs);

      synth.onvoiceschanged = () => {
        if (done) return;
        const v = synth.getVoices();
        if (v && v.length) {
          done = true;
          clearTimeout(timer);
          synth.onvoiceschanged = null;
          resolve(v);
        }
      };

      synth.getVoices();
    });
  }

  async function speakIntro() {
    const synth = window.speechSynthesis;
    await waitForVoices();

    const texto = [
      "Hola. Mi nombre es Alfa.",
      "Soy un programa de control por voz impulsado por inteligencia artificial.",
      "Escucho tus instrucciones desde el micrófono y las interpreto para convertirlas en acciones.",
      "Si no detecto voz durante unos segundos, entro en modo suspendido.",
      "Para despertarme, solo di: Alfa.",
      "Cuando quieras, estoy listo para recibir tus órdenes."
    ].join("  ");

    const msg = new SpeechSynthesisUtterance(texto);
    msg.lang = "es-MX";
    msg.rate = 0.92;
    msg.pitch = 1.05;
    msg.volume = 1;

    const v = getBestSpanishVoice();
    if (v) msg.voice = v;

    synth.cancel();
    setTimeout(() => synth.speak(msg), 80);
  }

  if (infoBtn && "speechSynthesis" in window) {
    infoBtn.addEventListener("click", async () => {
      try { await speakIntro(); } catch (e) { console.warn("TTS error:", e); }
    });
  }
});