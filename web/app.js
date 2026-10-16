/**
 * GourAI V1 — Frontend Application Controller
 * Handles SSE streaming from FastAPI backend, chat message flow,
 * playground completion, telemetry meters, and sampling presets.
 */

// Application State
const state = {
  currentMode: 'chat', // 'chat' | 'playground'
  temperature: 0.8,
  topK: 40,
  topP: 0.9,
  maxTokens: 120,
  repetitionPenalty: 1.1,
  activeReader: null,
  isGenerating: false,
  chatHistory: "",
};

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  fetchModelInfo();
});

/**
 * Fetch and display model runtime info & specs from /api/info
 */
async function fetchModelInfo() {
  const statusBadge = document.getElementById("status-badge");
  const statusText = document.getElementById("status-text");
  const deviceBadge = document.getElementById("device-badge");
  const metricDevice = document.getElementById("metric-device");

  try {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error("API status " + res.status);
    const data = await res.json();

    statusText.innerText = "Model Online";
    deviceBadge.innerText = data.device.toUpperCase();
    metricDevice.innerText = data.device.toUpperCase();

    // Populate sidebar specs
    if (data.parameter_string) {
      document.getElementById("spec-params").innerText = `${data.parameters.toLocaleString()} (~${data.parameter_string})`;
    }
    if (data.n_layer) {
      document.getElementById("spec-layers").innerText = data.n_layer;
    }
    if (data.n_head) {
      document.getElementById("spec-heads").innerText = `${data.n_head} (${data.n_embd / data.n_head} dim/head)`;
    }
    if (data.n_embd) {
      document.getElementById("spec-embd").innerText = data.n_embd;
    }
    if (data.block_size) {
      document.getElementById("spec-context").innerText = `${data.block_size} tokens`;
    }
  } catch (err) {
    console.warn("Could not reach /api/info:", err);
    statusText.innerText = "Offline / Local";
    statusBadge.style.color = "#f43f5e";
    statusBadge.style.borderColor = "rgba(244, 63, 94, 0.3)";
    document.querySelector(".status-dot").style.background = "#f43f5e";
  }
}

/**
 * Switch view modes between Chat and Playground
 */
function switchMode(mode) {
  state.currentMode = mode;
  document.querySelectorAll(".nav-tab").forEach(tab => tab.classList.remove("active"));
  document.querySelectorAll(".view-mode").forEach(view => view.classList.remove("active"));

  if (mode === "chat") {
    document.getElementById("tab-chat").classList.add("active");
    document.getElementById("chat-container").classList.add("active");
  } else {
    document.getElementById("tab-playground").classList.add("active");
    document.getElementById("playground-container").classList.add("active");
  }
}

/**
 * Handle Enter Key in Chat Input
 */
function handleChatKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}

/**
 * Send Quick Chip prompt into Chat
 */
function sendChipPrompt(promptText) {
  const chatInput = document.getElementById("chat-input");
  chatInput.value = promptText;
  sendChatMessage();
}

/**
 * Send user message in Chat mode and stream response
 */
async function sendChatMessage() {
  const inputElem = document.getElementById("chat-input");
  const text = inputElem.value.trim();
  if (!text || state.isGenerating) return;

  inputElem.value = "";
  appendUserMessage(text);

  // Build conversational prompt
  const prompt = `${state.chatHistory}\nUser: ${text}\nGourAI: `;

  // Create assistant bubble placeholder with streaming cursor
  const botBubble = appendBotPlaceholder();
  const textContainer = botBubble.querySelector(".bubble-text");

  let fullReply = "";
  const startTime = performance.now();
  let tokenCount = 0;

  state.isGenerating = true;

  try {
    await streamEndpoint(
      prompt,
      state.maxTokens,
      // On each token chunk
      (token) => {
        tokenCount++;
        fullReply += token;
        textContainer.innerText = fullReply;

        // Auto-scroll chat to bottom
        const chatMessages = document.getElementById("chat-messages");
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Update live telemetry
        updateTelemetry(startTime, tokenCount);
      },
      // On completion
      () => {
        state.isGenerating = false;
        // Clean history
        const cleanReply = fullReply.split("\nUser:")[0].trim();
        state.chatHistory += `\nUser: ${text}\nGourAI: ${cleanReply}`;
        if (state.chatHistory.length > 300) {
          state.chatHistory = state.chatHistory.slice(-300);
        }
      }
    );
  } catch (err) {
    console.error("Chat streaming error:", err);
    textContainer.innerText += "\n[Error: Connection interrupted]";
    state.isGenerating = false;
  }
}

/**
 * Append user message bubble to chat
 */
function appendUserMessage(text) {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble user-message";
  bubble.innerHTML = `
    <div class="bubble-avatar">👤</div>
    <div class="bubble-content">
      <div class="bubble-author">You</div>
      <div class="bubble-text">${escapeHtml(text)}</div>
    </div>
  `;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

/**
 * Append bot message placeholder bubble
 */
function appendBotPlaceholder() {
  const container = document.getElementById("chat-messages");
  const bubble = document.createElement("div");
  bubble.className = "message-bubble bot-message";
  bubble.innerHTML = `
    <div class="bubble-avatar">🍛</div>
    <div class="bubble-content">
      <div class="bubble-author">GourAI Assistant</div>
      <div class="bubble-text"></div>
    </div>
  `;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

/**
 * Run Text Completion in Playground Mode
 */
async function runPlaygroundGenerate() {
  const promptInput = document.getElementById("prompt-input");
  const prompt = promptInput.value;
  if (!prompt || state.isGenerating) return;

  const promptEcho = document.getElementById("prompt-echo");
  const streamedTokens = document.getElementById("streamed-tokens");
  const generateBtn = document.getElementById("generate-btn");
  const stopBtn = document.getElementById("stop-btn");
  const tokenCounter = document.getElementById("token-counter");

  promptEcho.innerText = prompt;
  streamedTokens.innerText = "";
  tokenCounter.innerText = "0 tokens";

  generateBtn.disabled = true;
  stopBtn.disabled = false;
  state.isGenerating = true;

  const startTime = performance.now();
  let tokenCount = 0;

  try {
    await streamEndpoint(
      prompt,
      state.maxTokens,
      // Token arrival
      (token) => {
        tokenCount++;
        streamedTokens.innerText += token;
        tokenCounter.innerText = `${tokenCount} tokens`;

        const outputDisplay = document.getElementById("output-display");
        outputDisplay.scrollTop = outputDisplay.scrollHeight;

        updateTelemetry(startTime, tokenCount);
      },
      // Stream finish
      () => {
        generateBtn.disabled = false;
        stopBtn.disabled = true;
        state.isGenerating = false;
      }
    );
  } catch (err) {
    console.error("Playground generation error:", err);
    generateBtn.disabled = false;
    stopBtn.disabled = true;
    state.isGenerating = false;
  }
}

/**
 * Universal SSE Stream consumer
 */
async function streamEndpoint(prompt, maxTokens, onToken, onComplete) {
  const response = await fetch("/api/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: prompt,
      max_new_tokens: maxTokens,
      temperature: state.temperature,
      top_k: state.topK,
      top_p: state.topP,
      repetition_penalty: state.repetitionPenalty,
    }),
  });

  if (!response.ok) {
    throw new Error(`Server returned HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  state.activeReader = reader;
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const payload = JSON.parse(line.slice(6));
          if (payload.token) {
            onToken(payload.token);
          }
          if (payload.done) {
            break;
          }
        } catch (e) {
          // ignore parse errors on partial chunk
        }
      }
    }
  }

  state.activeReader = null;
  if (onComplete) onComplete();
}

/**
 * Stop active generation
 */
function stopGeneration() {
  if (state.activeReader) {
    state.activeReader.cancel();
    state.activeReader = null;
    state.isGenerating = false;

    document.getElementById("generate-btn").disabled = false;
    document.getElementById("stop-btn").disabled = true;
  }
}

/**
 * Update Telemetry Meters
 */
function updateTelemetry(startTime, tokenCount) {
  const elapsedSec = (performance.now() - startTime) / 1000;
  const speed = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : "0.0";

  document.getElementById("metric-speed").innerHTML = `${speed} <span class="unit">tok/s</span>`;
  document.getElementById("metric-tokens").innerText = tokenCount;
  document.getElementById("metric-time").innerHTML = `${elapsedSec.toFixed(1)} <span class="unit">s</span>`;
}

/**
 * Slider Update Handler
 */
function updateParam(param, val) {
  const num = parseFloat(val);
  if (param === "temp") {
    state.temperature = num;
    document.getElementById("temp-val").innerText = num.toFixed(2);
  } else if (param === "topk") {
    state.topK = parseInt(val, 10);
    document.getElementById("topk-val").innerText = state.topK;
  } else if (param === "topp") {
    state.topP = num;
    document.getElementById("topp-val").innerText = num.toFixed(2);
  } else if (param === "tokens") {
    state.maxTokens = parseInt(val, 10);
    document.getElementById("tokens-val").innerText = state.maxTokens;
  }
}

/**
 * Apply Sampling Preset
 */
function applyPreset(name) {
  document.querySelectorAll(".preset-btn").forEach(btn => btn.classList.remove("active"));
  event.currentTarget.classList.add("active");

  if (name === "creative") {
    setSlider("temp", 1.1);
    setSlider("topk", 50);
    setSlider("topp", 0.95);
  } else if (name === "balanced") {
    setSlider("temp", 0.8);
    setSlider("topk", 40);
    setSlider("topp", 0.90);
  } else if (name === "precise") {
    setSlider("temp", 0.3);
    setSlider("topk", 20);
    setSlider("topp", 0.80);
  } else if (name === "greedy") {
    setSlider("temp", 0.0);
    setSlider("topk", 0);
    setSlider("topp", 1.0);
  }
}

function setSlider(param, val) {
  if (param === "temp") {
    document.getElementById("temp-slider").value = val;
    updateParam("temp", val);
  } else if (param === "topk") {
    document.getElementById("topk-slider").value = val;
    updateParam("topk", val);
  } else if (param === "topp") {
    document.getElementById("topp-slider").value = val;
    updateParam("topp", val);
  }
}

/**
 * Clear Chat
 */
function clearChat() {
  document.getElementById("chat-messages").innerHTML = `
    <div class="message-bubble bot-message">
      <div class="bubble-avatar">🍛</div>
      <div class="bubble-content">
        <div class="bubble-author">GourAI Assistant</div>
        <div class="bubble-text">Chat history reset. How may I serve thee, noble friend?</div>
      </div>
    </div>
  `;
  state.chatHistory = "";
}

/**
 * Clear Playground Output
 */
function clearPlayground() {
  document.getElementById("prompt-echo").innerText = "";
  document.getElementById("streamed-tokens").innerText = "";
  document.getElementById("token-counter").innerText = "0 tokens";
}

/**
 * Architecture Modal Controls
 */
function openArchModal() {
  document.getElementById("arch-modal").classList.add("open");
}

function closeArchModal(e) {
  if (!e || e.target.id === "arch-modal" || e.target.classList.contains("modal-close-btn")) {
    document.getElementById("arch-modal").classList.remove("open");
  }
}

/**
 * Sanitize HTML helper
 */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.innerText = str;
  return div.innerHTML;
}
