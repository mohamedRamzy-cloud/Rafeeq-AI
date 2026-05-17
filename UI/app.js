const chatBox = document.getElementById("chat-box");
const input = document.getElementById("input");
const hero = document.getElementById("hero");
const historyBox = document.getElementById("history");
const debugPanel = document.getElementById("debug-panel");
const debugContent = document.getElementById("debug-content");

// =============================
// STATE
// =============================
let ws = null;
let botMessage = null;
let isStreaming = false;
let hasStartedChat = false;

// =============================
// SESSION
// =============================
let sessionId = createSessionId();
saveSession();

// =============================
// CREATE SESSION
// =============================
function createSessionId() {
    return (
        "session_" +
        Date.now() +
        "_" +
        Math.random().toString(36).slice(2, 8)
    );
}

// =============================
// SAVE SESSION
// =============================
function saveSession() {
    localStorage.setItem("session_id", sessionId);
}

// =============================
// DEBUG LOG
// =============================
function debugLog(message) {
    console.log(message);
    if (!debugContent) return;
    const line = document.createElement("div");
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    debugContent.appendChild(line);
    debugContent.scrollTop = debugContent.scrollHeight;
}

// =============================
// TOGGLE DEBUG
// =============================
function toggleDebug() {
    if (!debugPanel) return;
    debugPanel.style.display =
        debugPanel.style.display === "block" ? "none" : "block";
}

// =============================
// CONNECT WS
// =============================
function connectWS() {
    return new Promise((resolve, reject) => {

        if (ws && ws.readyState === WebSocket.OPEN) {
            return resolve(ws);
        }

        const protocol = location.protocol === "https:" ? "wss" : "ws";
        const wsUrl = `${protocol}://${location.hostname}:8000/chat/ws`;

        debugLog("Connecting WS...");
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            debugLog("WS Connected ✔");
            resolve(ws);
        };

        ws.onmessage = (event) => {
            const raw = String(event.data || "");
            if (!raw) return;

            try {
                const data = JSON.parse(raw);

                // =============================
                // PING
                // =============================
                if (data.type === "ping") {
                    ws.send(JSON.stringify({ type: "pong" }));
                    return;
                }

                // =============================
                // STREAM FINISHED
                // =============================
                if (data.type === "done") {
                    isStreaming = false;
                    if (botMessage) {
                        botMessage.classList.remove("typing");
                    }
                    botMessage = null;
                    return;
                }

                // =============================
                // STREAM CHUNKS
                // =============================
                if (data.type === "chunk" && botMessage) {
                    botMessage.insertAdjacentText("beforeend", data.content || "");
                    scrollBottom();
                }

                // =============================
                // ERROR
                // =============================
                if (data.type === "error") {
                    isStreaming = false;
                    if (botMessage) {
                        botMessage.classList.remove("typing");
                        botMessage.textContent = "حدث خطأ، حاول مرة أخرى";
                    }
                    botMessage = null;
                    debugLog("WS Error from server: " + data.error);
                }

            } catch (err) {
                console.error("Invalid WS JSON:", err);
                debugLog("Invalid WS JSON ❌");
            }
        };

        ws.onerror = (err) => {
            debugLog("WS Error ❌");
            console.error(err);
            isStreaming = false;
            reject(err);
        };

        ws.onclose = () => {
            debugLog("WS Closed ⚠");
            ws = null;
            isStreaming = false;
        };
    });
}

// =============================
// SEND MESSAGE
// =============================
async function sendMessage() {
    const text = input.value.trim();
    if (!text || isStreaming) return;

    // =============================
    // FIRST MESSAGE UI TRANSITION
    // =============================
    if (!hasStartedChat) {
        startChatLayout();
        hasStartedChat = true;
    }

    input.value = "";
    addMessage("user", text);
    addHistory(text);

    botMessage = addMessage("bot", "");
    botMessage.classList.add("typing");
    isStreaming = true;

    try {
        const socket = await connectWS();
        socket.send(JSON.stringify({
            question: text,
            session_id: sessionId
        }));
    } catch (err) {
        console.error(err);
        isStreaming = false;
        if (botMessage) {
            botMessage.classList.remove("typing");
            botMessage.textContent = "حدث خطأ في الاتصال بالسيرفر";
        }
    }
}

// =============================
// START CHAT LAYOUT
// =============================
function startChatLayout() {
    if (!hero) return;
    hero.classList.add("hidden");
    chatBox.classList.add("active-chat");
    debugLog("Chat layout activated ✔");
}

// =============================
// ADD MESSAGE
// =============================
function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.dir = "auto";
    div.textContent = text;
    chatBox.appendChild(div);
    scrollBottom();
    return div;
}

// =============================
// SCROLL
// =============================
function scrollBottom() {
    requestAnimationFrame(() => {
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}

// =============================
// NEW CHAT
// =============================
function newChat() {
    sessionId = createSessionId();
    saveSession();
    chatBox.innerHTML = "";
    hero.classList.remove("hidden");
    chatBox.classList.remove("active-chat");
    hasStartedChat = false;
    input.focus();
    debugLog("New chat created ✔");

    
    fetch(`http://${location.hostname}:8000/chat/new`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
    }).catch(err => console.warn("New chat notify failed:", err));
}

// =============================
// HISTORY
// =============================
function addHistory(text) {
    if (!historyBox) return;
    const item = document.createElement("div");
    item.className = "history-item";
    item.textContent = text.slice(0, 45);
    historyBox.prepend(item);
}

// =============================
// THEME
// =============================
function toggleTheme() {
    document.body.classList.toggle("light");
    debugLog("Theme toggled");
}

// =============================
// VOICE
// =============================
function startVoice() {
    alert("ميزة الصوت قريباً");
}

// =============================
// ENTER KEY
// =============================
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
    }
});

// =============================
// AUTO FOCUS
// =============================
window.onload = () => {
    input.focus();
    debugLog("UI Ready");
};