let brochures = [];

fetch("brochures.json")
  .then((res) => res.json())
  .then((data) => {
    brochures = data;
  });

function appendMessage(sender, message, cssClass) {
  const div = document.createElement("div");
  div.className = cssClass;
  div.innerHTML = `<strong>${sender}:</strong> ${message}`;
  document.getElementById("chat-messages").appendChild(div);
  document.getElementById("chat-messages").scrollTop = document.getElementById("chat-messages").scrollHeight;
}

// Inject HTML with the image on top being the two toggle images
document.body.insertAdjacentHTML("beforeend", `
  <img id="mascot-collapsed" src="mascot-closed.png" alt="Mascot">
  <div id="chat-fab">
    <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor">
      <path d="M4 4h16v12H5.17L4 17.17V4zM2 2v20l4-4h14a2 2 0 0 0 2-2V2H2z"/>
    </svg>
  </div>
  <img id="mascot-expanded" src="mascot-open.png" alt="Mascot">
  <div id="chat-box">
    <div id="chat-messages"></div>
    <div class="chat-input-area">
      <input type="text" id="user-input" placeholder="พิมพ์ข้อความที่นี่...">
      <button id="send-btn">ส่ง</button>
    </div>
  </div>
`);

// Initialize mascot visibility
document.getElementById("mascot-expanded").style.display = "none";

function setChatOpen(isOpen) {
  const box = document.getElementById("chat-box");
  box.style.display = isOpen ? "flex" : "none";
  document.getElementById("mascot-collapsed").style.display = isOpen ? "none" : "block";
  document.getElementById("mascot-expanded").style.display = isOpen ? "block" : "none";
  localStorage.setItem("chatOpen", isOpen ? "true" : "false");
}

document.getElementById("chat-fab").addEventListener("click", () => {
  const isOpen = document.getElementById("chat-box").style.display === "flex";
  setChatOpen(!isOpen);
});

// On page load, restore previous state
window.addEventListener("DOMContentLoaded", () => {
  const isOpen = localStorage.getItem("chatOpen") === "true";
  setChatOpen(isOpen);
});

// 🌐 Smart BASE_URL detection
const isLocal = location.hostname === "localhost" || location.hostname === "127.0.0.1";
const BASE_URL = isLocal
  ? "http://localhost:5000"
  : "https://tcc-chatbot.onrender.com";

document.getElementById("send-btn").addEventListener("click", async () => {
  const userInput = document.getElementById("user-input");
  const input = userInput.value.trim();
  if (!input) return;

  appendMessage("คุณ", input, "user");
  userInput.value = "";

  // Show typing indicator
  const typingIndicator = document.createElement("div");
  typingIndicator.id = "bot-typing";
  typingIndicator.className = "bot";
  typingIndicator.innerText = "บอทกำลังพิมพ์...";
  document.getElementById("chat-messages").appendChild(typingIndicator);
  document.getElementById("chat-messages").scrollTop = document.getElementById("chat-messages").scrollHeight;

  try {
    const res = await fetch(`${BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),
    });

    const data = await res.json();

    const typingEl = document.getElementById("bot-typing");
    if (typingEl) typingEl.remove();

    appendMessage("บอท", data.reply, "bot");
  } catch (err) {
    console.error("❌ Fetch failed:", err);
    const typingEl = document.getElementById("bot-typing");
    if (typingEl) typingEl.remove();

    appendMessage("บอท", "ขออภัย ระบบมีปัญหาในการประมวลผล", "bot");
  }
});

// Bind Enter key to Send
document.getElementById("user-input").addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault(); // prevent newline
    document.getElementById("send-btn").click();
  }
});
