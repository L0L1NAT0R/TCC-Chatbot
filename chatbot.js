let brochures = [];

fetch("brochures.json")
  .then((res) => res.json())
  .then((data) => {
    brochures = data;
  });

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(sender, message, cssClass) {
  const div = document.createElement("div");
  div.className = cssClass;
  div.innerHTML = `<strong>${sender}:</strong> ${message}`;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

sendBtn.addEventListener("click", async () => {
  const input = userInput.value.trim();
  if (!input) return;

  appendMessage("คุณ", input, "user");
  userInput.value = "";

  try {
    const res = await fetch("http://localhost:5000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: input }),
    });

    const data = await res.json();
    appendMessage("บอท", data.reply, "bot");
  } catch (err) {
    appendMessage("บอท", "ขออภัย ระบบมีปัญหาในการประมวลผล", "bot");
  }
});
