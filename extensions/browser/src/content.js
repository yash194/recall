// Content script — injects a "memory" sidebar into ChatGPT / Claude / Gemini.
// Watches for user messages, observes them; on send, queries Recall for
// related memories and prepends them as context.

(function () {
  // Adapter table — selectors per host
  const ADAPTERS = {
    "chat.openai.com": {
      input: "#prompt-textarea",
      sendBtn: 'button[data-testid="send-button"]',
      messageContainer: "[data-message-author-role]",
    },
    "chatgpt.com": {
      input: "#prompt-textarea",
      sendBtn: 'button[data-testid="send-button"]',
      messageContainer: "[data-message-author-role]",
    },
    "claude.ai": {
      input: "div[contenteditable='true']",
      sendBtn: 'button[aria-label="Send Message"]',
      messageContainer: "[data-test-render-count]",
    },
    "gemini.google.com": {
      input: "rich-textarea",
      sendBtn: 'button[aria-label="Send message"]',
      messageContainer: "model-response",
    },
    "perplexity.ai": {
      input: "textarea",
      sendBtn: 'button[aria-label="Submit"]',
      messageContainer: ".prose",
    },
  };

  const host = window.location.hostname;
  const cfg = ADAPTERS[host];
  if (!cfg) return;

  // Sidebar UI
  const sidebar = document.createElement("div");
  sidebar.id = "recall-sidebar";
  sidebar.innerHTML = `
    <div class="recall-header">
      <div class="recall-title">Recall</div>
      <div class="recall-stats" id="recall-stats">—</div>
    </div>
    <div class="recall-search">
      <input id="recall-search-input" placeholder="Search memory…" />
    </div>
    <div class="recall-results" id="recall-results"></div>
  `;
  document.body.appendChild(sidebar);

  function send(msg) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(msg, resolve);
    });
  }

  async function refreshStats() {
    const r = await send({ type: "stats" });
    if (r.ok) {
      const s = r.data;
      document.getElementById("recall-stats").textContent =
        `${s.active_nodes} nodes · ${s.active_edges} edges`;
    }
  }

  async function searchMemory(q) {
    const r = await send({ type: "recall", query: q, mode: "path", k: 5 });
    const list = document.getElementById("recall-results");
    list.innerHTML = "";
    if (!r.ok) {
      list.innerHTML = `<div class="recall-error">${r.error}</div>`;
      return;
    }
    for (const node of r.data.nodes) {
      const el = document.createElement("div");
      el.className = "recall-result";
      el.innerHTML = `
        <div class="recall-role">${node.role || "fact"}</div>
        <div class="recall-text">${escapeHtml(node.text)}</div>
        <button class="recall-insert">insert</button>
      `;
      el.querySelector(".recall-insert").onclick = () => insertIntoInput(node.text);
      list.appendChild(el);
    }
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function insertIntoInput(text) {
    const input = document.querySelector(cfg.input);
    if (!input) return;
    if (input.tagName === "TEXTAREA") {
      input.value = (input.value ? input.value + "\n\n" : "") + `[memory] ${text}`;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      input.textContent += `\n\n[memory] ${text}`;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  // Search input
  document.getElementById("recall-search-input").addEventListener("input", (e) => {
    const q = e.target.value.trim();
    if (q.length > 2) searchMemory(q);
  });

  // Auto-observe sent messages
  document.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey && document.activeElement && document.activeElement.matches(cfg.input)) {
      const input = document.querySelector(cfg.input);
      const text = input.value || input.textContent;
      if (text && text.length > 10) {
        await send({ type: "observe", user_msg: text, source: host });
        setTimeout(refreshStats, 1000);
      }
    }
  });

  refreshStats();
  setInterval(refreshStats, 30000);
})();
