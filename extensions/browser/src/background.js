// Recall background service worker.
// Talks to the local Recall server (http://localhost:8765 by default) or to
// the Recall hosted endpoint, whichever is configured.

const DEFAULT_ENDPOINT = "http://localhost:8765";

async function getEndpoint() {
  const r = await chrome.storage.local.get("endpoint");
  return r.endpoint || DEFAULT_ENDPOINT;
}

async function getTenant() {
  const r = await chrome.storage.local.get("tenant");
  return r.tenant || "browser_user";
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    const ep = await getEndpoint();
    const tenant = await getTenant();
    if (msg.type === "observe") {
      const r = await fetch(`${ep}/v1/memory/observe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant,
          user_msg: msg.user_msg,
          agent_msg: msg.agent_msg || "",
          scope: { source: msg.source || "chat" },
        }),
      });
      const data = await r.json();
      sendResponse({ ok: true, data });
    } else if (msg.type === "recall") {
      const r = await fetch(`${ep}/v1/memory/recall`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant, query: msg.query, mode: msg.mode || "path", k: msg.k || 5,
        }),
      });
      const data = await r.json();
      sendResponse({ ok: true, data });
    } else if (msg.type === "stats") {
      const r = await fetch(`${ep}/v1/memory/stats?tenant=${tenant}`);
      const data = await r.json();
      sendResponse({ ok: true, data });
    } else {
      sendResponse({ ok: false, error: "unknown_msg_type" });
    }
  })().catch((err) => sendResponse({ ok: false, error: String(err) }));
  return true;
});
