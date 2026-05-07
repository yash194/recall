async function init() {
  const r = await chrome.storage.local.get(["endpoint", "tenant"]);
  document.getElementById("endpoint").value = r.endpoint || "http://localhost:8765";
  document.getElementById("tenant").value = r.tenant || "browser_user";
  await refreshStats();
}

document.getElementById("save").onclick = async () => {
  await chrome.storage.local.set({
    endpoint: document.getElementById("endpoint").value,
    tenant: document.getElementById("tenant").value,
  });
  await refreshStats();
};

async function refreshStats() {
  chrome.runtime.sendMessage({ type: "stats" }, (r) => {
    const el = document.getElementById("stats");
    if (r && r.ok) {
      const s = r.data;
      el.textContent = `${s.active_nodes} nodes · ${s.active_edges} edges`;
    } else {
      el.textContent = `(could not reach server)`;
    }
  });
}

init();
