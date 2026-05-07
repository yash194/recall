# Recall Browser Extension

Manifest V3 Chrome extension that adds Recall typed-edge memory across:
- ChatGPT / chat.openai.com / chatgpt.com
- Claude (claude.ai)
- Gemini (gemini.google.com)
- Perplexity (perplexity.ai)

## Architecture

- **content.js** — injects a sidebar into the host page, watches send events.
- **background.js** — service worker; proxies to Recall HTTP server.
- **popup** — settings UI for endpoint + tenant.
- **Storage** — none locally; talks to Recall server (default http://localhost:8765).

## Local development

1. Run the Recall server:
   ```
   cd recall/
   pip install -e ".[server]"
   uvicorn recall.server:app --port 8765
   ```

2. Load the unpacked extension at chrome://extensions → "Load unpacked" → this directory.

3. Visit ChatGPT/Claude. The Recall sidebar appears on the right.

## Distribution

- Chrome Web Store via `npm run zip` (TODO — manifest is ready).
- Firefox via Manifest V3 polyfill.

This is a v0 scaffold; production work needed:
- DOM selectors break weekly on ChatGPT — needs a maintenance loop
- IndexedDB local cache for offline mode
- Encryption + sync for cross-device tenant
- Auto-extract entities from messages before observe()
