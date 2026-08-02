/// Minimal built-in chat page served at GET /. Same-origin with the API,
/// so no CORS surface; talks to /v1/chat/completions with stream: true.
/// Deliberately a single embedded string: no assets, no bundle lookup,
/// nothing to install.
enum ChatPage {
    static let html = #"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>slipstream chat</title>
<style>
  :root { color-scheme: light dark;
          --bg: #fdfcfa; --fg: #2a2622; --line: #e6e0d8;
          --me: #f0e9df; --bot: #f7f4ef; --accent: #8a6d4a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #191715; --fg: #e8e2da; --line: #33302c;
            --me: #2b2620; --bot: #23201c; --accent: #c9a97e; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.55 -apple-system, "Helvetica Neue", sans-serif;
         display: flex; flex-direction: column; height: 100dvh; }
  header { padding: 10px 16px; border-bottom: 1px solid var(--line);
           display: flex; align-items: baseline; gap: 10px; }
  header b { font-size: 15px; }
  header span { font-size: 12px; opacity: .6; }
  #log { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { max-width: 46em; margin: 0 auto 12px; padding: 10px 14px;
         border-radius: 10px; white-space: pre-wrap; word-wrap: break-word; }
  .me  { background: var(--me); }
  .bot { background: var(--bot); border: 1px solid var(--line); }
  .meta { max-width: 46em; margin: -8px auto 14px; font-size: 11px;
          opacity: .55; padding: 0 14px; }
  form { display: flex; gap: 8px; padding: 12px 16px 16px;
         border-top: 1px solid var(--line); }
  textarea { flex: 1; resize: none; padding: 10px 12px; border-radius: 10px;
             border: 1px solid var(--line); background: var(--bot);
             color: var(--fg); font: inherit; height: 44px; }
  button { padding: 0 18px; border-radius: 10px; border: none;
           background: var(--accent); color: var(--bg); font: inherit;
           cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
</style>
</head>
<body>
<header><b>slipstream</b><span id="mid">loading model id...</span></header>
<div id="log"></div>
<form id="f">
  <textarea id="in" placeholder="Message (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="go" type="submit">Send</button>
</form>
<script>
const log = document.getElementById('log');
const input = document.getElementById('in');
const go = document.getElementById('go');
const midEl = document.getElementById('mid');
let modelID = null;
const history = [];

fetch('/v1/models').then(r => r.json()).then(j => {
  modelID = j.data[0].id;
  midEl.textContent = modelID;
}).catch(() => { midEl.textContent = 'model list unavailable'; });

function bubble(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
function meta(text) {
  const d = document.createElement('div');
  d.className = 'meta';
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

async function send(text) {
  history.push({ role: 'user', content: text });
  bubble('me', text);
  const out = bubble('bot', '');
  go.disabled = true;
  const t0 = performance.now();
  let t1 = null, tokens = 0, acc = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelID, messages: history, stream: true })
    });
    if (!resp.ok || !resp.body) {
      out.textContent = 'error: HTTP ' + resp.status + ' ' + (await resp.text());
      history.pop();
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') continue;
        try {
          const delta = JSON.parse(payload).choices?.[0]?.delta?.content;
          if (delta) {
            if (t1 === null) t1 = performance.now();
            tokens += 1;
            acc += delta;
            out.textContent = acc;
            log.scrollTop = log.scrollHeight;
          }
        } catch (_) {}
      }
    }
    history.push({ role: 'assistant', content: acc });
    const ttft = t1 ? ((t1 - t0) / 1000).toFixed(2) : '?';
    const rate = t1 && tokens > 1
      ? (tokens / ((performance.now() - t1) / 1000)).toFixed(1) : '?';
    meta('ttft ' + ttft + ' s · ~' + rate + ' chunks/s');
  } catch (e) {
    out.textContent = 'error: ' + e;
    history.pop();
  } finally {
    go.disabled = false;
    input.focus();
  }
}

document.getElementById('f').addEventListener('submit', e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || go.disabled) return;
  input.value = '';
  send(text);
});
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('f').requestSubmit();
  }
});
input.focus();
</script>
</body>
</html>
"""#
}
