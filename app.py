"""
Local web UI for the RAG system. Runs entirely on localhost, no external calls.

Usage:
    python app.py
    then open http://localhost:5050
"""
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
from rag_core import add_document, answer_question, SUPPORTED_EXTENSIONS, list_sources, delete_document

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Local RAG</title>
<style>
  :root {
    --bg: #101215;
    --panel: #16191d;
    --panel-2: #1c2024;
    --border: #262b31;
    --text: #e6e8eb;
    --text-dim: #8a919b;
    --accent: #4fd1a5;
    --accent-dim: #2c5f4c;
    --you: #5b9bff;
    --radius: 10px;
    --mono: 'SF Mono', 'JetBrains Mono', Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    max-width: 760px;
    margin: 0 auto;
    padding: 32px 20px 60px;
  }
  h1 {
    font-size: 18px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 24px;
    letter-spacing: -0.01em;
  }
  .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 16px;
  }
  .panel h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin: 0 0 12px;
    font-weight: 600;
  }

  .upload-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  input[type=file] {
    font-size: 13px; color: var(--text-dim);
    flex: 1; min-width: 180px;
  }
  input[type=file]::file-selector-button {
    font-family: var(--sans);
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    cursor: pointer;
    margin-right: 10px;
  }
  button {
    font-family: var(--sans);
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s, opacity 0.15s;
  }
  button:hover { background: #35735d; }
  button:disabled { opacity: 0.5; cursor: default; }
  #upload-status {
    font-size: 12px; color: var(--text-dim); margin-top: 8px; min-height: 14px;
  }

  #fileList { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
  #fileList li {
    display: flex; justify-content: space-between; align-items: center;
    font-family: var(--mono); font-size: 12.5px; color: var(--text-dim);
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 6px; padding: 7px 10px;
  }
  #fileList .del {
    background: transparent; border: none; color: #d97757;
    font-size: 12px; padding: 2px 6px; opacity: 0.7;
  }
  #fileList .del:hover { opacity: 1; background: rgba(217,119,87,0.12); }
  #fileList .empty { font-family: var(--sans); font-style: italic; opacity: 0.6; background: none; border: none; padding: 2px 0; }

  #chat {
    height: 420px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 14px;
    padding-right: 4px;
  }
  #chat:empty::before {
    content: "Ask something about your indexed documents.";
    color: var(--text-dim); font-size: 13px; font-style: italic;
  }
  .msg { max-width: 88%; }
  .msg .role {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 4px; font-weight: 600;
  }
  .msg .bubble {
    border-radius: var(--radius);
    padding: 10px 13px;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
  }
  .you { align-self: flex-end; }
  .you .role { color: var(--you); text-align: right; }
  .you .bubble { background: #1a2c4a; border: 1px solid #2a4a7a; }
  .bot { align-self: flex-start; }
  .bot .role { color: var(--accent); }
  .bot .bubble { background: var(--panel-2); border: 1px solid var(--border); }
  .sources {
    font-family: var(--mono); font-size: 11px; color: var(--text-dim);
    margin-top: 6px;
  }

  .thinking .bubble { display: flex; gap: 4px; align-items: center; padding: 12px 13px; }
  .thinking span {
    width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim);
    animation: bounce 1.2s infinite;
  }
  .thinking span:nth-child(2) { animation-delay: 0.15s; }
  .thinking span:nth-child(3) { animation-delay: 0.3s; }
  @keyframes bounce { 0%,60%,100% { transform: translateY(0); opacity: 0.4; } 30% { transform: translateY(-4px); opacity: 1; } }

  .ask-row { display: flex; gap: 8px; margin-top: 14px; }
  input[type=text] {
    flex: 1; font-family: var(--sans); background: var(--panel-2);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; color: var(--text); font-size: 14px;
  }
  input[type=text]:focus { outline: none; border-color: var(--accent-dim); }
  input[type=text]::placeholder { color: var(--text-dim); }
</style>
</head>
<body>

<h1><span class="dot"></span> Local RAG — offline</h1>

<div class="panel">
  <h3>Add documents</h3>
  <div class="upload-row">
    <input type="file" id="fileInput" multiple>
    <button onclick="upload()">Index files</button>
  </div>
  <div id="upload-status"></div>
</div>

<div class="panel">
  <h3>Indexed files</h3>
  <ul id="fileList"></ul>
</div>

<div class="panel">
  <h3>Chat</h3>
  <div id="chat"></div>
  <div class="ask-row">
    <input type="text" id="question" placeholder="Ask something about your docs..." onkeydown="if(event.key==='Enter') ask()">
    <button onclick="ask()" id="askBtn">Ask</button>
  </div>
</div>

<script>
function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

async function upload() {
  const input = document.getElementById('fileInput');
  const files = input.files;
  if (!files.length) return;
  const status = document.getElementById('upload-status');
  status.textContent = 'Indexing...';
  const formData = new FormData();
  for (const f of files) formData.append('files', f);
  const res = await fetch('/upload', { method: 'POST', body: formData });
  const data = await res.json();
  status.textContent = data.message;
  input.value = '';
  loadFileList();
}

async function loadFileList() {
  const res = await fetch('/files');
  const data = await res.json();
  const list = document.getElementById('fileList');
  list.innerHTML = '';
  if (!data.sources.length) {
    list.innerHTML = '<li class="empty">No files indexed yet</li>';
    return;
  }
  data.sources.forEach(name => {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = name;
    const btn = document.createElement('button');
    btn.className = 'del';
    btn.textContent = 'Delete';
    btn.onclick = () => deleteFile(name);
    li.appendChild(label);
    li.appendChild(btn);
    list.appendChild(li);
  });
}

async function deleteFile(name) {
  if (!confirm(`Delete all chunks for "${name}"?`)) return;
  await fetch('/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: name})
  });
  loadFileList();
}

loadFileList();

async function ask() {
  const input = document.getElementById('question');
  const askBtn = document.getElementById('askBtn');
  const q = input.value.trim();
  if (!q) return;
  const chat = document.getElementById('chat');

  chat.insertAdjacentHTML('beforeend', `
    <div class="msg you"><div class="role">You</div><div class="bubble">${escapeHtml(q)}</div></div>
  `);
  input.value = '';
  input.disabled = true;
  askBtn.disabled = true;
  chat.scrollTop = chat.scrollHeight;

  const thinkingId = 'thinking-' + Date.now();
  chat.insertAdjacentHTML('beforeend', `
    <div class="msg bot thinking" id="${thinkingId}">
      <div class="role">Assistant</div>
      <div class="bubble"><span></span><span></span><span></span></div>
    </div>
  `);
  chat.scrollTop = chat.scrollHeight;

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await res.json();
    document.getElementById(thinkingId).remove();
    const sourcesHtml = data.sources.length
      ? `<div class="sources">Sources: ${data.sources.map(escapeHtml).join(', ')}</div>` : '';
    chat.insertAdjacentHTML('beforeend', `
      <div class="msg bot">
        <div class="role">Assistant</div>
        <div class="bubble">${escapeHtml(data.answer)}</div>
        ${sourcesHtml}
      </div>
    `);
  } catch (e) {
    document.getElementById(thinkingId).remove();
    chat.insertAdjacentHTML('beforeend', `
      <div class="msg bot"><div class="role">Assistant</div><div class="bubble">Something went wrong: ${escapeHtml(String(e))}</div></div>
    `);
  } finally {
    input.disabled = false;
    askBtn.disabled = false;
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    indexed, skipped = 0, 0
    tmp_dir = tempfile.mkdtemp()
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        # keep the original filename so it shows up correctly as a source
        safe_name = os.path.basename(f.filename)
        path = os.path.join(tmp_dir, safe_name)
        f.save(path)
        try:
            add_document(path)
            indexed += 1
        finally:
            os.unlink(path)
    return jsonify({"message": f"Indexed {indexed} file(s), skipped {skipped} unsupported."})


@app.route("/files")
def files():
    return jsonify({"sources": list_sources()})


@app.route("/delete", methods=["POST"])
def delete():
    filename = request.json.get("filename", "")
    count = delete_document(filename)
    return jsonify({"deleted_chunks": count})


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    answer, hits = answer_question(question)
    sources = sorted({meta["source"] for _, meta in hits})
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)