# Diabetes Medical Chatbot — React + Ollama (Local UI)

A React chat interface that talks to a **Gemma 3 1B** diabetes model running locally via [Ollama](https://ollama.com).

---

## Prerequisites

| Tool | Install |
|------|---------|
| [Node.js ≥ 18](https://nodejs.org) | `brew install node` / official installer |
| [Ollama](https://ollama.com/download) | Download from ollama.com |

---

## Quick start

### 1 — Set up the Ollama model

**Option A — Base Gemma (easiest):**
```bash
ollama pull gemma3:1b
ollama create diabetes-chatbot -f Modelfile
```

**Option B — Your fine-tuned weights (best results):**

First, complete training in the notebook and merge the adapter (`MERGE_MODEL = True`), then convert:
```bash
# clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
pip install -r requirements.txt

# convert merged HF model → GGUF
python convert_hf_to_gguf.py ../gemma3-diabetes-chatbot/merged-model \
       --outtype q4_k_m --outfile gemma3-diabetes-q4.gguf

cd ..
# update the FROM line in Modelfile to point to the .gguf file, then:
ollama create diabetes-chatbot -f Modelfile
```

### 2 — Start Ollama (keep this terminal open)
```bash
ollama serve
```

### 3 — Run the React app
```bash
cd diabetes-chat-ui
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Project structure

```
diabetes-chat-ui/
├── src/
│   ├── App.jsx        # Chat UI with streaming
│   ├── App.css        # Styling
│   ├── main.jsx       # React entry point
│   └── index.css
├── Modelfile          # Ollama model definition
├── index.html
├── vite.config.js     # Proxies /ollama → localhost:11434
└── package.json
```

## Configuration

Click the **⚙️** button in the app header to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| Model name | `diabetes-chatbot` | Any model visible in `ollama list` |
| Temperature | `0.7` | Higher = more creative |
| Max tokens | `256` | Maximum response length |

---

## Disclaimer

> This chatbot is **for educational purposes only** and must not be used as a substitute
> for professional medical advice, diagnosis, or treatment.
