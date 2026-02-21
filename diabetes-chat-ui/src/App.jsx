import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const SYSTEM_PROMPT =
  'You are a knowledgeable and compassionate diabetes medical assistant. ' +
  'Answer the user\'s questions accurately and concisely using evidence-based ' +
  'medical information. Always recommend consulting a healthcare professional ' +
  'for personalised medical advice.'

const DEFAULT_MODEL = 'diabetes-chatbot'

const EXAMPLE_QUESTIONS = [
  'What are the early symptoms of type 2 diabetes?',
  'How should I manage blood sugar when sick?',
  'What foods should a diabetic avoid?',
  'What is HbA1c and what does my result mean?',
  'How does exercise affect blood glucose?',
  'What should I include in a diabetes emergency kit?',
]

const WELCOME_MESSAGE = {
  role: 'assistant',
  content:
    "Hello! I'm your **Diabetes Medical Assistant** 🩺\n\n" +
    'I can help you with questions about:\n' +
    '• Blood sugar management\n' +
    '• Diabetes symptoms & diagnosis\n' +
    '• Nutrition & diet\n' +
    '• Medication & insulin\n' +
    '• Lifestyle & prevention\n\n' +
    '*⚠️ For educational purposes only — always consult a healthcare professional.*',
}

// ── Simple inline markdown renderer ─────────────────────────────────────────
function renderContent(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br/>')
}

// ── Single chat message bubble ───────────────────────────────────────────────
function Message({ msg, isStreaming }) {
  return (
    <div className={`msg-row ${msg.role}`}>
      <div className="avatar">{msg.role === 'user' ? '👤' : '🩺'}</div>
      <div className="bubble">
        {msg.content ? (
          <span dangerouslySetInnerHTML={{ __html: renderContent(msg.content) }} />
        ) : isStreaming ? (
          <span className="cursor-blink">▍</span>
        ) : null}
      </div>
    </div>
  )
}

// ── Connection status badge ──────────────────────────────────────────────────
function StatusBadge({ status }) {
  const labels = { idle: 'Ready', streaming: 'Generating…', error: 'Error' }
  return <span className={`status-badge ${status}`}>{labels[status]}</span>
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([WELCOME_MESSAGE])
  const [input, setInput] = useState('')
  const [model, setModel] = useState(DEFAULT_MODEL)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(256)
  const [status, setStatus] = useState('idle')
  const [showSettings, setShowSettings] = useState(false)

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async () => {
    const text = input.trim()
    if (!text || status === 'streaming') return

    const history = [
      ...messages,
      { role: 'user', content: text },
    ]
    setMessages([...history, { role: 'assistant', content: '' }])
    setInput('')
    setStatus('streaming')

    abortRef.current = new AbortController()

    try {
      const res = await fetch('/ollama/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          model,
          stream: true,
          options: { temperature, num_predict: maxTokens },
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            ...history.map((m) => ({ role: m.role, content: m.content })),
          ],
        }),
      })

      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText)
        throw new Error(`Ollama returned ${res.status}: ${errText}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        for (const line of decoder.decode(value).split('\n')) {
          if (!line.trim()) continue
          try {
            const data = JSON.parse(line)
            const token = data?.message?.content ?? ''
            if (token) {
              setMessages((prev) => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                  role: 'assistant',
                  content: updated[updated.length - 1].content + token,
                }
                return updated
              })
            }
          } catch {
            // skip partial/malformed JSON lines
          }
        }
      }

      setStatus('idle')
    } catch (err) {
      if (err.name === 'AbortError') {
        setStatus('idle')
        return
      }
      setMessages((prev) => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'assistant',
          content:
            `❌ **Could not reach Ollama.**\n\n${err.message}\n\n` +
            '**Checklist:**\n' +
            '• Run `ollama serve` in a terminal\n' +
            '• Run `ollama list` to confirm the model exists\n' +
            '• See the README for setup instructions',
        }
        return updated
      })
      setStatus('error')
    } finally {
      inputRef.current?.focus()
    }
  }, [input, messages, model, temperature, maxTokens, status])

  const stopGeneration = () => {
    abortRef.current?.abort()
    setStatus('idle')
  }

  const clearChat = () => {
    abortRef.current?.abort()
    setMessages([WELCOME_MESSAGE])
    setStatus('idle')
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <span className="header-logo">🩺</span>
          <div>
            <h1>Diabetes Medical Chatbot</h1>
            <p>Running locally via Ollama · Model: <code>{model}</code></p>
          </div>
        </div>
        <div className="header-actions">
          <StatusBadge status={status} />
          <button className="icon-btn" title="Clear chat" onClick={clearChat}>🗑️</button>
          <button
            className={`icon-btn ${showSettings ? 'active' : ''}`}
            title="Settings"
            onClick={() => setShowSettings((s) => !s)}
          >
            ⚙️
          </button>
        </div>
      </header>

      <div className="layout">
        {/* ── Settings sidebar ── */}
        {showSettings && (
          <aside className="sidebar">
            <h3>Settings</h3>

            <label className="setting-label">
              Ollama model name
              <input
                className="setting-input"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder="diabetes-chatbot"
              />
            </label>

            <label className="setting-label">
              Temperature <span className="setting-val">{temperature}</span>
              <input
                type="range" min="0" max="1" step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>

            <label className="setting-label">
              Max tokens <span className="setting-val">{maxTokens}</span>
              <input
                type="range" min="64" max="512" step="32"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </label>

            <div className="sidebar-section">
              <strong>Quick commands</strong>
              <pre className="code-block">ollama serve</pre>
              <pre className="code-block">ollama list</pre>
              <pre className="code-block">ollama run {model}</pre>
            </div>
          </aside>
        )}

        {/* ── Chat area ── */}
        <main className="chat-area">
          {/* Example prompts (shown only at the start) */}
          {messages.length === 1 && (
            <div className="examples">
              <p className="examples-label">Try asking:</p>
              <div className="examples-grid">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    className="example-chip"
                    onClick={() => { setInput(q); inputRef.current?.focus() }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="messages">
            {messages.map((msg, i) => (
              <Message
                key={i}
                msg={msg}
                isStreaming={status === 'streaming' && i === messages.length - 1}
              />
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input bar */}
          <div className="input-bar">
            <textarea
              ref={inputRef}
              className="input-textarea"
              value={input}
              rows={2}
              placeholder="Ask a diabetes question… (Enter to send, Shift+Enter for newline)"
              disabled={status === 'streaming'}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            {status === 'streaming' ? (
              <button className="send-btn stop" onClick={stopGeneration} title="Stop">⏹</button>
            ) : (
              <button
                className="send-btn"
                disabled={!input.trim()}
                onClick={sendMessage}
                title="Send"
              >
                ➤
              </button>
            )}
          </div>
        </main>
      </div>

      <footer className="footer">
        ⚠️ For educational purposes only — not a substitute for professional medical advice.
      </footer>
    </div>
  )
}
