import { useCallback, useEffect, useState } from 'react'
import ActionCard from './components/ActionCard'
import ConversationHistory from './components/ConversationHistory'
import DatasetRecorder from './components/DatasetRecorder'
import DocumentsPanel from './components/DocumentsPanel'
import MemoryPanel from './components/MemoryPanel'
import StatusIndicator from './components/StatusIndicator'
import VoiceRecorder from './components/VoiceRecorder'
import { api } from './services/api'
import type { CommandResponse, DocumentsInfo, Health, Interaction, Memory } from './types'

function speak(text: string) {
  if (!('speechSynthesis' in window)) return
  window.speechSynthesis.cancel()
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text.replace(/[•*_#]/g, '')))
}

export default function App() {
  const [tab, setTab] = useState<'assistant' | 'dataset'>('assistant')
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<CommandResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<Interaction[]>([])
  const [memories, setMemories] = useState<Memory[]>([])
  const [documents, setDocuments] = useState<DocumentsInfo | null>(null)
  const [tts, setTts] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [nextHealth, nextHistory, nextMemories] = await Promise.all([
        api.health(),
        api.history(),
        api.memories(),
      ])
      setHealth(nextHealth)
      setHistory(nextHistory)
      setMemories(nextMemories)
      setOffline(false)
    } catch {
      setOffline(true)
    }
  }, [])

  useEffect(() => {
    void refresh()
    api.documents().then(setDocuments).catch(() => setDocuments(null))
  }, [refresh])

  async function run(command: () => Promise<CommandResponse>) {
    setBusy(true)
    setError(null)
    try {
      const response = await command()
      setResult(response)
      if (tts) speak(response.message)
      await refresh()
    } catch (cause) {
      setError((cause as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const isAssistant = tab === 'assistant'

  return (
    <div className="app-shell">
      <header className="masthead">
        <a className="brand" href="#top" aria-label="VoicePilot home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M16 5.5v12M11 10v7.5a5 5 0 0 0 10 0V10" />
              <path d="M7 16.5a9 9 0 0 0 18 0M16 25.5v3M12.5 28.5h7" />
            </svg>
          </span>
          <span className="brand-copy">
            <strong>VoicePilot</strong>
            <small>DESKTOP VOICE ASSISTANT</small>
          </span>
        </a>

        <nav className="primary-nav" role="tablist" aria-label="VoicePilot sections">
          <button
            className={isAssistant ? 'active' : ''}
            id="assistant-tab"
            type="button"
            role="tab"
            aria-selected={isAssistant}
            aria-controls="assistant-panel"
            onClick={() => setTab('assistant')}
          >
            Assistant
          </button>
          <button
            className={!isAssistant ? 'active' : ''}
            id="dataset-tab"
            type="button"
            role="tab"
            aria-selected={!isAssistant}
            aria-controls="dataset-panel"
            onClick={() => setTab('dataset')}
          >
            Dataset lab
          </button>
        </nav>
      </header>

      <section className="welcome-row" id="top">
        <div>
          <p className="eyebrow">{isAssistant ? 'VOICE CONTROL · 01' : 'COMMAND DATA · 02'}</p>
          <h1>{isAssistant ? 'A little less clicking.' : 'Build a better command set.'}</h1>
          <p className="welcome-copy">
            {isAssistant
              ? 'Say what you need. VoicePilot interprets the request, checks it, then acts.'
              : 'Capture labelled examples to explore how VoicePilot recognises everyday commands.'}
          </p>
        </div>
        <div className="welcome-note">
          <span className="note-mark" aria-hidden="true">✳</span>
          <span>AI suggests.<br /><strong>Python decides.</strong></span>
        </div>
      </section>

      <StatusIndicator
        health={health}
        offline={offline}
        tts={tts}
        onToggleTts={() => setTts((enabled) => !enabled)}
      />

      {isAssistant ? (
        <main
          className="workspace-grid"
          id="assistant-panel"
          role="tabpanel"
          aria-labelledby="assistant-tab"
        >
          <div className="main-column">
            <VoiceRecorder
              busy={busy}
              onAudio={(audio) => void run(() => api.sendVoice(audio))}
              onText={(text) => void run(() => api.sendText(text))}
            />
            {error && <p className="card error" role="alert">{error}</p>}
            <ActionCard result={result} />
          </div>
          <aside className="side-column" aria-label="Assistant information">
            <DocumentsPanel
              info={documents}
              onReindex={async () => setDocuments(await api.reindexDocuments())}
            />
            <MemoryPanel
              memories={memories}
              onDelete={async (id) => {
                await api.deleteMemory(id)
                await refresh()
              }}
            />
            <ConversationHistory
              items={history}
              onClear={async () => {
                await api.clearHistory()
                await refresh()
              }}
            />
          </aside>
        </main>
      ) : (
        <main id="dataset-panel" role="tabpanel" aria-labelledby="dataset-tab">
          <DatasetRecorder />
        </main>
      )}

      <footer className="app-footer">
        <span>VOICEPILOT · CONTROLLED DESKTOP ACTIONS</span>
        <span>Every action passes backend validation.</span>
      </footer>
    </div>
  )
}
