import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { useRecorder } from '../services/useRecorder'
import { INTENTS } from '../types'

const SUGGESTIONS: Record<string, string[]> = {
  OPEN_APP: ['Open Chrome', 'Start Visual Studio Code', 'Launch Calculator'],
  OPEN_FOLDER: ['Open my Downloads', 'Show the Documents folder', 'Take me to my desktop'],
  WEB_SEARCH: ['Search for Python tutorials', 'Google FastAPI examples', 'Look up the weather'],
  CREATE_NOTE: ['Create a note saying buy milk', 'Write down that the meeting moved to three'],
  SAVE_MEMORY: ['Remember that my meeting is at ten', 'I have class tomorrow'],
  RECALL_MEMORY: ['When is my meeting?', 'What did I tell you earlier?'],
  SUMMARIZE_CLIPBOARD: ['Summarize my clipboard', 'Summarize what I copied'],
  UNKNOWN: ['Tell me a joke', 'Shut down the computer', 'Order a pizza'],
}

export default function DatasetRecorder() {
  const { recording, seconds, error, start, stop } = useRecorder()
  const [label, setLabel] = useState<string>('OPEN_APP')
  const [target, setTarget] = useState('')
  const [count, setCount] = useState(0)
  const [message, setMessage] = useState('')

  useEffect(() => {
    api.sampleCount().then((response) => setCount(response.count)).catch(() => {})
  }, [])

  async function toggle() {
    if (!recording) return start()
    const audio = await stop()
    try {
      await api.saveSample(audio, label, target, '')
      setCount((current) => current + 1)
      setMessage(`Sample saved as ${label}${target ? ` · ${target}` : ''}.`)
    } catch (cause) {
      setMessage(`Could not save the sample: ${(cause as Error).message}`)
    }
  }

  return (
    <section className="card dataset" aria-labelledby="dataset-title">
      <p className="eyebrow">RAW VOICE DATA</p>
      <h2 id="dataset-title">Collect command examples</h2>
      <p className="muted">
        Record your own commands with an intent label. Samples are stored in <code>data/raw/</code>;
        run <code>python scripts/preprocess.py</code> to transcribe, clean and split them.
      </p>
      <p className="sample-count"><strong>{count}</strong><span>recorded {count === 1 ? 'sample' : 'samples'}</span></p>

      <div className="row">
        <label>
          Intent label
          <select value={label} onChange={(event) => setLabel(event.target.value)} disabled={recording}>
            {INTENTS.map((intent) => <option key={intent} value={intent}>{intent}</option>)}
          </select>
        </label>
        <label>
          Target (optional)
          <input
            value={target}
            onChange={(event) => setTarget(event.target.value)}
            placeholder="App, folder, search or note"
            disabled={recording}
          />
        </label>
      </div>

      <div className="suggestions">
        <strong>Phrase ideas</strong>
        {SUGGESTIONS[label].map((suggestion) => `“${suggestion}”`).join(' · ')}
      </div>
      <button className={`record-btn ${recording ? 'record-btn--on' : ''}`} type="button" onClick={toggle}>
        {recording ? `Stop and save · ${seconds}s` : 'Record a sample'}
      </button>
      {error && <p className="error" role="alert">{error}</p>}
      {message && <p className="dataset-message" role="status">{message}</p>}
    </section>
  )
}
