import type { CommandResponse } from '../types'

const STATUS_ICON = { success: '✓', rejected: '!', error: '×' }
const EXAMPLES = [
  'Open Visual Studio Code',
  'Show my Downloads folder',
  'Search for FastAPI tutorials',
  'Create a note about my report',
  'Remember that my presentation is Friday',
  'What did I ask you to remember?',
  'Summarize my clipboard',
]

/** Displays the transcript, interpreted intent and execution result. */
export default function ActionCard({ result }: { result: CommandResponse | null }) {
  if (!result) {
    return (
      <section className="card action-card action-card--empty" aria-labelledby="examples-title">
        <h2 id="examples-title">Start with a simple request</h2>
        <p className="empty-intro">VoicePilot supports a focused set of everyday actions.</p>
        <ol className="examples">
          {EXAMPLES.map((example) => <li key={example}>{example}</li>)}
        </ol>
      </section>
    )
  }

  return (
    <section className="card action-card" aria-label="Command result">
      <div className="step">
        <span className="step-label">You said</span>
        <p className="quote">“{result.transcript || '…'}”</p>
      </div>
      <div className="step">
        <span className="step-label">VoicePilot understood</span>
        <div className="chips">
          <span className="chip chip--intent">{result.intent}</span>
          {result.target && <span className="chip">{result.target}</span>}
          <span className="chip chip--muted">
            {result.interpreted_by === 'llm' ? `LLM · prompt ${result.prompt_version}` : result.interpreted_by}
            {' · '}{result.duration_ms} ms
          </span>
        </div>
      </div>
      <div className={`step status--${result.status}`}>
        <span className="step-label">Result</span>
        <p className="message">
          <span className="status-icon" aria-hidden="true">{STATUS_ICON[result.status]}</span>
          <span className="pre">{result.message}</span>
        </p>
      </div>
    </section>
  )
}
