import type { Health } from '../types'

interface Props {
  health: Health | null
  offline: boolean
  tts: boolean
  onToggleTts: () => void
}

export default function StatusIndicator({ health, offline, tts, onToggleTts }: Props) {
  const connectionLabel = offline ? 'Backend offline' : health ? 'System ready' : 'Connecting'
  const connectionClass = offline ? 'dot--off' : health ? 'dot--on' : 'dot--pending'

  return (
    <section className="status-bar" aria-label="System status">
      <span className="connection">
        <span className={`dot ${connectionClass}`} aria-hidden="true" />
        {connectionLabel}
      </span>
      {offline ? (
        <span className="badge">Start the API with uvicorn app.main:app</span>
      ) : health ? (
        <>
          <span className="badge">LLM · {health.llm_provider === 'none' ? 'offline classifier' : `${health.llm_provider} · ${health.llm_model}`}</span>
          <span className="badge">Whisper · {health.whisper_model}</span>
          {health.dry_run && <span className="badge badge--warn">Dry run</span>}
        </>
      ) : (
        <span className="badge">Checking local services…</span>
      )}
      <label className="tts-toggle">
        <input type="checkbox" checked={tts} onChange={onToggleTts} />
        <span className="switch" aria-hidden="true" />
        <span>Speak replies</span>
      </label>
    </section>
  )
}
