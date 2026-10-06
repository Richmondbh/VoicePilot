import { useState } from 'react'
import type { FormEvent } from 'react'
import { useRecorder } from '../services/useRecorder'

interface Props {
  busy: boolean
  onAudio: (audio: Blob) => void
  onText: (text: string) => void
}

export default function VoiceRecorder({ busy, onAudio, onText }: Props) {
  const { recording, seconds, error, start, stop } = useRecorder()
  const [text, setText] = useState('')

  async function toggle() {
    if (recording) onAudio(await stop())
    else await start()
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const command = text.trim()
    if (!command) return
    onText(command)
    setText('')
  }

  const label = busy ? 'Processing your request…' : recording ? `Listening · ${seconds}s · select to send` : 'Select the microphone to speak'

  return (
    <section className="card recorder" aria-labelledby="recorder-title">
      <div className="recorder-heading">
        <div>
          <p className="eyebrow">COMMAND INPUT</p>
          <h2 id="recorder-title">What should VoicePilot do?</h2>
          <p className="muted">Try a voice command or type one below.</p>
        </div>
      </div>

      <div className="recorder-center">
        <div className="mic-halo">
          <button
            className={`mic ${recording ? 'mic--on' : ''}`}
            type="button"
            onClick={toggle}
            disabled={busy}
            aria-label={recording ? 'Stop and send recording' : 'Start recording'}
          >
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z" fill="currentColor" />
              <path d="M19 11a7 7 0 0 1-14 0M12 18v3m-3 0h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <p className="recording-label" aria-live="polite">{label}</p>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      <form className="text-command" onSubmit={submit}>
        <input
          aria-label="Type a command"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="For example, open my Downloads folder"
          disabled={busy}
        />
        <button className="send-button" type="submit" disabled={busy || !text.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
