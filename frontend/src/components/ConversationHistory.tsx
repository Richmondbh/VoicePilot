import type { Interaction } from '../types'

interface Props {
  items: Interaction[]
  onClear: () => void
}

export default function ConversationHistory({ items, onClear }: Props) {
  return (
    <section className="card" aria-labelledby="history-title">
      <div className="card-head">
        <div>
          <p className="eyebrow">RECENT ACTIVITY</p>
          <h2 id="history-title">Conversation history</h2>
        </div>
        {items.length > 0 && (
          <button className="link" type="button" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {items.length === 0 ? (
        <p className="muted">Your recent requests will appear here.</p>
      ) : (
        <ol className="history">
          {[...items].reverse().map((item) => (
            <li key={item.id}>
              <div className="bubble user">{item.user_message}</div>
              <div className={`bubble bot status--${item.status}`}>
                <small>
                  {item.intent || 'UNKNOWN'}
                  {item.target ? ` · ${item.target}` : ''} · {item.timestamp.slice(11, 16)}
                </small>
                <span className="pre">{item.assistant_response}</span>
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
