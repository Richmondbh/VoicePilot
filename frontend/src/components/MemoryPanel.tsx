import type { Memory } from '../types'

interface Props {
  memories: Memory[]
  onDelete: (id: number) => void
}

export default function MemoryPanel({ memories, onDelete }: Props) {
  return (
    <section className="card" aria-labelledby="memory-title">
      <div className="card-head">
        <h2 id="memory-title">Saved memory</h2>
        <span className="badge">{memories.length} {memories.length === 1 ? 'item' : 'items'}</span>
      </div>
      {memories.length === 0 ? (
        <p className="muted">Say “Remember that…” to save a detail for later.</p>
      ) : (
        <ul className="memories">
          {memories.map((memory) => (
            <li key={memory.id}>
              <span>{memory.content}</span>
              <button
                className="link"
                type="button"
                onClick={() => onDelete(memory.id)}
                aria-label={`Forget ${memory.content}`}
                title="Remove saved memory"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
