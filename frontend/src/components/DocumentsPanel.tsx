import { useState } from 'react'
import type { DocumentsInfo } from '../types'

interface Props {
  info: DocumentsInfo | null
  onReindex: () => Promise<void>
}

/** Files the agent can search (read-only), and how they were indexed. */
export default function DocumentsPanel({ info, onReindex }: Props) {
  const [busy, setBusy] = useState(false)

  async function reindex() {
    setBusy(true)
    try {
      await onReindex()
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card" aria-labelledby="documents-title">
      <div className="card-head">
        <div>
          <p className="eyebrow">READ-ONLY SEARCH</p>
          <h2 id="documents-title">Documents</h2>
        </div>
        <button className="link link--action" type="button" onClick={reindex} disabled={busy}>
          {busy ? 'Indexing…' : 'Re-index'}
        </button>
      </div>
      {!info ? (
        <p className="muted">Loading documents…</p>
      ) : info.files.length === 0 ? (
        <p className="muted">
          Add .pdf, .docx, .txt or .md files to the <code>documents</code> folder, then select Re-index.
        </p>
      ) : (
        <>
          <ul className="documents">
            {info.files.map((file) => (
              <li key={file.path}>
                <span className="doc-name">{file.path.split(/[\\/]/).pop()}</span>
                <span className="doc-meta">{file.chunks} chunks</span>
              </li>
            ))}
          </ul>
          <p className="muted doc-backend">
            {info.backend === 'ollama'
              ? `Semantic search · ${info.model}`
              : 'Keyword search (TF-IDF) · start Ollama with nomic-embed-text for semantic search'}
          </p>
        </>
      )}
    </section>
  )
}
