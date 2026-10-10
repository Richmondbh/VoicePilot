import type { AgentStep } from '../types'

const STATUS_LABEL = { ok: 'Done', blocked: 'Blocked', error: 'Error' }

function formatArgs(args: Record<string, unknown>) {
  const values = Object.values(args).filter((value) => value !== '' && value !== null && value !== undefined)
  return values.length ? values.map(String).join(', ') : ''
}

/** Step-by-step view of what the agent did: tool, input, result, and whether Python allowed it. */
export default function AgentTrace({ steps, sources }: { steps: AgentStep[]; sources: string[] }) {
  return (
    <div className="step agent-trace">
      <span className="step-label">Agent steps</span>
      <ol className="trace-list">
        {steps.map((step) => (
          <li key={step.step} className={`trace-item trace-item--${step.status}`}>
            <div className="trace-head">
              <span className="trace-number" aria-hidden="true">{step.step}</span>
              <code>{step.tool}</code>
              {formatArgs(step.args) && <span className="trace-args">“{formatArgs(step.args)}”</span>}
              <span className={`trace-status trace-status--${step.status}`}>{STATUS_LABEL[step.status]}</span>
            </div>
            {step.thought && <p className="trace-thought">{step.thought}</p>}
            <p className="trace-result pre">{step.result}</p>
          </li>
        ))}
      </ol>
      {sources.length > 0 && (
        <div className="sources">
          <span className="step-label">Sources</span>
          <div className="chips">
            {sources.map((source) => (
              <span key={source} className="chip chip--source">{source}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
