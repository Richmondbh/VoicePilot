// Shapes returned by the FastAPI backend (see backend/app/models.py)

export interface AgentStep {
  step: number
  tool: string
  args: Record<string, unknown>
  status: 'ok' | 'blocked' | 'error'
  result: string
  thought: string | null
}

export interface DocumentsInfo {
  folder: string
  backend: 'ollama' | 'tfidf'
  model: string
  chunks: number
  files: { path: string; chunks: number }[]
}

export interface CommandResponse {
  transcript: string
  intent: string
  target: string | null
  status: 'success' | 'rejected' | 'error'
  message: string
  interpreted_by: string
  prompt_version: string
  duration_ms: number
  mode: 'single' | 'agent'
  steps: AgentStep[]
  sources: string[]
}

export interface Interaction {
  id: number
  timestamp: string
  user_message: string
  intent: string | null
  target: string | null
  status: string | null
  assistant_response: string | null
}

export interface Memory {
  id: number
  timestamp: string
  content: string
}

export interface Health {
  status: string
  llm_provider: string
  llm_model: string
  prompt_version: string
  whisper_model: string
  dry_run: boolean
  platform: string
  agent_enabled?: boolean
}

export const INTENTS = [
  'OPEN_APP',
  'OPEN_FOLDER',
  'WEB_SEARCH',
  'CREATE_NOTE',
  'SAVE_MEMORY',
  'RECALL_MEMORY',
  'SUMMARIZE_CLIPBOARD',
  'UNKNOWN',
] as const
