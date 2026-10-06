// Shapes returned by the FastAPI backend (see backend/app/models.py)

export interface CommandResponse {
  transcript: string
  intent: string
  target: string | null
  status: 'success' | 'rejected' | 'error'
  message: string
  interpreted_by: string
  prompt_version: string
  duration_ms: number
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
