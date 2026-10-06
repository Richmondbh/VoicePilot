// Small wrapper around the Fetch API.
// Fetch + FormData reference: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
import type { CommandResponse, Health, Interaction, Memory } from '../types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/health'),

  sendText: (text: string) =>
    request<CommandResponse>('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    }),

  sendVoice: (audio: Blob) => {
    const form = new FormData()
    form.append('audio', audio, 'command.webm')
    return request<CommandResponse>('/api/voice', { method: 'POST', body: form })
  },

  history: () => request<Interaction[]>('/api/history'),
  clearHistory: () => request('/api/history', { method: 'DELETE' }),
  memories: () => request<Memory[]>('/api/memories'),
  deleteMemory: (id: number) => request(`/api/memories/${id}`, { method: 'DELETE' }),

  saveSample: (audio: Blob, label: string, target: string, promptText: string) => {
    const form = new FormData()
    form.append('audio', audio, 'sample.webm')
    form.append('label', label)
    form.append('target', target)
    form.append('prompt_text', promptText)
    return request<{ ok: boolean; id: string }>('/api/dataset/sample', { method: 'POST', body: form })
  },
  sampleCount: () => request<{ count: number }>('/api/dataset/count'),
}
