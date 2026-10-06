// React hook that records microphone audio with the MediaRecorder API.
// Based on MDN: https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Using_the_MediaStream_Recording_API
import { useRef, useState } from 'react'

export function useRecorder() {
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | undefined>(undefined)
  const resolveRef = useRef<((b: Blob) => void) | null>(null)

  async function start() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop()) // release the microphone
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        resolveRef.current?.(blob)
      }
      recorder.start()
      recorderRef.current = recorder
      setSeconds(0)
      timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000)
      setRecording(true)
    } catch {
      setError('Microphone access was denied or no microphone was found.')
    }
  }

  /** Stops recording and resolves with the recorded audio. */
  function stop(): Promise<Blob> {
    return new Promise((resolve) => {
      resolveRef.current = resolve
      window.clearInterval(timerRef.current)
      setRecording(false)
      recorderRef.current?.stop()
    })
  }

  return { recording, seconds, error, start, stop }
}
