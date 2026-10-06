// Vite config. The proxy forwards /api calls to the FastAPI backend so the
// browser sees one origin. Docs: https://vitejs.dev/config/server-options.html#server-proxy
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
