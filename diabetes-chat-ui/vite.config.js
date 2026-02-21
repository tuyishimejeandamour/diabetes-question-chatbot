import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api calls to the local Ollama server to avoid CORS issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/ollama': {
        target: 'http://localhost:11434',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ollama/, ''),
      },
    },
  },
})
