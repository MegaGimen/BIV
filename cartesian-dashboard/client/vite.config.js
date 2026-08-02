import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/BIV/',
  plugins: [react()],
  server: {
    allowedHosts: ['llinker.com', 'www.llinker.com', '8.216.48.82'],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:3033',
        changeOrigin: true,
      }
    }
  }
})
