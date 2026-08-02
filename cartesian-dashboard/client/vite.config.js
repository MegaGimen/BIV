import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/BIV/',
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
    proxy: {
      '/BIV/api': {
        target: 'http://127.0.0.1:3033',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/BIV/, ''),
      },
      '/api': {
        target: 'http://127.0.0.1:3033',
        changeOrigin: true,
      },
    },
  },
})
