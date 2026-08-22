import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // First code-split boundary in this app (2026-08-16): three.js +
        // @react-three/fiber + @react-three/drei back the lazy-loaded
        // CausalityScene3D component (see components/CausalityGraphPanel.tsx).
        // Everything else still ships in the single main bundle as before --
        // this chunk exists so that weight only loads for users who actually
        // reach the 3D causality graph.
        manualChunks(id) {
          if (id.includes('node_modules/three') || id.includes('node_modules/@react-three')) {
            return 'three'
          }
        },
      },
    },
  },
})
