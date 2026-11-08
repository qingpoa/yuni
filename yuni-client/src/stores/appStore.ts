import { create } from 'zustand'
import { appApi } from '../services/api'
import type { BootstrapData } from '../types/domain'

export type AppStatus = 'booting_tauri' | 'starting_agent' | 'loading_bootstrap' | 'needs_companion' | 'ready' | 'degraded' | 'failed'

interface AppState {
  status: AppStatus
  bootstrapData: BootstrapData | null
  error: string | null
  bootstrap: () => Promise<void>
  setHasCompanion: (hasCompanion: boolean) => void
  setModelConfigured: (configured: boolean) => void
}

export const useAppStore = create<AppState>((set, get) => ({
  status: 'booting_tauri',
  bootstrapData: null,
  error: null,
  bootstrap: async () => {
    set({ status: 'loading_bootstrap', error: null })
    try {
      const data = await appApi.bootstrap()
      set({ bootstrapData: data, status: data.hasCompanion ? 'ready' : 'needs_companion' })
    } catch (error) {
      set({ status: 'failed', error: error instanceof Error ? error.message : '本地 Agent 启动失败' })
    }
  },
  setHasCompanion: (hasCompanion) => {
    const current = get().bootstrapData
    set({ bootstrapData: current ? { ...current, hasCompanion } : current, status: hasCompanion ? 'ready' : 'needs_companion' })
  },
  setModelConfigured: (modelConfigured) => {
    const current = get().bootstrapData
    set({ bootstrapData: current ? { ...current, modelConfigured } : current })
  },
}))
