import { create } from 'zustand'
import { appApi } from '../services/api'
import type { ImageConfigInput, ModelConfigInput, ProactiveSettingsInput, Settings, VoiceConfigInput } from '../types/domain'
import { useAppStore } from './appStore'

interface SettingsState {
  settings: Settings | null
  loading: boolean
  saving: boolean
  load: () => Promise<void>
  setMemoryEnabled: (enabled: boolean) => Promise<void>
  saveProactive: (config: ProactiveSettingsInput) => Promise<void>
  saveModel: (config: ModelConfigInput) => Promise<void>
  clearModelKey: () => Promise<void>
  saveEmbedding: (config: ModelConfigInput) => Promise<void>
  clearEmbeddingKey: () => Promise<void>
  saveImage: (config: ImageConfigInput) => Promise<void>
  clearImageKey: () => Promise<void>
  saveVoice: (config: VoiceConfigInput) => Promise<void>
  clearVoiceKey: () => Promise<void>
  reset: () => void
}

export const useSettingsStore = create<SettingsState>((set) => ({
  settings: null, loading: false, saving: false,
  load: async () => { set({ loading: true }); try { const settings = await appApi.settings.get(); set({ settings }); useAppStore.getState().setModelConfigured(settings.model.configured) } finally { set({ loading: false }) } },
  setMemoryEnabled: async (enabled) => { set((state) => ({ saving: true, settings: state.settings ? { ...state.settings, memoryEnabled: enabled } : state.settings })); try { set({ settings: await appApi.settings.setMemoryEnabled(enabled) }) } catch (error) { set((state) => ({ settings: state.settings ? { ...state.settings, memoryEnabled: !enabled } : state.settings })); throw error } finally { set({ saving: false }) } },
  saveProactive: async (config) => { set({ saving: true }); try { set({ settings: await appApi.settings.saveProactive(config) }) } finally { set({ saving: false }) } },
  saveModel: async (config) => { set({ saving: true }); try { const settings = await appApi.settings.saveModel(config); set({ settings }); useAppStore.getState().setModelConfigured(settings.model.configured) } finally { set({ saving: false }) } },
  clearModelKey: async () => { set({ saving: true }); try { const settings = await appApi.settings.clearModelKey(); set({ settings }); useAppStore.getState().setModelConfigured(false) } finally { set({ saving: false }) } },
  saveEmbedding: async (config) => { set({ saving: true }); try { set({ settings: await appApi.settings.saveEmbedding(config) }) } finally { set({ saving: false }) } },
  clearEmbeddingKey: async () => { set({ saving: true }); try { set({ settings: await appApi.settings.clearEmbeddingKey() }) } finally { set({ saving: false }) } },
  saveImage: async (config) => { set({ saving: true }); try { set({ settings: await appApi.settings.saveImage(config) }) } finally { set({ saving: false }) } },
  clearImageKey: async () => { set({ saving: true }); try { set({ settings: await appApi.settings.clearImageKey() }) } finally { set({ saving: false }) } },
  saveVoice: async (config) => { set({ saving: true }); try { set({ settings: await appApi.settings.saveVoice(config) }) } finally { set({ saving: false }) } },
  clearVoiceKey: async () => { set({ saving: true }); try { set({ settings: await appApi.settings.clearVoiceKey() }) } finally { set({ saving: false }) } },
  reset: () => set({ settings: null }),
}))
