import { create } from 'zustand'
import { appApi } from '../services/api'
import type { AllowedRoot, ChatStreamEvent, ToolCall, ToolDecision } from '../types/domain'

interface ToolState {
  calls: ToolCall[]
  roots: AllowedRoot[]
  pending: ToolCall | null
  active: ToolCall | null
  loading: boolean
  deciding: boolean
  load: () => Promise<void>
  handleEvent: (event: Extract<ChatStreamEvent, { type: 'tool_confirmation_required' | 'tool_started' | 'tool_completed' }>) => void
  decide: (decision: ToolDecision) => Promise<void>
  addRoot: (path: string) => Promise<void>
  removeRoot: (id: number) => Promise<void>
  reset: () => void
}

const upsert = (calls: ToolCall[], call: ToolCall) => [call, ...calls.filter((item) => item.id !== call.id)]

export const useToolStore = create<ToolState>((set, get) => ({
  calls: [], roots: [], pending: null, active: null, loading: false, deciding: false,
  load: async () => { set({ loading: true }); try { const [calls, roots] = await Promise.all([appApi.tools.list(), appApi.roots.list()]); set({ calls, roots }) } finally { set({ loading: false }) } },
  handleEvent: (event) => {
    const call = event.toolCall
    if (event.type === 'tool_confirmation_required') set({ calls: upsert(get().calls, call), pending: call })
    else if (event.type === 'tool_started') set({ calls: upsert(get().calls, call), active: call })
    else set({ calls: upsert(get().calls, call), active: null, pending: null })
  },
  decide: async (decision) => { const call = get().pending; if (!call) return; set({ deciding: true }); try { if (decision === 'approve' && call.toolName === 'create_reminder' && 'Notification' in window && Notification.permission === 'default') await Notification.requestPermission(); await appApi.tools.decide(call.id, decision); set({ pending: null }) } finally { set({ deciding: false }) } },
  addRoot: async (path) => { const root = await appApi.roots.add(path); set({ roots: [...get().roots, root] }) },
  removeRoot: async (id) => { await appApi.roots.remove(id); set({ roots: get().roots.filter((root) => root.id !== id) }) },
  reset: () => set({ calls: [], roots: [], pending: null, active: null }),
}))
