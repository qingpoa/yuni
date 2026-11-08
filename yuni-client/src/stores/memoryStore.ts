import { create } from 'zustand'
import { appApi } from '../services/api'
import type { MemoryItem, MemoryType, MemoryFilter, MemoryDraft, MemoryPatch, MemoryEntityItem, MemoryCandidateItem } from '../types/domain'

interface MemoryState {
  items: MemoryItem[]
  entities: MemoryEntityItem[]
  candidates: MemoryCandidateItem[]
  nextCursor: string | null
  filter: MemoryType | ''
  criteria: MemoryFilter
  loading: boolean
  saving: boolean
  error: string | null
  load: (filter?: MemoryType | '', criteria?: MemoryFilter) => Promise<void>
  loadMore: () => Promise<void>
  create: (draft: MemoryDraft) => Promise<MemoryItem>
  update: (id: number, patch: MemoryPatch) => Promise<MemoryItem>
  resolveConflict: (id: number, content: string, options?: { action?: 'keep' | 'adopt' | 'coexist'; qualifier?: string; otherQualifier?: string }) => Promise<void>
  decideCandidate: (id: number, decision: 'approve' | 'reject') => Promise<void>
  remove: (id: number) => Promise<void>
  clear: () => Promise<void>
  reset: () => void
}

let loadVersion = 0
export const useMemoryStore = create<MemoryState>((set, get) => ({
  items: [], entities: [], candidates: [], nextCursor: null, filter: '', criteria: {}, loading: false, saving: false, error: null,
  load: async (filter = '', criteria = {}) => {
    const version = ++loadVersion
    set({ loading: true, filter, criteria, error: null })
    try {
      const [page, entities, candidates] = await Promise.all([appApi.memory.list(filter, null, 50, criteria), appApi.memory.entities(), appApi.memory.candidates()])
      if (version === loadVersion) set({ items: page.items, nextCursor: page.nextCursor, entities, candidates })
    } catch (error) { if (version === loadVersion) set({ error: error instanceof Error ? error.message : '记忆加载失败' }) }
    finally { if (version === loadVersion) set({ loading: false }) }
  },
  loadMore: async () => {
    const { nextCursor, filter, criteria } = get()
    if (!nextCursor) return
    const version = loadVersion
    try {
      const page = await appApi.memory.list(filter, nextCursor, 50, criteria)
      if (version === loadVersion) set({ items: [...get().items, ...page.items.filter(item => !get().items.some(old => old.id === item.id))], nextCursor: page.nextCursor })
    } catch (error) { if (version === loadVersion) set({ error: error instanceof Error ? error.message : '加载更多失败' }) }
  },
  create: async draft => {
    set({ saving: true })
    try { const item = await appApi.memory.create(draft); await get().load(get().filter, get().criteria); return item }
    finally { set({ saving: false }) }
  },
  update: async (id, patch) => {
    set({ saving: true })
    try { const updated = await appApi.memory.update(id, patch); await get().load(get().filter, get().criteria); return updated }
    finally { set({ saving: false }) }
  },
  resolveConflict: async (id, content, options) => {
    set({ saving: true })
    try { await appApi.memory.resolveConflict(id, content, options); await get().load(get().filter, get().criteria) }
    finally { set({ saving: false }) }
  },
  decideCandidate: async (id, decision) => {
    set({ saving: true })
    try { await appApi.memory.decideCandidate(id, decision); await get().load(get().filter, get().criteria) }
    finally { set({ saving: false }) }
  },
  remove: async id => { set({ saving: true }); try { await appApi.memory.remove(id); await get().load(get().filter, get().criteria) } finally { set({ saving: false }) } },
  clear: async () => { set({ saving: true }); try { await appApi.memory.clear(); await get().load(get().filter, get().criteria) } finally { set({ saving: false }) } },
  reset: () => { ++loadVersion; set({ items: [], entities: [], candidates: [], nextCursor: null, filter: '', criteria: {}, error: null, loading: false }) },
}))
