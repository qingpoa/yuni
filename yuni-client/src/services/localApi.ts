import type { ApiResponse, AppStreamEvent, ChatStreamEvent, CreatePartnerDraft } from '../types/domain'
import type { AppApi } from './appApi'
import { getAgentConnection } from './connection'
import { parseSse } from './sseParser'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const { baseUrl, session } = getAgentConnection()
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers: { 'Content-Type': 'application/json', 'X-Yuni-Session': session, ...init?.headers } })
  const body = await response.json() as ApiResponse<T>
  if (!response.ok || ![0, 200].includes(body.code)) throw new Error(body.message || `Agent 请求失败（${response.status}）`)
  return body.data as T
}

async function* stream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
  const { baseUrl, session } = getAgentConnection()
  const response = await fetch(`${baseUrl}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', 'X-Yuni-Session': session }, body: JSON.stringify(body), signal })
  for await (const frame of parseSse(response)) yield { type: frame.event, ...(frame.data as object) } as ChatStreamEvent
}

async function upload<T>(path: string, file: File): Promise<T> {
  const form = new FormData(); form.append('file', file)
  const { baseUrl, session } = getAgentConnection()
  const response = await fetch(`${baseUrl}${path}`, { method: 'POST', headers: { 'X-Yuni-Session': session }, body: form })
  const body = await response.json() as ApiResponse<T>
  if (!response.ok || ![0, 200].includes(body.code) || body.data === null) throw new Error(body.message || '上传失败')
  return body.data
}

function partnerBody(draft: CreatePartnerDraft) {
  const { personalityTags, interactionStyles, ...body } = draft
  void personalityTags; void interactionStyles
  return JSON.stringify(body)
}

export const localApi: AppApi = {
  events: {
    async *stream(afterId, signal) {
      const { baseUrl, session } = getAgentConnection()
      const response = await fetch(`${baseUrl}/api/events${afterId == null ? '' : `?afterId=${afterId}`}`, {
        headers: { Accept: 'text/event-stream', 'X-Yuni-Session': session }, signal,
      })
      for await (const frame of parseSse(response)) yield { id: frame.id == null ? null : Number(frame.id), event: { type: frame.event, ...(frame.data as object) } as AppStreamEvent }
    },
  },
  bootstrap: () => request('/local/bootstrap'),
  profile: {
    get: () => request('/api/profile'),
    update: (draft) => request('/api/profile', { method: 'PUT', body: JSON.stringify(draft) }),
    uploadAvatar: (file) => upload<{ avatarFileId: number; avatarUrl: string }>('/api/profile/avatar', file),
  },
  partner: {
    getCurrent: () => request('/api/partner'), getOptions: () => request('/api/partner/options'), getPresets: () => request('/api/partner/avatar-presets'),
    create: (draft) => request('/api/partner', { method: 'POST', body: partnerBody(draft) }),
    update: (draft) => request('/api/partner', { method: 'PUT', body: partnerBody(draft) }),
    uploadAvatar: (file) => upload<{ avatarFileId: number; avatarUrl: string }>('/api/partner/avatar', file),
  },
  chat: {
    stickers: () => request('/api/chat/stickers'),
    async stickerContent(stickerId) {
      const { baseUrl, session } = getAgentConnection()
      const response = await fetch(`${baseUrl}/api/chat/stickers/${stickerId}/content`, { headers: { 'X-Yuni-Session': session } })
      if (!response.ok) throw new Error('表情图片加载失败')
      return response.blob()
    },
    async imageContent(imageId) {
      const { baseUrl, session } = getAgentConnection()
      const response = await fetch(`${baseUrl}/api/chat/images/${encodeURIComponent(imageId)}`, { headers: { 'X-Yuni-Session': session } })
      if (!response.ok) throw new Error('图片加载失败')
      return response.blob()
    },
    async voiceContent(voiceId) {
      const { baseUrl, session } = getAgentConnection()
      const response = await fetch(`${baseUrl}/api/chat/voices/${encodeURIComponent(voiceId)}`, { headers: { 'X-Yuni-Session': session } })
      if (!response.ok) throw new Error('语音加载失败')
      return response.blob()
    },
    submit: (content, clientMessageId, signal, stickerId) => request('/api/chat/messages', { method: 'POST', body: JSON.stringify({ content, clientMessageId, stickerId }), signal }),
    submitRetry: (messageId, signal) => request(`/api/chat/messages/${messageId}/submit-retry`, { method: 'POST', signal }),
    history: (cursor, limit = 30) => request(`/api/chat/messages?cursor=${encodeURIComponent(cursor ?? '')}&limit=${limit}`),
    stream: (content, clientMessageId, signal) => stream('/api/chat/messages/stream', { content, clientMessageId }, signal),
    retry: (messageId, signal) => stream(`/api/chat/messages/${messageId}/retry`, {}, signal),
    pollProactive: (afterId) => request(`/api/chat/proactive${afterId == null ? '' : `?afterId=${afterId}`}`),
    triggerProactive: () => request('/api/chat/proactive/trigger', { method: 'POST' }),
  },
  memory: {
    create: (draft) => request('/api/memories', { method: 'POST', body: JSON.stringify(draft) }),
    entities: () => request('/api/memories/entities'),
    candidates: () => request('/api/memories/candidates'),
    decideCandidate: (id, decision) => request(`/api/memories/candidates/${id}/${decision}`, { method: 'POST' }),
    history: (id) => request(`/api/memories/${id}/history`),
    list: (type, cursor, limit = 50, filter = {}) => {
      const params = new URLSearchParams({ type, cursor: cursor ?? '', limit: String(limit) })
      for (const [key, value] of Object.entries(filter)) if (value !== undefined && value !== '') params.set(key, String(value))
      return request(`/api/memories?${params}`)
    },
    update: (id, patch) => request(`/api/memories/${id}`, { method: 'PUT', body: JSON.stringify(patch) }),
    resolveConflict: (id, content, options = {}) => request(`/api/memories/${id}/resolve`, { method: 'POST', body: JSON.stringify({ content, ...options }) }),
    remove: (id) => request(`/api/memories/${id}`, { method: 'DELETE' }), clear: () => request('/api/memories', { method: 'DELETE' }),
  },
  directories: { browse: (path) => request('/api/directories/browse', { method: 'POST', body: JSON.stringify({ path }) }) },
  settings: {
    get: () => request('/api/settings'),
    setMemoryEnabled: (memoryEnabled) => request('/api/settings/memory', { method: 'PUT', body: JSON.stringify({ memoryEnabled }) }),
    saveProactive: (config) => request('/api/settings/proactive', { method: 'PUT', body: JSON.stringify(config) }),
    saveModel: (config) => request('/api/settings/model', { method: 'PUT', body: JSON.stringify(config) }),
    clearModelKey: () => request('/api/settings/model/key', { method: 'DELETE' }),
    saveEmbedding: (config) => request('/api/settings/embedding', { method: 'PUT', body: JSON.stringify(config) }),
    clearEmbeddingKey: () => request('/api/settings/embedding/key', { method: 'DELETE' }),
    saveImage: (config) => request('/api/settings/image', { method: 'PUT', body: JSON.stringify(config) }),
    clearImageKey: () => request('/api/settings/image/key', { method: 'DELETE' }),
    saveVoice: (config) => request('/api/settings/voice', { method: 'PUT', body: JSON.stringify(config) }),
    clearVoiceKey: () => request('/api/settings/voice/key', { method: 'DELETE' }),
    listVoices: (config) => request('/api/settings/voice/voices', { method: 'POST', body: JSON.stringify(config) }),
    async previewVoice(config) {
      const { baseUrl, session } = getAgentConnection()
      const response = await fetch(`${baseUrl}/api/settings/voice/preview`, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Yuni-Session': session }, body: JSON.stringify(config) })
      if (!response.ok) {
        let message = '语音试听失败'
        try { const body = await response.json(); message = body.message || message } catch {}
        throw new Error(message)
      }
      return response.blob()
    },
    rebuildExpressions: () => request('/api/settings/embedding/rebuild', { method: 'POST' }),
    listModels: (config) => request('/api/settings/models', { method: 'POST', body: JSON.stringify(config) }),
  },
  roots: { list: () => request('/api/allowed-roots'), add: (path) => request('/api/allowed-roots', { method: 'POST', body: JSON.stringify({ path }) }), remove: (id) => request(`/api/allowed-roots/${id}`, { method: 'DELETE' }) },
  tools: { list: () => request('/api/tool-calls'), decide: (callId, decision) => request(`/api/tool-calls/${callId}/decision`, { method: 'POST', body: JSON.stringify({ decision }) }) },
  data: {
    export: async () => { const { baseUrl, session } = getAgentConnection(); const response = await fetch(`${baseUrl}/api/data/export`, { headers: { 'X-Yuni-Session': session } }); if (!response.ok) throw new Error('导出失败'); return response.blob() },
    import: (file) => upload('/api/data/import', file), clear: () => request('/api/data', { method: 'DELETE' }),
  },
  reminders: {
    due: () => request('/api/reminders/due'),
    markDelivered: (id) => request(`/api/reminders/${id}/delivered`, { method: 'POST' }),
  },
  system: { openLogs: () => request('/local/logs/open', { method: 'POST' }) },
}
