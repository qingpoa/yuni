import type { AppApi } from './appApi'
import type { AvatarPreset, ChatMessage, ChatStreamEvent, HelpResource, MemoryItem, PartnerOptions, PartnerProfile, Settings, ToolCall, ToolDecision, UserProfile } from '../types/domain'
import { ApiError } from '../types/domain'
import type { AppEventFrame, AppStreamEvent } from '../types/domain'

const DB_KEY = 'yuni.mock.local.v3'

interface MockDatabase {
  profile: UserProfile
  partner: PartnerProfile | null
  settings: Settings
  messages: ChatMessage[]
  memories: MemoryItem[]
  roots: Array<{ id: number; path: string; displayName: string; createdAt: string }>
  toolCalls: ToolCall[]
  nextId: number
}

const partnerOptions: PartnerOptions = {
  personalityTags: [
    { value: 'gentle', label: '温柔' }, { value: 'reliable', label: '可靠' }, { value: 'patient', label: '耐心' },
    { value: 'lively', label: '有趣' }, { value: 'rational', label: '理性' }, { value: 'delicate', label: '细腻' },
  ],
  interactionStyles: [
    { value: 'daily_companion', label: '日常陪伴' }, { value: 'emotional_comfort', label: '情绪安慰' },
    { value: 'light_humor', label: '轻松玩笑' }, { value: 'grow_together', label: '共同成长' },
  ],
  intimacyLevels: [
    { value: 'reserved', label: '克制' }, { value: 'gentle_natural', label: '温柔自然' }, { value: 'warm_close', label: '亲近温暖' },
  ],
}

export const avatarPresets: AvatarPreset[] = [
  { id: 'mist', name: '晨雾', url: 'mock-avatar://mist', color: '#819587' },
  { id: 'river', name: '青川', url: 'mock-avatar://river', color: '#657b83' },
  { id: 'sunset', name: '暖霞', url: 'mock-avatar://sunset', color: '#a0837d' },
  { id: 'night', name: '夜蓝', url: 'mock-avatar://night', color: '#7b788d' },
]

const iso = (minutesAgo = 0) => new Date(Date.now() - minutesAgo * 60_000).toISOString()

function createPartner(): PartnerProfile {
  return {
    id: 10, name: '林夏', gender: 'female', avatarType: 'preset', avatarPresetId: 'mist', avatarFileId: null, avatarUrl: 'mock-avatar://mist',
    userNickname: '小予', relationshipStatus: 'stable_partner', personalityTags: ['gentle', 'reliable', 'patient'],
    interactionStyles: ['daily_companion', 'emotional_comfort'], intimacyLevel: 'gentle_natural',
    boundaryText: '不要用命令式语气催我。难过时先陪我一会儿，再问我是否需要建议。',
    customDescription: '',
    createdAt: new Date(Date.now() - 74 * 86_400_000).toISOString(), updatedAt: iso(60),
  }
}

function createUserProfile(): UserProfile {
  return { id: 1, nickname: '我', avatarType: 'preset', avatarPresetId: 'river', avatarFileId: null, avatarUrl: 'mock-avatar://river', createdAt: iso(), updatedAt: iso() }
}

function message(id: number, role: 'user' | 'assistant', content: string, minutesAgo: number, parentMessageId: number | null = null): ChatMessage {
  return { localId: `server-${id}`, id, clientMessageId: role === 'user' ? crypto.randomUUID() : null, role, content, status: 'succeeded', messageType: 'normal', helpResources: [], parentMessageId, errorCode: null, errorMessage: null, createdAt: iso(minutesAgo) }
}

function seedDatabase(): MockDatabase {
  const partner = createPartner()
  return {
    partner,
    profile: createUserProfile(),
    settings: { memoryEnabled: true, privacyVersion: 'v1.0', model: { provider: 'openai_compatible', baseUrl: 'https://api.openai.com/v1', modelName: 'gpt-5-mini', configured: true } },
    messages: [
      message(101, 'assistant', '晚上好，小予。今天想先安静地待一会儿，还是想和我说说发生了什么？', 42),
      message(102, 'user', '今天开了好多会，脑子一直停不下来。', 38),
      message(103, 'assistant', '辛苦了。我记得连续开会后你最需要一点缓冲，不急着复盘。先喝口水，我们慢慢把节奏降下来。', 37, 102),
      message(104, 'user', '嗯。你还记得我不喜欢被催，这就很好。', 35),
      message(105, 'assistant', '记得。今晚不催你做决定，我们就把步子放慢一点。', 34, 104),
    ],
    memories: [
      { id: 201, type: 'user_preference', title: '疲惫的时候，不喜欢被催促', content: '你希望在疲惫时先得到安静的陪伴，等情绪缓下来之后再一起讨论建议。', sourceMessageId: 104, sourceLabel: '昨天晚上的聊天', sourceExcerpt: '你还记得我不喜欢被催，这就很好。', importance: 5, embeddingStatus: 'indexed', status: 'active', conflictContent: null, createdAt: iso(1440), updatedAt: iso(1440) },
      { id: 202, type: 'emotion_pattern', title: '连续开会后容易停不下思绪', content: '工作中连续参加会议后，你通常需要一点独处和缓冲时间。', sourceMessageId: 102, sourceLabel: '今天 18:26', sourceExcerpt: '今天开了好多会，脑子一直停不下来。', importance: 4, embeddingStatus: 'indexed', status: 'active', conflictContent: null, createdAt: iso(50), updatedAt: iso(50) },
      { id: 203, type: 'important_event', title: '九月准备第一次独自旅行', content: '你正在计划去海边待三天，这是你第一次完全由自己安排的旅行。', sourceMessageId: null, sourceLabel: '8 月 29 日', sourceExcerpt: null, importance: 4, embeddingStatus: 'indexed', status: 'conflicted', conflictContent: '旅行计划已改到十月，目的地还没有确定。', createdAt: iso(12_000), updatedAt: iso(120) },
      { id: 204, type: 'boundary', title: '先陪伴，再给建议', content: '当你在表达难过时，不要立刻分析问题或给出任务清单。', sourceMessageId: null, sourceLabel: '创建伴侣时设置', sourceExcerpt: null, importance: 5, embeddingStatus: 'indexed', status: 'active', conflictContent: null, createdAt: partner.createdAt, updatedAt: partner.updatedAt },
    ],
    roots: [{ id: 260, path: 'D:\\Yuni Workspace', displayName: 'Yuni Workspace', createdAt: iso(240) }],
    toolCalls: [], nextId: 300,
  }
}

function loadDatabase() {
  try { const stored = localStorage.getItem(DB_KEY); const loaded = stored ? JSON.parse(stored) as MockDatabase : seedDatabase(); loaded.profile ??= createUserProfile(); if (loaded.partner) { loaded.partner.gender ??= null; loaded.partner.customDescription ??= '' } return loaded }
  catch { return seedDatabase() }
}

let database = loadDatabase()
const clone = <T,>(value: T): T => structuredClone(value)
const saveDatabase = () => { try { localStorage.setItem(DB_KEY, JSON.stringify(database)) } catch { /* Mock session remains usable. */ } }
const wait = (ms = 240, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  const timer = window.setTimeout(resolve, ms)
  signal?.addEventListener('abort', () => { window.clearTimeout(timer); reject(new DOMException('请求已取消', 'AbortError')) }, { once: true })
})
const dataUrl = (file: File) => new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(new ApiError(50000, '头像读取失败')); reader.readAsDataURL(file) })

const decisionWaiters = new Map<string, (decision: ToolDecision) => void>()
function waitForDecision(callId: string, signal?: AbortSignal) {
  return new Promise<ToolDecision>((resolve, reject) => {
    decisionWaiters.set(callId, resolve)
    signal?.addEventListener('abort', () => { decisionWaiters.delete(callId); reject(new DOMException('请求已取消', 'AbortError')) }, { once: true })
  })
}

function buildReply(content: string) {
  if (/累|疲惫|加班|开会/.test(content)) return { text: '听起来你已经撑了很久。我们先不急着解决什么，你可以在这里慢一点。', messageType: 'normal' as const, helpResources: [] as HelpResource[] }
  if (/开心|顺利|完成了|成功/.test(content)) return { text: '这件事值得好好开心一下。你愿意和我说说，哪一刻让你最有成就感吗？', messageType: 'normal' as const, helpResources: [] as HelpResource[] }
  return { text: '我在听。你不用一次把所有事情讲清楚，从此刻最想说的那一点开始就好。', messageType: 'normal' as const, helpResources: [] as HelpResource[] }
}

function maybeCreateMemory(messageItem: ChatMessage) {
  if (!database.settings.memoryEnabled || !/喜欢|不喜欢|准备|害怕|习惯/.test(messageItem.content)) return
  if (database.memories.some((item) => item.content.includes(messageItem.content))) return
  database.memories.unshift({ id: database.nextId++, type: /不喜欢|喜欢/.test(messageItem.content) ? 'user_preference' : 'user_fact', title: messageItem.content.slice(0, 34), content: messageItem.content, sourceMessageId: messageItem.id, sourceLabel: '刚刚的聊天', sourceExcerpt: messageItem.content, importance: 3, embeddingStatus: 'indexed', status: 'active', conflictContent: null, createdAt: iso(), updatedAt: iso() })
}

async function* generateReply(userMessage: ChatMessage, signal?: AbortSignal, isRetry = false): AsyncGenerator<ChatStreamEvent> {
  await wait(220, signal); yield { type: 'message_created', messageId: userMessage.id! }; await wait(320, signal)
  if (userMessage.content.includes('模拟失败') && !isRetry) {
    const failed: ChatMessage = { localId: `server-${database.nextId}`, id: database.nextId++, clientMessageId: null, role: 'assistant', content: '', status: 'failed', messageType: 'normal', helpResources: [], parentMessageId: userMessage.id, errorCode: 40303, errorMessage: '回复生成失败，请稍后重试', createdAt: iso() }
    database.messages.push(failed); saveDatabase(); yield { type: 'error', code: 40303, message: failed.errorMessage! }; return
  }

  let toolReply = ''
  if (/创建.*文件|写入.*文件|修改.*文件|删除.*文件/.test(userMessage.content)) {
    const destructive = /删除/.test(userMessage.content)
    const target = `${database.roots[0]?.path ?? 'D:\\Yuni Workspace'}\\yuni-note.txt`
    const toolCall: ToolCall = { id: crypto.randomUUID(), toolName: destructive ? 'delete_file' : 'create_text_file', risk: destructive ? 'destructive' : 'write', status: 'awaiting_confirmation', summary: destructive ? '删除用户指定的文本文件' : '创建一份文本备忘录', targetPaths: [target], diffPreview: destructive ? '- 文件将被删除' : '+ 这是 Yuni 根据对话整理的备忘录。', errorMessage: null, createdAt: iso(), completedAt: null }
    database.toolCalls.unshift(toolCall); saveDatabase(); yield { type: 'tool_confirmation_required', toolCall: clone(toolCall) }
    const decision = await waitForDecision(toolCall.id, signal)
    if (decision === 'reject') {
      toolCall.status = 'rejected'; toolCall.completedAt = iso(); saveDatabase(); yield { type: 'tool_completed', toolCall: clone(toolCall) }
      toolReply = '已按你的决定取消文件操作，没有更改任何内容。'
    } else {
      toolCall.status = 'running'; saveDatabase(); yield { type: 'tool_started', toolCall: clone(toolCall) }; await wait(480, signal)
      toolCall.status = 'succeeded'; toolCall.completedAt = iso(); saveDatabase(); yield { type: 'tool_completed', toolCall: clone(toolCall) }
      toolReply = destructive ? '文件已按你的确认删除。' : '备忘录已写入授权目录。'
    }
  } else if (/读取.*文件|查看.*文件/.test(userMessage.content)) {
    const toolCall: ToolCall = { id: crypto.randomUUID(), toolName: 'read_text_file', risk: 'read', status: 'running', summary: '读取授权目录内的文本文件', targetPaths: [`${database.roots[0]?.path ?? 'D:\\Yuni Workspace'}\\notes.txt`], diffPreview: null, errorMessage: null, createdAt: iso(), completedAt: null }
    database.toolCalls.unshift(toolCall); yield { type: 'tool_started', toolCall: clone(toolCall) }; await wait(350, signal); toolCall.status = 'succeeded'; toolCall.completedAt = iso(); saveDatabase(); yield { type: 'tool_completed', toolCall: clone(toolCall) }; toolReply = '我已读取授权目录内的文件，可以继续帮你整理。'
  }

  const reply = toolReply ? { text: toolReply, messageType: 'normal' as const, helpResources: [] as HelpResource[] } : buildReply(userMessage.content)
  const assistantId = database.nextId++
  const replyParts = reply.text.split(/(?<=[。！？])/u).map(part => part.trim()).filter(Boolean)
  const assistant: ChatMessage = { localId: `server-${assistantId}`, id: assistantId, clientMessageId: null, role: 'assistant', content: replyParts.join('\n'), replyParts, status: 'succeeded', messageType: reply.messageType, helpResources: reply.helpResources, parentMessageId: userMessage.id, errorCode: null, errorMessage: null, createdAt: iso() }
  database.messages.push(assistant); maybeCreateMemory(userMessage); saveDatabase()
  yield { type: 'completed', assistantMessageId: assistantId, content: assistant.content, replyParts, messageType: reply.messageType, helpResources: reply.helpResources }
}

const mockEvents: AppEventFrame[] = []
const mockListeners = new Set<() => void>(), mockActive = new Set<number>()
const mockControllers = new Map<number, AbortController>()
const cancelMockWork = () => { for (const controller of mockControllers.values()) controller.abort(); mockControllers.clear(); mockActive.clear() }
const emitMock = (event: AppStreamEvent) => {
  mockEvents.push({ id: mockEvents.length + 1, event })
  for (const listener of mockListeners) listener()
}
const runMockReply = (message: ChatMessage, retry = false) => {
  if (mockActive.has(message.id!)) return
  mockActive.add(message.id!)
  const controller = new AbortController()
  mockControllers.set(message.id!, controller)
  emitMock({ type: 'turn_queued', userMessageIds: [message.id!], parentMessageId: message.id! })
  void (async () => {
    try {
      for await (const event of generateReply(message, controller.signal, retry)) {
        if (event.type !== 'message_created') emitMock({ ...event, parentMessageId: message.id!, userMessageIds: [message.id!] })
      }
    } catch (error) {
      if (!controller.signal.aborted) emitMock({ type: 'error', code: 40303, message: error instanceof Error ? error.message : '回复失败', parentMessageId: message.id!, userMessageIds: [message.id!] })
    } finally { if (mockControllers.get(message.id!) === controller) { mockActive.delete(message.id!); mockControllers.delete(message.id!) } }
  })()
}

export const mockApi: AppApi = {
  events: {
    async *stream(afterId, signal) {
      let cursor = afterId ?? mockEvents.length
      yield { id: null, event: { type: 'ready', cursor, reset: false, messages: clone(database.messages.slice(-30)), activeUserMessageIds: [...mockActive], toolCalls: [], reminders: [] } }
      while (!signal.aborted) {
        const events = mockEvents.filter(item => item.id! > cursor)
        if (events.length) { for (const frame of events) { cursor = frame.id!; yield clone(frame) }; continue }
        await new Promise<void>(resolve => {
          const wake = () => { mockListeners.delete(wake); signal.removeEventListener('abort', wake); resolve() }
          mockListeners.add(wake); signal.addEventListener('abort', wake, { once: true })
          if (signal.aborted) wake()
        })
      }
    },
  },
  async bootstrap() { await wait(180); return { initialized: true, hasCompanion: Boolean(database.partner), memoryEnabled: database.settings.memoryEnabled, modelConfigured: database.settings.model.configured, agentVersion: '0.3.0-mock', protocolVersion: 1 } },
  profile: {
    async get() { await wait(100); return clone(database.profile) },
    async update(draft) { await wait(180); if (!draft.nickname.trim() || draft.nickname.length > 20) throw new ApiError(40000, '请填写 1-20 字的昵称'); database.profile = { ...database.profile, ...draft, nickname: draft.nickname.trim(), updatedAt: iso() }; saveDatabase(); return clone(database.profile) },
    async uploadAvatar(file) { return mockApi.partner.uploadAvatar(file) },
  },
  partner: {
    async getCurrent() { await wait(120); return clone(database.partner) }, async getOptions() { await wait(80); return clone(partnerOptions) }, async getPresets() { await wait(80); return clone(avatarPresets) },
    async create(draft) { await wait(280); if (database.partner) throw new ApiError(40101, '已经创建过伴侣'); const now = iso(); database.partner = { id: database.nextId++, ...draft, relationshipStatus: 'stable_partner', createdAt: now, updatedAt: now }; saveDatabase(); return clone(database.partner) },
    async update(draft) { await wait(280); if (!database.partner) throw new ApiError(40102, '尚未创建伴侣'); database.partner = { ...database.partner, ...draft, updatedAt: iso() }; saveDatabase(); return clone(database.partner) },
    async uploadAvatar(file) { if (file.size > 5 * 1024 * 1024) throw new ApiError(40204, '文件不能超过 5 MB'); if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new ApiError(40205, '仅支持 jpeg、png 或 webp'); await wait(240); return { avatarFileId: database.nextId++, avatarUrl: await dataUrl(file) } },
  },
  chat: {
    async stickers() { return [] },
    async stickerContent() { throw new ApiError(40401, '表情不存在') },
    async imageContent() { throw new ApiError(40401, '图片不存在') },
    async voiceContent() { throw new ApiError(40401, '语音不存在') },
    async submit(content, clientMessageId, _signal, stickerId) {
      if (stickerId != null) throw new ApiError(40401, '表情库中没有这个表情')
      let message = database.messages.find(item => item.clientMessageId === clientMessageId)
      if (!message) {
        const id = database.nextId++
        message = { localId: 'server-' + id, id, clientMessageId, role: 'user', content, status: 'succeeded', messageType: 'normal', helpResources: [], parentMessageId: null, errorCode: null, errorMessage: null, createdAt: iso() }
        database.messages.push(message); saveDatabase(); emitMock({ type: 'user_message_created', message: clone(message) })
        runMockReply(message)
      }
      return { message: clone(message), turnStatus: mockActive.has(message.id!) ? 'running' : 'completed' }
    },
    async submitRetry(id) {
      const message = database.messages.find(item => item.id === id && item.role === 'user')
      if (!message) throw new ApiError(40303, '找不到需要重试的消息')
      runMockReply(message, true)
      return { message: clone(message), turnStatus: 'running' }
    },
    async history(cursor, limit = 30) { await wait(160); const end = cursor ? Number(cursor) : database.messages.length; const start = Math.max(0, end - limit); return { items: clone(database.messages.slice(start, end)), nextCursor: start > 0 ? String(start) : null, proactiveCursor: Math.max(0, ...database.messages.filter(item => item.role === 'assistant' && item.parentMessageId === null && item.status === 'succeeded').map(item => item.id ?? 0)) } },
    async *stream(content, clientMessageId, signal) {
      if (!database.settings.model.configured) { yield { type: 'error', code: 41001, message: '请先配置模型' }; return }
      let userMessage = database.messages.find((item) => item.clientMessageId === clientMessageId)
      if (!userMessage) { userMessage = { localId: `server-${database.nextId}`, id: database.nextId++, clientMessageId, role: 'user', content, status: 'succeeded', messageType: 'normal', helpResources: [], parentMessageId: null, errorCode: null, errorMessage: null, createdAt: iso() }; database.messages.push(userMessage); saveDatabase() }
      yield* generateReply(userMessage, signal)
    },
    async *retry(messageId, signal) { const userMessage = database.messages.find((item) => item.id === messageId && item.role === 'user'); if (!userMessage) throw new ApiError(40303, '找不到需要重试的消息'); yield* generateReply(userMessage, signal, true) },
    async pollProactive(afterId) {
      const proactive = database.messages.filter(item => item.role === 'assistant' && item.parentMessageId === null && item.status === 'succeeded').sort((a, b) => (a.id ?? 0) - (b.id ?? 0))
      const items = afterId == null ? [] : proactive.filter(item => (item.id ?? 0) > afterId).slice(0, 10)
      return clone({ items, cursor: afterId == null ? proactive.at(-1)?.id ?? 0 : items.at(-1)?.id ?? afterId })
    },
    async triggerProactive() { const id = database.nextId++; const msg: ChatMessage = { localId: `server-${id}`, id, clientMessageId: null, role: 'assistant', content: '宝宝，你在干啥呀', replyParts: ['宝宝，你在干啥呀'], status: 'succeeded', messageType: 'normal', helpResources: [], parentMessageId: null, errorCode: null, errorMessage: null, createdAt: iso() }; database.messages.push(msg); saveDatabase(); return clone(msg) },
  },
  memory: {
    async create(draft) { const item: MemoryItem = { ...draft, id: database.nextId++, sourceMessageId: null, sourceLabel: '用户整理', sourceExcerpt: null, embeddingStatus: 'pending', status: 'active', conflictContent: null, createdAt: iso(), updatedAt: iso() }; database.memories.unshift(item); saveDatabase(); return clone(item) },
    async entities() { return [] },
    async candidates() { return [] },
    async decideCandidate() { throw new ApiError(40401, '候选不存在') },
    async history(id) { const ids = new Set([id]); for (let i = 0; i < database.memories.length; i++) for (const item of database.memories) { if (item.supersededById && ids.has(item.supersededById)) ids.add(item.id); if (ids.has(item.id) && item.supersededById) ids.add(item.supersededById) }; return clone(database.memories.filter(item => ids.has(item.id))) },
    async list(type, cursor, limit = 50, filter = {}) { await wait(150); const all = database.memories.filter((item) => (!type || item.type === type) && (!filter.kind || (item.kind ?? 'fact') === filter.kind) && (filter.status ? filter.status.split(',').includes(item.status) : ['active','conflicted'].includes(item.status)) && (!filter.q || `${item.title}${item.content}`.includes(filter.q))); const offset = cursor ? Number(cursor) : 0; return { items: clone(all.slice(offset, offset + limit)), nextCursor: offset + limit < all.length ? String(offset + limit) : null } },
    async update(id, patch) { await wait(180); const item = database.memories.find((entry) => entry.id === id); if (!item) throw new ApiError(40401, '记忆不存在'); const next = { ...item, ...patch, id: database.nextId++, status: 'active' as const, updatedAt: iso(), createdAt: iso() }; item.status = patch.mode === 'transition' ? 'historical' : 'retracted'; item.supersededById = next.id; database.memories.unshift(next); saveDatabase(); return clone(next) },
    async resolveConflict(id, content) { await wait(180); const item = database.memories.find((entry) => entry.id === id); if (!item) throw new ApiError(40401, '记忆不存在'); Object.assign(item, { content, status: 'active', conflictContent: null, updatedAt: iso() }); saveDatabase(); return clone(item) },
    async remove(id) { await wait(160); database.memories = database.memories.filter((entry) => entry.id !== id); saveDatabase() }, async clear() { await wait(200); database.memories = []; saveDatabase() },
  },
  settings: {
    async get() { await wait(100); return clone(database.settings) }, async setMemoryEnabled(enabled) { await wait(140); database.settings.memoryEnabled = enabled; saveDatabase(); return clone(database.settings) },
    async saveProactive(config) { await wait(140); Object.assign(database.settings, config); saveDatabase(); return clone(database.settings) },
    async saveModel(config) { await wait(220); database.settings.model = { provider: config.provider, baseUrl: config.baseUrl, modelName: config.modelName, configured: Boolean(config.apiKey?.trim()) || database.settings.model.configured }; saveDatabase(); return clone(database.settings) },
    async clearModelKey() { await wait(160); database.settings.model.configured = false; saveDatabase(); return clone(database.settings) },
    async saveEmbedding(config) { await wait(160); database.settings.embedding = { provider: config.provider, baseUrl: config.baseUrl, modelName: config.modelName, configured: Boolean(config.apiKey?.trim()) || Boolean(database.settings.embedding?.configured) }; saveDatabase(); return clone(database.settings) },
    async clearEmbeddingKey() { await wait(160); if (database.settings.embedding) database.settings.embedding.configured = false; saveDatabase(); return clone(database.settings) },
    async saveImage(config) { await wait(160); database.settings.image = { enabled: config.enabled, baseUrl: config.baseUrl, modelName: config.modelName, defaultSize: config.defaultSize, stylePrefix: config.stylePrefix, configured: Boolean(config.apiKey?.trim()) || Boolean(database.settings.image?.configured) }; saveDatabase(); return clone(database.settings) },
    async clearImageKey() { await wait(160); if (database.settings.image) database.settings.image.configured = false; saveDatabase(); return clone(database.settings) },
    async saveVoice(config) { await wait(160); database.settings.voice = { enabled: config.enabled, baseUrl: config.baseUrl, modelName: config.modelName, voice: config.voice, triggerMode: config.triggerMode, configured: Boolean(config.apiKey?.trim()) || Boolean(database.settings.voice?.configured) }; saveDatabase(); return clone(database.settings) },
    async clearVoiceKey() { await wait(160); if (database.settings.voice) database.settings.voice.configured = false; saveDatabase(); return clone(database.settings) },
    async previewVoice() { await wait(500); return new Blob(['fake-audio-data'], { type: 'audio/mpeg' }) },
    async listVoices() { await wait(160); return [{ id: 'female-shaonv', name: '少女音色', kind: 'system' }, { id: 'demo-clone', name: '示例克隆音色', kind: 'cloned' }] },
    async rebuildExpressions() { await wait(160) },
    async listModels() { await wait(160); return ['deepseek-flash', 'deepseek-v4-pro', 'jev-latest', 'jev-preview'] },
  },
  roots: {
    async list() { await wait(100); return clone(database.roots) }, async add(path) { await wait(160); const normalized = path.trim(); if (!normalized) throw new ApiError(42001, '请选择授权目录'); if (database.roots.some((root) => root.path.toLowerCase() === normalized.toLowerCase())) throw new ApiError(42002, '该目录已授权'); const root = { id: database.nextId++, path: normalized, displayName: normalized.split(/[\\/]/).filter(Boolean).pop() ?? normalized, createdAt: iso() }; database.roots.push(root); saveDatabase(); return clone(root) },
    async remove(id) { await wait(140); database.roots = database.roots.filter((root) => root.id !== id); saveDatabase() },
  },
  directories: {
    async browse(path) {
      await wait(100)
      if (path === null) return { path: null, parentPath: null, items: [{ name: 'Yuni Workspace', path: 'D:\\Yuni Workspace' }] }
      return { path, parentPath: null, items: [] }
    },
  },
  tools: {
    async list() { await wait(100); return clone(database.toolCalls) }, async decide(callId, decision) { const toolCall = database.toolCalls.find((entry) => entry.id === callId); if (!toolCall) throw new ApiError(43001, '工具调用不存在'); decisionWaiters.get(callId)?.(decision); decisionWaiters.delete(callId); return clone(toolCall) },
  },
  data: {
    async export() { await wait(160); return new Blob([JSON.stringify(database, null, 2)], { type: 'application/json' }) },
    async import(file) { await wait(240); const parsed = JSON.parse(await file.text()) as MockDatabase; if (!parsed.settings || !Array.isArray(parsed.messages) || !Array.isArray(parsed.memories)) throw new ApiError(44001, '导入文件格式不正确'); parsed.profile ??= createUserProfile(); if (parsed.partner) parsed.partner.customDescription ??= ''; cancelMockWork(); database = parsed; saveDatabase(); emitMock({ type: 'state_reset', reason: 'imported' }); return { hasCompanion: Boolean(database.partner), importedMessages: database.messages.length, importedMemories: database.memories.length } },
    async clear() { await wait(220); cancelMockWork(); database = { ...seedDatabase(), partner: null, messages: [], memories: [], toolCalls: [], roots: [], settings: { ...seedDatabase().settings, model: { ...seedDatabase().settings.model, configured: false } } }; saveDatabase(); emitMock({ type: 'state_reset', reason: 'cleared' }) },
  },
  reminders: { async due() { return [] }, async markDelivered() { await wait(20) } },
  system: { async openLogs() { await wait(80) } },
}
