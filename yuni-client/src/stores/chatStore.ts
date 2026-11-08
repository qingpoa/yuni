import { create } from 'zustand'
import { appApi } from '../services/api'
import type { AppStreamEvent, ChatMessage, ChatStreamEvent, MessageAccepted, Reminder, StickerOption } from '../types/domain'
import { useAppStore } from './appStore'
import { useSettingsStore } from './settingsStore'
import { useToolStore } from './toolStore'

const activeControllers = new Set<AbortController>()
const activeTurns = new Set<number>(), finishedTurns = new Set<number>()
let generation = 0
let presentationController = new AbortController()
let presentation: Promise<void> = Promise.resolve()
let pendingPresentations = 0
const receivedIds = new Set<number>(), failedTurns = new Set<number>()
const receivedToolEvents = new Set<string>(), seenReminders = new Set<string>()
const recentSubmissions = new Map<string, number>()
const DUPLICATE_SEND_WINDOW_MS = 600
interface EventOwner { lifecycle: AbortController; connection: AbortController | null }
let eventOwner: EventOwner | null = null
export type ChatNotification = { kind: 'message'; message: ChatMessage } | { kind: 'reminder'; reminder: Reminder }

interface ChatState {
  messages: ChatMessage[]
  nextCursor: string | null
  loading: boolean
  loadingEarlier: boolean
  streamStatus: 'idle' | 'connecting' | 'thinking' | 'using_tool' | 'presenting' | 'failed'
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
  eventCursor: number | null
  streamingBuffer: string
  error: string | null
  loadLatest: () => Promise<void>
  loadEarlier: () => Promise<number>
  send: (content: string, sticker?: StickerOption) => Promise<void>
  retry: (userMessageId: number) => Promise<void>
  receiveProactive: (incoming: ChatMessage[]) => ChatMessage[]
  startEvents: (notify: (notification: ChatNotification) => void) => void
  stopEvents: () => void
  cancel: () => void
  reset: () => void
}

const mergeMessages = (current: ChatMessage[], incoming: ChatMessage[]) => {
  const map = new Map<string, ChatMessage>()
  for (const message of [...current, ...incoming]) {
    if (message.status === 'streaming') continue
    const key = message.clientMessageId ? 'client-' + message.clientMessageId : message.id ? 'id-' + message.id : message.localId
    map.set(key, message)
  }
  return [...map.values()].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime() || (a.id ?? 0) - (b.id ?? 0))
}

function waitForDelay(signal: AbortSignal, delay = 1000): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return }
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('Aborted', 'AbortError')) }
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve() }, delay)
    signal.addEventListener('abort', abort, { once: true })
  })
}

function modelConfigured() {
  return useSettingsStore.getState().settings?.model.configured ?? useAppStore.getState().bootstrapData?.modelConfigured ?? false
}

export const useChatStore = create<ChatState>((set, get) => {
  const refreshStatus = () => {
    const tools = useToolStore.getState()
    set({ streamStatus: activeControllers.size || activeTurns.size
      ? tools.active || tools.pending ? 'using_tool' : pendingPresentations ? 'presenting' : 'thinking'
      : pendingPresentations ? 'presenting' : get().error ? 'failed' : 'idle' })
  }
  const queueMessage = (message: ChatMessage): boolean => {
    const id = message.id
    if (id == null || receivedIds.has(id)) return false
    const parts = message.replyParts?.length ? message.replyParts : [message.content]
    if (!parts.length || parts.some(part => !part.trim())) return false
    receivedIds.add(id)
    const signal = presentationController.signal, version = generation
    pendingPresentations++
    refreshStatus()
    presentation = presentation.then(async () => {
      if (signal.aborted || version !== generation) return
      const existing = get().messages.find(item => item.id === id)
      const reply = { ...message, replyParts: parts, visibleParts: existing ? parts.length : 1 }
      set({ messages: mergeMessages(get().messages, [reply]), streamingBuffer: '' })
      for (let count = reply.visibleParts + 1; count <= parts.length; count++) {
        await waitForDelay(signal)
        if (signal.aborted || version !== generation) return
        set({ messages: get().messages.map(item => item.id === id ? { ...item, visibleParts: count } : item) })
      }
    }).catch(error => {
      if (!signal.aborted && version === generation) set({ error: error instanceof Error ? error.message : '回复展示失败' })
    }).finally(() => { if (version === generation) { pendingPresentations--; refreshStatus() } })
    return true
  }
  const queueReply = (event: Extract<ChatStreamEvent, { type: 'reply_completed' | 'completed' }>) => {
    if (event.assistantMessageId == null) return undefined
    const existing = get().messages.find(item => item.id === event.assistantMessageId)
    const message: ChatMessage = { localId: 'server-' + event.assistantMessageId, id: event.assistantMessageId, clientMessageId: null,
      role: 'assistant', content: event.content ?? '', replyParts: event.replyParts ?? [event.content ?? ''],
      sticker: event.sticker ?? existing?.sticker ?? null,
      image: event.image ?? existing?.image ?? null,
      voice: event.voice ?? existing?.voice ?? null,
      status: 'succeeded',
      messageType: event.messageType ?? 'normal', helpResources: event.helpResources ?? [], parentMessageId: event.parentMessageId ?? null,
      errorCode: null, errorMessage: null, createdAt: event.createdAt ?? existing?.createdAt ?? new Date().toISOString() }
    return queueMessage(message) ? message : undefined
  }
  const finish = (ids: number[]) => { for (const id of ids) { activeTurns.delete(id); finishedTurns.add(id) } }
  const fail = (message: string, parentId: number | null, code: number | null = null) => {
    set({ error: message })
    if (parentId != null && (failedTurns.has(parentId) || get().messages.some(item => item.status === 'failed' && item.parentMessageId === parentId))) return
    if (parentId != null) failedTurns.add(parentId)
    const version = generation, signal = presentationController.signal
    const failure: ChatMessage = { localId: 'failed-' + crypto.randomUUID(), id: null, clientMessageId: null, role: 'assistant', content: '',
      status: 'failed', messageType: 'normal', helpResources: [], parentMessageId: parentId, errorCode: code, errorMessage: message, createdAt: new Date().toISOString() }
    pendingPresentations++
    presentation = presentation.then(() => {
      if (version === generation && !signal.aborted) set({ messages: [...get().messages, failure] })
    }).catch(() => {}).finally(() => { if (version === generation) { pendingPresentations--; refreshStatus() } })
  }
  const accept = (result: MessageAccepted) => {
    const id = result.message.id!
    set({ messages: mergeMessages(get().messages, [result.message]) })
    if (['queued', 'running'].includes(result.turnStatus) && !finishedTurns.has(id)) activeTurns.add(id)
    else activeTurns.delete(id)
    refreshStatus()
  }
  const reminder = (item: Reminder, notify: (notification: ChatNotification) => void) => {
    const key = `${item.id}:${item.remindAt}`
    if (!seenReminders.has(key)) { seenReminders.add(key); notify({ kind: 'reminder', reminder: item }) }
    void appApi.reminders.markDelivered(item.id).catch(() => {})
  }
  const receiveEvent = (event: AppStreamEvent, notify: (notification: ChatNotification) => void) => {
    if (event.type === 'ready') {
      if (event.reset) { get().reset(); return }
      activeTurns.clear()
      for (const id of event.activeUserMessageIds) activeTurns.add(id)
      const calls = event.toolCalls
      useToolStore.setState({ pending: calls.find(item => item.status === 'awaiting_confirmation') ?? null, active: calls.find(item => item.status === 'running') ?? null,
        calls: [...calls, ...useToolStore.getState().calls.filter(item => !calls.some(call => call.id === item.id))] })
      set({ messages: mergeMessages(get().messages, event.messages), eventCursor: get().eventCursor ?? event.cursor, connectionStatus: 'connected' })
      for (const item of event.reminders) reminder(item, notify)
    } else if (event.type === 'state_reset') {
      get().reset()
      void useAppStore.getState().bootstrap()
    } else if (event.type === 'user_message_created') {
      if (!finishedTurns.has(event.message.id!)) activeTurns.add(event.message.id!)
      set({ messages: mergeMessages(get().messages, [event.message]), error: null })
    } else if (event.type === 'turn_queued') {
      for (const id of event.userMessageIds) { finishedTurns.delete(id); activeTurns.add(id) }
    } else if (event.type === 'reminder_due') {
      reminder(event.reminder, notify)
    } else if (event.type === 'tool_confirmation_required' || event.type === 'tool_started' || event.type === 'tool_completed') {
      const key = `${event.type}:${event.toolCall.id}`
      if (!receivedToolEvents.has(key)) { receivedToolEvents.add(key); useToolStore.getState().handleEvent(event) }
    } else if (event.type === 'delta') {
      set({ streamingBuffer: get().streamingBuffer + event.text })
    } else if (event.type === 'reply_completed' || event.type === 'completed') {
      const message = queueReply(event)
      if (message) notify({ kind: 'message', message })
      if (event.type === 'completed') finish(event.userMessageIds ?? (event.parentMessageId == null ? [] : [event.parentMessageId]))
    } else if (event.type === 'error') {
      finish(event.userMessageIds ?? (event.parentMessageId == null ? [] : [event.parentMessageId]))
      fail(event.message, event.parentMessageId ?? null, event.code)
    }
    refreshStatus()
  }
  return {
    messages: [], nextCursor: null, loading: false, loadingEarlier: false, streamStatus: 'idle', connectionStatus: 'disconnected', eventCursor: null, streamingBuffer: '', error: null,
    loadLatest: async () => {
      if (get().loading) return
      const version = generation
      set({ loading: true })
      try { const page = await appApi.chat.history(null, 30); if (version === generation) set({ messages: mergeMessages(page.items, get().messages), nextCursor: page.nextCursor }) }
      catch (error) { if (version === generation) set({ error: error instanceof Error ? error.message : '聊天记录加载失败' }) }
      finally { if (version === generation) set({ loading: false }) }
    },
    loadEarlier: async () => {
      const cursor = get().nextCursor
      if (!cursor || get().loadingEarlier) return 0
      const version = generation
      set({ loadingEarlier: true })
      try { const page = await appApi.chat.history(cursor, 30); if (version !== generation) return 0; set({ messages: mergeMessages(page.items, get().messages), nextCursor: page.nextCursor }); return page.items.length }
      finally { if (version === generation) set({ loadingEarlier: false }) }
    },
    send: async (content, sticker) => {
      content = content.trim()
      if (get().loading || (!content && !sticker)) return
      if (!modelConfigured()) { set({ error: '请先在设置中配置模型' }); return }
      const now = performance.now()
      for (const [value, sentAt] of recentSubmissions) if (now - sentAt >= DUPLICATE_SEND_WINDOW_MS) recentSubmissions.delete(value)
      const submissionKey = JSON.stringify([content, sticker?.id])
      if (recentSubmissions.has(submissionKey)) return
      recentSubmissions.set(submissionKey, now)
      const clientMessageId = crypto.randomUUID(), localId = 'local-' + clientMessageId
      const optimistic: ChatMessage = { localId, id: null, clientMessageId, role: 'user', content, sticker, status: 'pending', messageType: 'normal', helpResources: [], parentMessageId: null, errorCode: null, errorMessage: null, createdAt: new Date().toISOString() }
      set({ messages: [...get().messages, optimistic], error: null })
      const controller = new AbortController(), version = generation
      activeControllers.add(controller); refreshStatus()
      try { const result = await appApi.chat.submit(content, clientMessageId, controller.signal, sticker?.id); if (version === generation) accept(result) }
      catch (error) { if (version === generation && !controller.signal.aborted && !get().messages.some(item => item.clientMessageId === clientMessageId && item.id != null)) set({ error: error instanceof Error ? error.message : '消息发送状态暂未确认' }) }
      finally { activeControllers.delete(controller); if (version === generation) refreshStatus() }
    },
    retry: async userMessageId => {
      if (get().loading || activeTurns.size || activeControllers.size) return
      failedTurns.delete(userMessageId); finishedTurns.delete(userMessageId)
      set({ messages: get().messages.filter(item => !(item.status === 'failed' && item.parentMessageId === userMessageId)), error: null })
      const controller = new AbortController(), version = generation
      activeControllers.add(controller); refreshStatus()
      try { const result = await appApi.chat.submitRetry(userMessageId, controller.signal); if (version === generation) { accept(result); if (!['queued', 'running'].includes(result.turnStatus)) await get().loadLatest() } }
      catch (error) { if (version === generation && !controller.signal.aborted) fail(error instanceof Error ? error.message : '重试失败', userMessageId) }
      finally { activeControllers.delete(controller); if (version === generation) refreshStatus() }
    },
    receiveProactive: incoming => incoming.filter(queueMessage),
    startEvents: notify => {
      if (eventOwner) return
      const owner: EventOwner = { lifecycle: new AbortController(), connection: null }
      eventOwner = owner
      void (async () => {
        let delay = 1000
        while (eventOwner === owner && !owner.lifecycle.signal.aborted) {
          const version = generation
          owner.connection = new AbortController()
          set({ connectionStatus: get().eventCursor == null ? 'connecting' : 'reconnecting' })
          try {
            for await (const frame of appApi.events.stream(get().eventCursor, owner.connection.signal)) {
              if (eventOwner !== owner || version !== generation) break
              receiveEvent(frame.event, notify)
              if (version !== generation) break
              if (frame.id != null) set({ eventCursor: Math.max(get().eventCursor ?? 0, frame.id) })
              delay = 1000
            }
          } catch { /* Transport errors reconnect; backend tasks continue independently. */ }
          if (eventOwner !== owner || owner.lifecycle.signal.aborted) break
          if (version !== generation) continue
          set({ connectionStatus: 'reconnecting' })
          try { await waitForDelay(owner.lifecycle.signal, delay) } catch { break }
          delay = Math.min(delay * 2, 15000)
        }
      })()
    },
    stopEvents: () => {
      const owner = eventOwner
      eventOwner = null
      owner?.lifecycle.abort(); owner?.connection?.abort()
      set({ connectionStatus: 'disconnected' })
    },
    cancel: () => {
      generation++
      for (const controller of activeControllers) controller.abort()
      activeControllers.clear(); activeTurns.clear(); finishedTurns.clear()
      eventOwner?.connection?.abort()
      presentationController.abort(); presentationController = new AbortController(); presentation = Promise.resolve(); pendingPresentations = 0
      receivedIds.clear(); failedTurns.clear(); receivedToolEvents.clear(); recentSubmissions.clear(); seenReminders.clear()
      useToolStore.setState({ pending: null, active: null })
      set({ streamStatus: 'idle', streamingBuffer: '', loading: false, eventCursor: null })
    },
    reset: () => { get().cancel(); set({ messages: [], nextCursor: null, loadingEarlier: false, error: null }) },
  }
})
