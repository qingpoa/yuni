import { motion } from 'framer-motion'
import { ArrowUp, ExternalLink, FileCheck2, RefreshCw, Search, ShieldCheck, Smile, Wrench, X } from 'lucide-react'
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { Link } from 'react-router-dom'
import { CompanionAvatar } from '../components/CompanionAvatar'
import { Button } from '../components/ui/button'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog'
import { Tooltip } from '../components/ui/tooltip'
import { openExternal } from '../services/desktop'
import { appApi } from '../services/api'
import { useAppStore } from '../stores/appStore'
import { useChatStore } from '../stores/chatStore'
import { usePartnerStore } from '../stores/partnerStore'
import { useProfileStore } from '../stores/profileStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useToolStore } from '../stores/toolStore'
import type { ChatMessage, GeneratedImageAttachment, StickerOption, GeneratedVoiceAttachment } from '../types/domain'

const TIME_GAP_MS = 5 * 60 * 1000
const VOICE_PLAYED_STORAGE_KEY = 'yuni.voice.played.v1'
const VOICE_PLAYED_LIMIT = 500

// A single module-level owner guarantees one voice is audible at a time even
// when two bubbles share the same audio id or their state updates interleave.
let activeVoiceAudio: HTMLAudioElement | null = null

function readPlayedVoiceKeys(): Set<string> {
  try {
    const raw = window.localStorage.getItem(VOICE_PLAYED_STORAGE_KEY)
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : [])
  } catch {
    return new Set()
  }
}

function rememberPlayedVoice(key: string): void {
  try {
    const keys = readPlayedVoiceKeys()
    keys.add(key)
    window.localStorage.setItem(VOICE_PLAYED_STORAGE_KEY, JSON.stringify([...keys].slice(-VOICE_PLAYED_LIMIT)))
  } catch {
    // Reading state is non-critical; blocked storage must never break playback.
  }
}

function useVoicePlayed(key: string) {
  const [played, setPlayed] = useState(() => readPlayedVoiceKeys().has(key))
  const markPlayed = useCallback(() => {
    if (played) return
    setPlayed(true)
    rememberPlayedVoice(key)
  }, [key, played])
  return [played, markPlayed] as const
}

const formatTime = (value: string) => new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value))

function dayKey(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`
}

function formatDividerTime(value: string, includeTodayLabel: boolean): string {
  const date = new Date(value)
  const now = new Date()
  const timeStr = formatTime(value)
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfTarget = new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
  const diffDays = Math.round((startOfToday - startOfTarget) / 86_400_000)

  if (diffDays === 0) return includeTodayLabel ? `今天 ${timeStr}` : timeStr
  if (diffDays === 1) return `昨天 ${timeStr}`
  if (diffDays === 2) return `前天 ${timeStr}`
  if (date.getFullYear() === now.getFullYear()) {
    return `${date.getMonth() + 1}月${date.getDate()}日 ${timeStr}`
  }
  return `${date.getFullYear()}年${date.getMonth() + 1}月${date.getDate()}日 ${timeStr}`
}

export function ChatPage() {
  const partner = usePartnerStore((state) => state.partner)
  const profile = useProfileStore((state) => state.profile)
  const settingsConfigured = useSettingsStore((state) => state.settings?.model.configured)
  const bootstrapConfigured = useAppStore((state) => state.bootstrapData?.modelConfigured ?? false)
  const modelConfigured = settingsConfigured ?? bootstrapConfigured
  const activeTool = useToolStore((state) => state.active)
  const { messages, nextCursor, loading, loadingEarlier, streamStatus, error, loadLatest, loadEarlier, send, retry } = useChatStore()
  const [playingVoiceKey, setPlayingVoiceKey] = useState<string | null>(null)
  const [text, setText] = useState('')
  const draftRef = useRef('')
  const [searching, setSearching] = useState(false)
  const [query, setQuery] = useState('')
  const [stickersOpen, setStickersOpen] = useState(false)
  const [stickersLoading, setStickersLoading] = useState(false)
  const [stickersError, setStickersError] = useState<string | null>(null)
  const [stickers, setStickers] = useState<StickerOption[]>([])
  const stickerAnchorRef = useRef<HTMLDivElement>(null)
  const [pickerPosition, setPickerPosition] = useState({ left: -142, width: 320, arrow: 160 })
  const paneRef = useRef<HTMLElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const busy = loading || ['connecting', 'thinking', 'using_tool', 'presenting'].includes(streamStatus)
  const sendDisabled = loading || !modelConfigured || !text.trim()
  const displayedCount = messages.reduce((count, message) => count + (message.replyParts?.slice(0, message.visibleParts).length ?? 1), 0)
  const positionStickerPicker = useCallback(() => {
    const anchor = stickerAnchorRef.current?.getBoundingClientRect()
    if (!anchor) return
    const width = Math.min(320, window.innerWidth - 32)
    const center = anchor.left + anchor.width / 2
    const left = Math.max(16, Math.min(center - width / 2, window.innerWidth - width - 16))
    setPickerPosition({ left: left - anchor.left, width, arrow: center - left })
  }, [])

  useEffect(() => { void loadLatest() }, [loadLatest])
  useEffect(() => { if (!paneRef.current || loadingEarlier) return; paneRef.current.scrollTop = paneRef.current.scrollHeight }, [loading, loadingEarlier, displayedCount])
  useEffect(() => {
    if (!stickersOpen) return
    const outside = (event: PointerEvent) => {
      if (!stickerAnchorRef.current?.contains(event.target as Node)) setStickersOpen(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setStickersOpen(false); stickerAnchorRef.current?.querySelector('button')?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    window.addEventListener('resize', positionStickerPicker)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
      window.removeEventListener('resize', positionStickerPicker)
    }
  }, [stickersOpen, positionStickerPicker])

  const visibleMessages = useMemo(() => {
    const rows = messages
      .filter(message => message.status !== 'streaming')
      .flatMap(message => {
        if (message.role === 'assistant' && message.status === 'succeeded' && message.replyParts?.length) {
          // A voice reply is one WeChat-style bubble: never split it into text parts,
          // otherwise the preceding parts would render as separate plain messages.
          if (message.voice) {
            return [{
              ...message,
              content: message.content || message.replyParts.join('\n'),
              sticker: null,
              image: null,
              voice: message.voice,
            }]
          }
          const total = message.replyParts.length
          const shown = message.replyParts.slice(0, message.visibleParts)
          return shown.map((content, index) => ({
            ...message,
            content,
            sticker: index === total - 1 ? message.sticker : null,
            image: index === total - 1 ? message.image : null,
            voice: null,
            localId: `${message.localId}-part-${index}`,
          }))
        }
        return [message]
      })
    return query.trim() ? rows.filter(item => `${item.content} ${item.sticker?.description ?? ''} ${item.image?.description ?? ''}`.includes(query.trim())) : rows
  }, [messages, query])
  const sendMessage = () => {
    const content = draftRef.current.trim()
    if (!content || loading || !modelConfigured) return
    // Consume this draft synchronously, before React renders the cleared value.
    // A second click/Enter in the same render must not submit it again.
    draftRef.current = ''
    setText('')
    composerRef.current?.focus()
    void send(content)
  }
  const sendSticker = (sticker: StickerOption) => {
    if (loading || !modelConfigured) return
    setStickersOpen(false)
    composerRef.current?.focus()
    void send('', sticker)
  }
  const loadOlderAtTop = async () => { if (!paneRef.current || !nextCursor || loadingEarlier) return; const pane = paneRef.current; const previousHeight = pane.scrollHeight; await loadEarlier(); requestAnimationFrame(() => { pane.scrollTop = pane.scrollHeight - previousHeight }) }
  const name = partner?.name ?? '伴侣'
  const openStickers = async () => {
    positionStickerPicker()
    setStickersOpen(true)
    setStickersLoading(true)
    setStickersError(null)
    try { setStickers(await appApi.chat.stickers()) }
    catch (error) { setStickersError(error instanceof Error ? error.message : '表情加载失败') }
    finally { setStickersLoading(false) }
  }
  const stickerPicker = stickersOpen && <div id="chat-sticker-picker" role="dialog" aria-label="表情包" aria-modal="false" className="sticker-picker-popover" style={{ left: pickerPosition.left, width: pickerPosition.width }}>
    <div className="sticker-picker-header"><span>表情包</span><Button variant="icon" aria-label="关闭表情面板" onClick={() => { setStickersOpen(false); stickerAnchorRef.current?.querySelector('button')?.focus() }}><X size={14} /></Button></div>
    <div className="sticker-picker-body">{!stickersLoading && !stickersError && stickers.length ? <div className="sticker-grid">{stickers.map(sticker => <button key={sticker.id} type="button" className="sticker-option" title={sticker.description} aria-label={sticker.description || `表情 ${sticker.id}`} disabled={loading || !modelConfigured} onClick={() => sendSticker(sticker)}><StickerImage sticker={sticker} compact /></button>)}</div> : <div className="sticker-picker-empty" aria-live="polite"><Smile size={28} strokeWidth={1.4} /><strong>{stickersLoading ? '正在加载表情…' : stickersError ? '表情加载失败' : '暂无表情'}</strong><p>{stickersError ?? (stickersLoading ? '稍等一下' : '表情库里还没有可用的表情包。')}</p>{stickersError && <Button variant="secondary" onClick={() => void openStickers()}>重新加载</Button>}</div>}</div>
    <span className="sticker-picker-arrow" style={{ left: pickerPosition.arrow }} aria-hidden="true" />
  </div>

  return <div className="chat-page">
    <header className="chat-toolbar glass-strip"><div className="chat-person"><div className="avatar-shell"><CompanionAvatar label={name} src={partner?.avatarUrl} /></div><div><h1>{name}</h1></div></div><div className="toolbar-actions">{searching && <input className="chat-search-input" autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索当前聊天" />}<Tooltip label={searching ? '关闭搜索' : '搜索聊天'}><Button variant="icon" aria-label={searching ? '关闭搜索' : '搜索聊天'} onClick={() => { setSearching(!searching); if (searching) setQuery('') }}>{searching ? <X size={17} /> : <Search size={17} />}</Button></Tooltip></div></header>
    <section className="messages-pane" ref={paneRef} onScroll={(event) => { if (event.currentTarget.scrollTop < 20) void loadOlderAtTop() }}>
      {(loadingEarlier || nextCursor) && <div className="conversation-date"><span>{loadingEarlier ? '正在加载更早消息...' : '向上滚动加载更早消息'}</span></div>}
      {loading ? <div className="inline-loading"><span /><span /><span /></div> : error && !messages.length ? <ErrorState message={error} onRetry={() => void loadLatest()} /> : <div className="message-stack">
        {visibleMessages.map((message, index) => {
          const prev = index > 0 ? visibleMessages[index - 1] : null
          const currDate = new Date(message.createdAt)
          const prevDate = prev ? new Date(prev.createdAt) : null
          const crossedDay = !prevDate || dayKey(currDate) !== dayKey(prevDate)
          const showDivider = crossedDay || currDate.getTime() - prevDate.getTime() >= TIME_GAP_MS
          return <Fragment key={message.localId}>
            {showDivider && <div className="conversation-date message-time-divider"><span>{formatDividerTime(message.createdAt, crossedDay)}</span></div>}
            <MessageRow message={message} partnerName={name} avatarUrl={partner?.avatarUrl} userNickname={profile?.nickname ?? '我'} userAvatarUrl={profile?.avatarUrl} retryDisabled={busy} onRetry={() => message.parentMessageId && void retry(message.parentMessageId)} playingVoiceKey={playingVoiceKey} setPlayingVoiceKey={setPlayingVoiceKey} />
          </Fragment>
        })}
        {!visibleMessages.length && <div className="empty-state"><Search size={20} /><strong>没有找到相关对话</strong><p>换一个关键词试试。</p></div>}
        {busy && <motion.article className="message-row assistant" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><div className="message-avatar"><CompanionAvatar label={name} src={partner?.avatarUrl} /></div><div className="thinking-indicator glass-surface"><span /><span /><span /><p>{streamStatus === 'using_tool' ? `${name}正在处理本地操作` : streamStatus === 'presenting' ? `${name}正在输入` : `${name}正在思考`}</p></div></motion.article>}
      </div>}
    </section>
    <footer className="composer-wrap">
      {!modelConfigured && <div className="model-warning"><Wrench size={15} /><span>配置本地模型后才能发送消息。</span><Link to="/settings">去设置</Link></div>}
      {activeTool && activeTool.toolName !== 'generate_image' && <div className="tool-progress"><FileCheck2 size={14} /><span>{activeTool.summary}</span><small>{activeTool.targetPaths[0]}</small></div>}
      <div className="composer glass-surface"><textarea ref={composerRef} value={text} disabled={loading || !modelConfigured} maxLength={2000} onChange={(event) => { draftRef.current = event.target.value; setText(event.target.value) }} placeholder={modelConfigured ? `和${name}说点什么...` : '请先配置模型'} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!event.repeat) sendMessage() } }} /><div className="composer-bottom"><div className="sticker-picker-anchor" ref={stickerAnchorRef}><Button variant="icon" className={stickersOpen ? 'is-open' : undefined} disabled={loading || !modelConfigured} onClick={() => { if (stickersOpen) setStickersOpen(false); else void openStickers() }} aria-label="表情包" title="表情包" aria-haspopup="dialog" aria-expanded={stickersOpen} aria-controls={stickersOpen ? 'chat-sticker-picker' : undefined}><Smile size={19} /></Button>{stickerPicker}</div><Tooltip label="发送消息"><Button variant="icon" className="send-button" disabled={sendDisabled} onClick={sendMessage} aria-label="发送消息"><ArrowUp size={18} /></Button></Tooltip></div></div>
    </footer>
  </div>
}

function StickerImage({ sticker, compact = false }: { sticker: StickerOption; compact?: boolean }) {
  const [loaded, setLoaded] = useState<{ id: number; src: string | null; failed: boolean } | null>(null)
  const src = loaded?.id === sticker.id ? loaded.src : null
  const failed = loaded?.id === sticker.id && loaded.failed
  useEffect(() => {
    let active = true, objectUrl: string | null = null
    void appApi.chat.stickerContent(sticker.id).then(blob => {
      if (!active) return
      objectUrl = URL.createObjectURL(blob)
      setLoaded({ id: sticker.id, src: objectUrl, failed: false })
    }).catch(() => { if (active) setLoaded({ id: sticker.id, src: null, failed: true }) })
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [sticker.id])
  return <span className={`chat-sticker${compact ? ' compact' : ''}`}>{src && !failed ? <img src={src} alt={sticker.description || '表情包'} onError={() => setLoaded({ id: sticker.id, src: null, failed: true })} /> : <span role="img" aria-label={sticker.description || '表情包'}>{failed ? '表情加载失败' : '加载中…'}</span>}</span>
}

function GeneratedImage({ image }: { image: GeneratedImageAttachment }) {
  const [loaded, setLoaded] = useState<{ id: string; src: string | null; failed: boolean } | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const src = loaded?.id === image.id ? loaded.src : null
  const failed = loaded?.id === image.id && loaded.failed
  useEffect(() => {
    let active = true, objectUrl: string | null = null
    void appApi.chat.imageContent(image.id).then(blob => {
      if (!active) return
      objectUrl = URL.createObjectURL(blob)
      setLoaded({ id: image.id, src: objectUrl, failed: false })
    }).catch(() => { if (active) setLoaded({ id: image.id, src: null, failed: true }) })
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [image.id])
  const label = image.description || '伴侣生成的图片'
  return <>
    <div className="chat-generated-image">
      {src && !failed ? (
        <button type="button" className="chat-generated-image-button" title="点击放大查看" aria-label={`放大查看：${label}`} onClick={() => setPreviewOpen(true)}>
          <img src={src} alt={label} onError={() => setLoaded({ id: image.id, src: null, failed: true })} />
        </button>
      ) : (
        <span className="chat-generated-image-placeholder" role="img" aria-label={label}>{failed ? '图片加载失败' : '图片加载中…'}</span>
      )}
    </div>
    <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
      <DialogContent className="image-preview-dialog">
        <DialogTitle>图片预览</DialogTitle>
        <DialogDescription>{label}</DialogDescription>
        {src && <div className="image-preview-stage"><img src={src} alt={label} /></div>}
        <div className="dialog-actions"><DialogClose asChild><Button variant="secondary">关闭</Button></DialogClose></div>
      </DialogContent>
    </Dialog>
  </>
}

type VoiceLoadState = { id: string; url: string | null; duration: number; status: 'loading' | 'ready' | 'failed' }

function VoiceBubble({ voice, transcript, storageKey, active, onActiveChange }: { voice: GeneratedVoiceAttachment; transcript: string; storageKey: string; active: boolean; onActiveChange: (playing: boolean) => void }) {
  const [loaded, setLoaded] = useState<VoiceLoadState>({ id: voice.id, url: null, duration: voice.duration, status: 'loading' })
  const [transcriptOpen, setTranscriptOpen] = useState(false)
  const [playbackError, setPlaybackError] = useState<{ id: string; message: string; retryable: boolean } | null>(null)
  const [played, markPlayed] = useVoicePlayed(storageKey)
  const audioRef = useRef<HTMLAudioElement>(null)
  const mountedRef = useRef(true)
  const pendingPlayRef = useRef(false)

  // Ignore the previous voice's load state for the one frame before its effect
  // resets, so a replaced attachment can never briefly play stale audio.
  const current = loaded.id === voice.id ? loaded : { id: voice.id, url: null, duration: voice.duration, status: 'loading' as const }
  const playbackFailure = playbackError?.id === voice.id ? playbackError : null
  const failure = current.status === 'failed' ? '语音加载失败' : playbackFailure?.message ?? null
  const hardFailure = current.status === 'failed' || Boolean(playbackFailure && !playbackFailure.retryable)
  const seconds = Math.min(60, Math.max(1, Math.round(current.duration || voice.duration || 1)))
  const width = Math.min(240, Math.max(88, 76 + seconds * 6))
  const showTranscript = transcriptOpen || Boolean(failure)
  const transcriptText = transcript.trim() || '这条语音没有可显示的文字。'

  useEffect(() => {
    mountedRef.current = true
    const audio = audioRef.current
    return () => {
      mountedRef.current = false
      // A filtered-out or removed message must not leave stale playback state
      // that could resume visually when the same message appears again.
      onActiveChange(false)
      if (!audio) return
      // Release the element and its pending playback when the chat page unmounts.
      audio.pause()
      audio.removeAttribute('src')
      audio.load()
      if (activeVoiceAudio === audio) activeVoiceAudio = null
    }
  }, [onActiveChange])

  useEffect(() => {
    let disposed = false
    let objectUrl: string | null = null
    void appApi.chat.voiceContent(voice.id).then(blob => {
      if (disposed) return
      objectUrl = URL.createObjectURL(blob)
      setLoaded({ id: voice.id, url: objectUrl, duration: voice.duration, status: 'ready' })
    }).catch(() => {
      if (disposed) return
      setLoaded({ id: voice.id, url: null, duration: voice.duration, status: 'failed' })
      setTranscriptOpen(true)
      onActiveChange(false)
    })
    return () => {
      disposed = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [voice.id, voice.duration, onActiveChange])

  const startPlayback = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (activeVoiceAudio && activeVoiceAudio !== audio) activeVoiceAudio.pause()
    activeVoiceAudio = audio
    if (audio.ended || (Number.isFinite(audio.duration) && audio.currentTime >= audio.duration - 0.05)) audio.currentTime = 0
    void audio.play().then(() => {
      if (mountedRef.current) setPlaybackError(null)
    }).catch((error: unknown) => {
      if (!mountedRef.current) return
      if (error instanceof DOMException && error.name === 'AbortError') return
      if (activeVoiceAudio === audio) activeVoiceAudio = null
      setTranscriptOpen(true)
      const blocked = error instanceof DOMException && error.name === 'NotAllowedError'
      setPlaybackError({ id: voice.id, message: blocked ? '浏览器阻止了播放，请再次点击语音' : '语音播放失败', retryable: blocked })
      onActiveChange(false)
    })
  }, [voice.id, onActiveChange])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (!active) {
      pendingPlayRef.current = false
      audio.pause()
      if (activeVoiceAudio === audio) activeVoiceAudio = null
      return
    }
    // A click made before the blob is ready is held here until it becomes ready.
    if (!pendingPlayRef.current || current.status !== 'ready' || !current.url) return
    pendingPlayRef.current = false
    startPlayback()
  }, [active, current.status, current.url, startPlayback])

  const togglePlayback = () => {
    if (active) {
      pendingPlayRef.current = false
      onActiveChange(false)
      return
    }
    onActiveChange(true)
    if (current.status === 'ready' && current.url) startPlayback()
    else pendingPlayRef.current = true
  }

  const failureClass = hardFailure ? ' is-failed' : active && current.status === 'ready' ? ' is-playing' : ''
  const loadingClass = current.status === 'loading' ? ' is-loading' : ''
  const buttonLabel = current.status === 'loading'
    ? `语音加载中，约${seconds}秒`
    : hardFailure
      ? '语音播放失败，文字已展开'
      : `${active ? '暂停' : '播放'}语音，${seconds}秒${played ? '' : '，未播放'}`

  return <div className="voice-bubble-wrapper" role="group" aria-label="语音消息">
    <button
      type="button"
      className={`voice-bubble${failureClass}${loadingClass}`}
      style={{ width }}
      disabled={hardFailure}
      aria-label={buttonLabel}
      aria-pressed={active}
      aria-busy={current.status === 'loading'}
      onClick={togglePlayback}
    >
      <span className="voice-wave" aria-hidden="true"><span /><span /><span /></span>
      <span>{seconds}"</span>
      {!played && !failure && <span className="voice-unread-dot" aria-hidden="true" />}
    </button>
    {!hardFailure && <button type="button" className="voice-transcript-toggle" aria-expanded={showTranscript} onClick={() => setTranscriptOpen(value => !value)}>{showTranscript ? '收起文字' : '转文字'}</button>}
    <audio
      ref={audioRef}
      src={current.status === 'ready' ? current.url ?? undefined : undefined}
      preload="metadata"
      onPlay={markPlayed}
      onEnded={(event) => { event.currentTarget.currentTime = 0; onActiveChange(false) }}
      onError={(event) => {
        if (!mountedRef.current) return
        const audio = event.currentTarget
        if (!audio.getAttribute('src')) return
        audio.pause()
        if (activeVoiceAudio === audio) activeVoiceAudio = null
        setTranscriptOpen(true)
        setPlaybackError({ id: voice.id, message: '语音播放失败', retryable: false })
        onActiveChange(false)
      }}
      onLoadedMetadata={(event) => {
        const duration = event.currentTarget.duration
        if (Number.isFinite(duration) && duration > 0) setLoaded(state => state.id === voice.id ? { ...state, duration } : state)
      }}
    />
    {showTranscript && <div className="voice-transcript-bubble" role="status" aria-live="polite">
      {failure && <p className="voice-transcript-error">{failure}，以下为文字内容</p>}
      <p className="voice-transcript-text">{transcriptText}</p>
    </div>}
  </div>
}

function MessageRow({ message, partnerName, avatarUrl, userNickname, userAvatarUrl, retryDisabled, onRetry, playingVoiceKey, setPlayingVoiceKey }: { message: ChatMessage; partnerName: string; avatarUrl?: string; userNickname: string; userAvatarUrl?: string; retryDisabled: boolean; onRetry: () => void; playingVoiceKey: string | null; setPlayingVoiceKey: Dispatch<SetStateAction<string | null>> }) {
  const failed = message.status === 'failed'
  const safety = !failed && message.messageType === 'safety'
  const isVoice = !failed && !safety && !!message.voice
  const voiceKey = message.id != null ? `voice-${message.id}` : message.localId
  const onVoiceActiveChange = useCallback((playing: boolean) => {
    setPlayingVoiceKey(current => playing ? voiceKey : current === voiceKey ? null : current)
  }, [voiceKey, setPlayingVoiceKey])
  return <motion.article className={`message-row ${message.role}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
    {message.role === 'assistant' && <div className="message-avatar"><CompanionAvatar label={partnerName} src={avatarUrl} /></div>}
    <div className="message-content">
      <div className="message-meta"><strong>{message.role === 'assistant' ? partnerName : userNickname}</strong></div>
      {isVoice ? <VoiceBubble key={voiceKey} voice={message.voice!} transcript={message.content} storageKey={voiceKey} active={playingVoiceKey === voiceKey} onActiveChange={onVoiceActiveChange} /> : (failed || safety || message.content.trim()) && <div className={safety ? 'safety-message' : 'message-bubble'}>
        {safety && <ShieldCheck size={17} />}
        {(failed || message.content) && <p>{failed ? '网络跑丢了' : message.content}</p>}
        {safety && message.helpResources.map((resource) => <button key={resource.url} onClick={() => void openExternal(resource.url)}>{resource.label}<ExternalLink size={13} /></button>)}
      </div>}
      {!failed && !isVoice && message.image && <div className="message-generated-image"><GeneratedImage image={message.image} /></div>}
      {!failed && !isVoice && message.sticker && <div className="message-sticker"><StickerImage sticker={message.sticker} /></div>}
      {failed && message.parentMessageId && <Tooltip label="重新回复"><Button variant="icon" className="message-retry" aria-label="重新回复" disabled={retryDisabled} onClick={onRetry}><RefreshCw size={14} /></Button></Tooltip>}
    </div>
    {message.role === 'user' && <div className="user-avatar"><CompanionAvatar label={userNickname} src={userAvatarUrl} /></div>}
  </motion.article>
}

function ErrorState({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="empty-state"><RefreshCw size={20} /><strong>聊天记录加载失败</strong><p>{message}</p><Button variant="secondary" onClick={onRetry}>重新加载</Button></div>
}
