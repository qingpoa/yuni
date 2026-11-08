import { AlertTriangle, Brain, ChevronRight, Eraser, LoaderCircle, Pencil, Plus, RefreshCw, Search, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { PageHeader } from '../components/PageHeader'
import { Button } from '../components/ui/button'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '../components/ui/dialog'
import { Switch } from '../components/ui/switch'
import { Tabs, TabsList, TabsTrigger } from '../components/ui/tabs'
import { cn } from '../lib/utils'
import { appApi } from '../services/api'
import { useMemoryStore } from '../stores/memoryStore'
import { usePartnerStore } from '../stores/partnerStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useUiStore } from '../stores/uiStore'
import type { MemoryDraft, MemoryFilter, MemoryItem, MemoryKind, MemoryPatch } from '../types/domain'

const tabs = [
  { value: 'fact', label: '事实与偏好' }, { value: 'ongoing_state', label: '近期状态' },
  { value: 'event', label: '经历与计划' }, { value: 'history', label: '历史与冲突' },
]
const eventLabels = { planned: '计划中', in_progress: '进行中', completed: '已完成', cancelled: '已取消' }
const statusLabels: Record<MemoryItem['status'], string> = { active: '当前记录', historical: '过往事实', conflicted: '待确认冲突', retracted: '已纠正作废', deleted: '已删除' }
const dateLabel = (value?: string | null) => value ? new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value)) : '未明确'
const sourceLabel = (item: MemoryItem) => item.sources?.[0]?.messageSentAt ? dateLabel(item.sources[0].messageSentAt) : item.sourceLabel
function badge(item: MemoryItem) {
  if (item.status !== 'active') return statusLabels[item.status]
  if (item.kind === 'event') return item.eventStatus ? eventLabels[item.eventStatus] : '经历或计划'
  if (item.kind === 'ongoing_state') {
    const days = item.freshUntil ? Math.ceil((new Date(item.freshUntil).getTime() - Date.now()) / 86400000) : 0
    return days > 0 ? `近期状态 · 保鲜剩 ${days} 天` : '近期未确认 · 不代表已恢复'
  }
  return '事实与偏好'
}

export function MemoryPage() {
  const partner = usePartnerStore(state => state.partner)
  const store = useMemoryStore()
  const load = store.load
  const { settings, saving: settingsSaving, load: loadSettings, setMemoryEnabled } = useSettingsStore()
  const notify = useUiStore(state => state.notify)
  const [tab, setTab] = useState('fact')
  const [query, setQuery] = useState('')
  const [entityId, setEntityId] = useState('')
  const [activeId, setActiveId] = useState<number | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<number | null>(null)
  const [clearOpen, setClearOpen] = useState(false)
  const [historyState, setHistoryState] = useState<{ id: number; rows: MemoryItem[]; error: string }>({ id: 0, rows: [], error: '' })
  const criteria = useMemo<MemoryFilter>(() => ({ ...(tab === 'history' ? { status: 'historical,conflicted,retracted' } : { kind: tab as MemoryKind, status: 'active' }), ...(entityId ? { entityId: Number(entityId) } : {}), q: query.trim() }), [tab, entityId, query])
  useEffect(() => { const timer = window.setTimeout(() => { void load('', criteria) }, query ? 250 : 0); return () => window.clearTimeout(timer) }, [criteria, load, query])
  useEffect(() => { if (!settings) void loadSettings() }, [settings, loadSettings])
  const active = store.items.find(item => item.id === activeId) ?? store.items[0] ?? null
  const selectedId = active?.id
  const selectedUpdatedAt = active?.updatedAt
  const history = historyState.id === selectedId ? historyState.rows : []
  const historyError = historyState.id === selectedId ? historyState.error : ''
  useEffect(() => {
    let cancelled = false
    if (selectedId) void appApi.memory.history(selectedId).then(rows => { if (!cancelled) setHistoryState({ id: selectedId, rows, error: '' }) }).catch(() => { if (!cancelled) setHistoryState({ id: selectedId, rows: [], error: '历史记录加载失败，可刷新重试' }) })
    return () => { cancelled = true }
  }, [selectedId, selectedUpdatedAt])
  const name = partner?.name ?? '伴侣'
  const run = async (operation: () => Promise<unknown>, message: string) => {
    try { await operation(); notify(message, 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '操作失败', 'error') }
  }
  return <div className="standard-page memory-page structured-memory-page">
    <PageHeader eyebrow="记忆中心" title={`${name}记住的事`} description="看看现在的你，也留住一路走来的变化。" actions={<><MemoryEditor saving={store.saving} onSave={async draft => { const item = await store.create(draft); setTab(item.kind ?? 'fact'); setActiveId(item.id); notify('记忆已新增', 'success') }} /><Button variant="icon" aria-label="刷新记忆" disabled={store.loading || store.saving} onClick={() => void store.load('', criteria)}><RefreshCw size={17} /></Button></>} />
    <section className="memory-control glass-strip"><div className="memory-control-main"><span className="status-icon status-icon-large"><Brain size={19} /></span><div><strong>{name}的长期记忆</strong><p>按话题使用相关记忆，时间和来源可以随时查看</p></div></div><div className="memory-toggle"><span>{settings?.memoryEnabled ? '已开启' : '已关闭'}</span><Switch checked={settings?.memoryEnabled ?? false} disabled={settingsSaving} onCheckedChange={enabled => void run(() => setMemoryEnabled(enabled), enabled ? '长期记忆已开启' : '长期记忆已关闭')} aria-label="长期记忆开关" /></div></section>
    <div className="memory-workspace">
      <section className="memory-browser">
        <div className="memory-tools"><Tabs value={tab} onValueChange={value => { setTab(value); setActiveId(null) }}><TabsList>{tabs.map(item => <TabsTrigger key={item.value} value={item.value}>{item.label}</TabsTrigger>)}</TabsList></Tabs></div>
        <div className="structured-memory-filters"><label className="memory-query"><Search size={15} /><input aria-label="搜索记忆" placeholder="搜索记忆" value={query} onChange={event => setQuery(event.target.value)} /></label><select aria-label="按人物或实体筛选" value={entityId} onChange={event => setEntityId(event.target.value)}><option value="">所有人物与事物</option>{store.entities.map(entity => <option key={entity.id} value={entity.id}>{entity.canonicalName}（{entity.memoryCount}）</option>)}</select></div>
        <div className="memory-list">
          {store.loading ? <div className="inline-loading"><span /><span /><span /></div> : store.error ? <div className="empty-state"><strong>记忆加载失败</strong><p>{store.error}</p><Button variant="secondary" onClick={() => void store.load('', criteria)}>重试</Button></div> : store.items.length ? store.items.map(item => <button key={item.id} className={cn('memory-row', active?.id === item.id && 'active', item.status === 'conflicted' && 'conflicted')} onClick={() => setActiveId(item.id)}><span className="memory-category">{badge(item)}</span><div><strong>{item.title}</strong><p>{item.timeText || item.qualifier || sourceLabel(item)}</p></div>{item.status === 'conflicted' ? <AlertTriangle size={16} /> : <ChevronRight size={16} />}</button>) : <div className="empty-state"><Brain size={20} /><strong>{query ? '没有匹配的记忆' : '这里还没有记忆'}</strong><p>可以手动记下一件事，也可以继续和伴侣聊天。</p></div>}
          {store.nextCursor && <Button variant="ghost" className="load-more" onClick={() => void store.loadMore()}>加载更多</Button>}
        </div>
        <Button variant="danger" className="clear-memory" disabled={store.saving} onClick={() => setClearOpen(true)}><Eraser size={16} />清空全部长期记忆</Button>
        {store.candidates.length > 0 && <details className="memory-candidates"><summary>待你确认的敏感记忆（{store.candidates.length}）</summary>{store.candidates.map(candidate => <article key={candidate.id}><strong>{candidate.title}</strong><p>{candidate.content}</p><div className="memory-candidate-actions"><Button variant="secondary" disabled={store.saving} onClick={() => void run(() => store.decideCandidate(candidate.id, 'approve'), '已保存这条记忆')}>允许记住</Button><Button variant="ghost" disabled={store.saving} onClick={() => void run(() => store.decideCandidate(candidate.id, 'reject'), '已拒绝保存')}>不保存</Button></div></article>)}</details>}
      </section>
      <aside className="memory-detail glass-surface">{active ? <>
        <div className="memory-detail-head"><span className="memory-category">{badge(active)}</span><div className="detail-actions">{['active','conflicted'].includes(active.status) && <MemoryEditor key={`${active.id}:${active.updatedAt}`} memory={active} saving={store.saving} onSave={async draft => { const { subjectKey: _subjectKey, subjectPerson: _subjectPerson, domain: _domain, sceneKey: _sceneKey, ...patch } = draft; const item = await store.update(active.id, patch); setActiveId(item.id); notify('记忆已更新', 'success') }} />}<Button variant="icon" aria-label="删除记忆" disabled={store.saving} onClick={() => setDeleteTarget(active.id)}><Trash2 size={16} /></Button></div></div>
        <h2>{active.title}</h2><p className="memory-detail-copy">{active.content}</p>
        {active.outcome && <p className="memory-detail-copy">结果：{active.outcome}</p>}
        {active.status === 'conflicted' && active.conflictContent && <div className="memory-conflict"><div><AlertTriangle size={16} /><strong>这两种说法还需要你确认</strong></div><p>{active.conflictContent}</p><div className="memory-conflict-actions"><Button variant="ghost" disabled={store.saving} onClick={() => void run(() => store.resolveConflict(active.id, active.content), '已保留当前说法')}>保留当前说法</Button><Button disabled={store.saving} onClick={() => void run(() => store.resolveConflict(active.id, active.conflictContent!, { action: 'adopt' }), '已采用另一说法')}>采用另一说法</Button><CoexistDialog memory={active} saving={store.saving} onSave={(qualifier, otherQualifier) => store.resolveConflict(active.id, active.content, { action: 'coexist', qualifier, otherQualifier })} /></div></div>}
        <dl className="memory-meta"><div><dt>关于谁</dt><dd>{active.subjectEntityId ? store.entities.find(entity => entity.id === active.subjectEntityId)?.canonicalName ?? '其他人物' : '你本人'}</dd></div><div><dt>记忆范围</dt><dd>{active.domain === 'fictional' ? `小剧场 · ${active.sceneKey}` : active.domain === 'conversation' ? '我们的聊天互动' : '现实生活'}</dd></div><div><dt>适用条件</dt><dd>{active.qualifier || '未限定'}</dd></div><div><dt>提到的时间</dt><dd>{active.timeText || '未明确'}</dd></div>{active.kind === 'event' ? <><div><dt>计划时间</dt><dd>{dateLabel(active.timeMeta?.planned_start)}</dd></div><div><dt>实际时间</dt><dd>{dateLabel(active.timeMeta?.actual_start)}</dd></div></> : <><div><dt>开始生效</dt><dd>{dateLabel(active.validFrom)}</dd></div>{active.validUntil && <div><dt>结束生效</dt><dd>{dateLabel(active.validUntil)}</dd></div>}</>}{active.kind === 'ongoing_state' && <div><dt>近期关注截至</dt><dd>{dateLabel(active.freshUntil)}</dd></div>}<div><dt>最后整理</dt><dd>{dateLabel(active.updatedAt)}</dd></div><div><dt>使用状态</dt><dd>{statusLabels[active.status]}</dd></div></dl>
        <details className="memory-history"><summary>查看历史演变（{history.length}）</summary>{historyError && <p className="form-error">{historyError}</p>}{history.map(item => <article key={item.id}><span>{statusLabels[item.status]} · {dateLabel(item.validFrom || item.createdAt)}</span><p>{item.content}</p></article>)}</details>
        <details className="memory-history"><summary>查看来源证据（{active.sources?.length ?? (active.sourceExcerpt ? 1 : 0)}）</summary>{active.sources?.length ? active.sources.map(source => <article key={source.messageId}><span>{dateLabel(source.messageSentAt)}</span><p>{source.evidenceText}</p></article>) : <p>{active.sourceExcerpt || '你在记忆中心手动整理的内容。'}</p>}</details>
        <div className="memory-privacy"><p>删除会阻止旧消息把它重新记回；你之后重新提起时，可以形成新的记忆。</p></div>
      </> : <div className="empty-detail"><Brain size={24} /><strong>选择一条记忆</strong><p>时间、来源和变化会显示在这里。</p></div>}</aside>
    </div>
    <ConfirmDialog open={deleteTarget !== null} onOpenChange={open => !open && setDeleteTarget(null)} title="删除这条记忆？" description="旧消息不会再将这条记忆自动保存回来。" confirmLabel="删除记忆" busy={store.saving} onConfirm={() => run(async () => { if (deleteTarget !== null) await store.remove(deleteTarget); setDeleteTarget(null) }, '记忆已删除')} />
    <ConfirmDialog open={clearOpen} onOpenChange={setClearOpen} title="清空全部长期记忆？" description="清空当前、历史和冲突记忆；已有聊天消息保留，之后的新消息仍可形成新记忆。" confirmLabel="全部清空" busy={store.saving} onConfirm={() => run(async () => { await store.clear(); setClearOpen(false) }, '全部长期记忆已清空')} />
  </div>
}

function MemoryEditor({ memory, saving, onSave }: { memory?: MemoryItem; saving: boolean; onSave: (draft: MemoryDraft & Pick<MemoryPatch, 'mode'>) => Promise<void> }) {
  const [open, setOpen] = useState(false)
  const initialKind = memory?.kind ?? 'fact'
  const [kind, setKind] = useState<MemoryKind>(initialKind)
  const [title, setTitle] = useState(memory?.title ?? '')
  const [subjectKey, setSubjectKey] = useState(memory?.subjectKey ?? '')
  const [subjectPerson, setSubjectPerson] = useState('本人')
  const [domain, setDomain] = useState<NonNullable<MemoryDraft['domain']>>(memory?.domain ?? 'real_world')
  const [sceneKey, setSceneKey] = useState(memory?.sceneKey ?? '')
  const [content, setContent] = useState(memory?.content ?? '')
  const [qualifier, setQualifier] = useState(memory?.qualifier ?? '')
  const [importance, setImportance] = useState<MemoryItem['importance']>(memory?.importance ?? 3)
  const [mode, setMode] = useState<'correct' | 'transition'>('correct')
  const [timeText, setTimeText] = useState(memory?.timeText ?? '')
  const [validFrom, setValidFrom] = useState('')
  const [planned, setPlanned] = useState('')
  const [eventStatus, setEventStatus] = useState<NonNullable<MemoryItem['eventStatus']>>(memory?.eventStatus ?? 'planned')
  const [outcome, setOutcome] = useState(memory?.outcome ?? '')
  const [error, setError] = useState('')
  const save = async () => {
    if (!title.trim() || !content.trim() || (!memory && !subjectKey.trim())) return setError('请填写主题、标题和内容')
    if (domain === 'fictional' && !sceneKey.trim()) return setError('请填写小剧场名称')
    try {
      await onSave({ type: memory?.type ?? (kind === 'event' ? 'important_event' : 'user_fact'), kind, title: title.trim(), content: content.trim(), subjectKey, subjectPerson: !subjectPerson.trim() || subjectPerson.trim() === '本人' ? 'user' : subjectPerson.trim(), domain, sceneKey: sceneKey || null, qualifier: qualifier.trim(), importance, mode, timeText,
        ...(validFrom ? { validFrom: new Date(validFrom).toISOString(), timePrecision: 'exact' as const } : {}),
        ...(kind === 'event' ? { eventStatus, outcome, timeMeta: { ...memory?.timeMeta, ...(planned ? { planned_start: new Date(planned).toISOString() } : {}) } } : {}) })
      setOpen(false)
      if (!memory) { setTitle(''); setContent(''); setSubjectKey(''); setTimeText(''); setValidFrom(''); setPlanned(''); setQualifier(''); setOutcome('') }
    } catch (caught) { setError(caught instanceof Error ? caught.message : '保存失败') }
  }
  return <Dialog open={open} onOpenChange={value => { setOpen(value); setError('') }}><DialogTrigger asChild>{memory ? <Button variant="icon" aria-label="编辑记忆"><Pencil size={16} /></Button> : <Button variant="secondary"><Plus size={15} />新增记忆</Button>}</DialogTrigger><DialogContent className="memory-editor-dialog"><DialogTitle>{memory ? '编辑记忆' : '手动记下一件事'}</DialogTitle><DialogDescription>{memory ? '修正错误会作废原说法；状态变化会保留过去的事实。' : '填写你希望伴侣记住的内容，未知时间可以留空。'}</DialogDescription><div className="dialog-form">
    {memory && <label className="field-group"><span>这次修改属于</span><select aria-label="修改方式" value={mode} onChange={event => setMode(event.target.value as typeof mode)}><option value="correct">修正之前的错误</option><option value="transition">状态发生变化，保留历史</option></select></label>}
    <label className="field-group"><span>记忆形态</span><select value={kind} onChange={event => setKind(event.target.value as MemoryKind)}><option value="fact">事实与偏好</option><option value="ongoing_state">近期状态</option><option value="event">经历与计划</option></select></label>
    {!memory && <><label className="field-group"><span>主题</span><input maxLength={200} placeholder="例如：居住城市、咖啡偏好" value={subjectKey} onChange={event => setSubjectKey(event.target.value)} /></label><label className="field-group"><span>关于谁（默认你本人）</span><input maxLength={100} placeholder="本人，或同事小王等具体称呼" value={subjectPerson} onChange={event => setSubjectPerson(event.target.value)} /></label><label className="field-group"><span>记忆范围</span><select value={domain} onChange={event => { const next = event.target.value as typeof domain; setDomain(next); if (next === 'conversation') setKind('event') }}><option value="real_world">现实生活</option><option value="conversation">我们的聊天互动</option><option value="fictional">小剧场</option></select></label>{domain === 'fictional' && <label className="field-group"><span>小剧场名称</span><input value={sceneKey} maxLength={200} onChange={event => setSceneKey(event.target.value)} /></label>}</>}
    <label className="field-group"><span>标题</span><input maxLength={200} value={title} onChange={event => setTitle(event.target.value)} /></label><label className="field-group"><span>记忆内容</span><textarea rows={4} maxLength={2000} value={content} onChange={event => setContent(event.target.value)} /></label><label className="field-group"><span>适用条件（可选）</span><input maxLength={200} placeholder="例如工作日、周末" value={qualifier} onChange={event => setQualifier(event.target.value)} /></label><label className="field-group"><span>时间描述（可选）</span><input maxLength={200} placeholder="例如十月初、每年10月3日" value={timeText} onChange={event => setTimeText(event.target.value)} /></label>
    {kind === 'event' ? <><label className="field-group"><span>事件状态</span><select value={eventStatus} onChange={event => setEventStatus(event.target.value as typeof eventStatus)}>{Object.entries(eventLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><label className="field-group"><span>计划时间（可选）</span><input type="datetime-local" value={planned} onChange={event => setPlanned(event.target.value)} /></label><label className="field-group"><span>结果（可选）</span><input maxLength={1000} value={outcome} onChange={event => setOutcome(event.target.value)} /></label></> : <label className="field-group"><span>{memory && mode === 'transition' ? '变化发生时间（留空表示现在）' : '开始生效时间（可选）'}</span><input type="datetime-local" value={validFrom} onChange={event => setValidFrom(event.target.value)} /></label>}
    <label className="field-group"><span>重要程度</span><input type="range" min="1" max="5" value={importance} onChange={event => setImportance(Number(event.target.value) as MemoryItem['importance'])} /></label>{error && <p className="form-error">{error}</p>}
  </div><div className="dialog-actions"><DialogClose asChild><Button variant="ghost">取消</Button></DialogClose><Button disabled={saving} onClick={() => void save()}>{saving && <LoaderCircle className="spin" size={15} />}保存记忆</Button></div></DialogContent></Dialog>
}

function CoexistDialog({ memory, saving, onSave }: { memory: MemoryItem; saving: boolean; onSave: (first: string, second: string) => Promise<void> }) {
  const [open, setOpen] = useState(false)
  const [first, setFirst] = useState(memory.qualifier ?? '')
  const [second, setSecond] = useState('')
  const [error, setError] = useState('')
  const save = async () => { if (!first.trim() || !second.trim() || first.trim() === second.trim()) return setError('请填写两个不同的适用条件'); try { await onSave(first.trim(), second.trim()); setOpen(false) } catch (caught) { setError(caught instanceof Error ? caught.message : '保存失败') } }
  return <Dialog open={open} onOpenChange={setOpen}><DialogTrigger asChild><Button variant="secondary">分别适用于不同情况</Button></DialogTrigger><DialogContent><DialogTitle>让两种说法并存</DialogTitle><DialogDescription>分别说明它们适用的情况，例如工作日与周末。</DialogDescription><div className="dialog-form"><label className="field-group"><span>当前说法的条件</span><input maxLength={200} value={first} onChange={event => setFirst(event.target.value)} /></label><label className="field-group"><span>另一说法的条件</span><input maxLength={200} value={second} onChange={event => setSecond(event.target.value)} /></label>{error && <p className="form-error">{error}</p>}</div><div className="dialog-actions"><Button disabled={saving} onClick={() => void save()}>确认并存</Button></div></DialogContent></Dialog>
}
