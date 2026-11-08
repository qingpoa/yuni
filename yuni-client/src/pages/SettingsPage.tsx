import { Check, Database, ExternalLink, FileClock, FolderOpen, HardDriveDownload, HardDriveUpload, LoaderCircle, Monitor, NotebookTabs, Plus, RefreshCw, ScrollText, ShieldCheck, Trash2, Wrench, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { DirectoryBrowser } from '../components/DirectoryBrowser'
import { ModelSettingsDialog } from '../components/ModelSettingsDialog'
import { PageHeader } from '../components/PageHeader'
import { Button } from '../components/ui/button'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '../components/ui/dialog'
import { Switch } from '../components/ui/switch'
import { Select } from '../components/ui/select'
import { appApi } from '../services/api'
import { isTauriRuntime, openExternal, pickDirectory } from '../services/desktop'
import { useAppStore } from '../stores/appStore'
import { useChatStore } from '../stores/chatStore'
import { useMemoryStore } from '../stores/memoryStore'
import { usePartnerStore } from '../stores/partnerStore'
import { useProfileStore } from '../stores/profileStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useToolStore } from '../stores/toolStore'
import { useUiStore } from '../stores/uiStore'
import type { ToolCall } from '../types/domain'

export function SettingsPage() {
  const navigate = useNavigate()
  const bootstrap = useAppStore((state) => state.bootstrapData)
  const appBootstrap = useAppStore((state) => state.bootstrap)
  const setHasCompanion = useAppStore((state) => state.setHasCompanion)
  const setModelConfigured = useAppStore((state) => state.setModelConfigured)
  const { settings, saving, load, setMemoryEnabled, saveProactive } = useSettingsStore()
  const { roots, calls, loading: toolsLoading, load: loadTools, addRoot, removeRoot } = useToolStore()
  const notify = useUiStore((state) => state.notify)
  const importRef = useRef<HTMLInputElement>(null)
  const [clearOpen, setClearOpen] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [triggeringProactive, setTriggeringProactive] = useState(false)

  useEffect(() => { if (!settings) void load(); void loadTools() }, [load, loadTools, settings])

  const toggleMemory = async (enabled: boolean) => {
    try { await setMemoryEnabled(enabled); notify(enabled ? '长期记忆已开启' : '长期记忆已关闭', 'success') }
    catch { notify('记忆设置保存失败，已恢复原状态', 'error') }
  }

  const updateProactive = async (patch: Partial<{ proactiveChatEnabled: boolean; proactiveIntervalMinutes: number; quietHoursStart: string; quietHoursEnd: string }>) => {
    if (!settings) return
    try {
      await saveProactive({
        proactiveChatEnabled: patch.proactiveChatEnabled ?? settings.proactiveChatEnabled ?? true,
        proactiveIntervalMinutes: patch.proactiveIntervalMinutes ?? settings.proactiveIntervalMinutes ?? 120,
        quietHoursStart: patch.quietHoursStart ?? settings.quietHoursStart ?? '23:30',
        quietHoursEnd: patch.quietHoursEnd ?? settings.quietHoursEnd ?? '08:30',
      })
      notify('主动聊天设置已保存', 'success')
    } catch (error) {
      notify(error instanceof Error ? error.message : '主动聊天设置保存失败', 'error')
    }
  }

  const triggerProactiveNow = async () => {
    setTriggeringProactive(true)
    try {
      const msg = await appApi.chat.triggerProactive()
      useChatStore.getState().receiveProactive([msg])
      notify(`伴侣已主动发起消息：「${msg.content}」`, 'success')
      navigate('/chat')
    } catch (error) {
      notify(error instanceof Error ? error.message : '触发主动聊天失败', 'error')
    } finally {
      setTriggeringProactive(false)
    }
  }

  const exportData = async () => {
    try { const blob = await appApi.data.export(); const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `yuni-backup-${new Date().toISOString().slice(0, 10)}.json`; anchor.click(); URL.revokeObjectURL(url); notify('备份已导出', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '导出失败', 'error') }
  }

  const importData = async (file?: File) => {
    if (!file) return
    try { const result = await appApi.data.import(file); usePartnerStore.getState().reset(); useProfileStore.getState().reset(); useChatStore.getState().reset(); useMemoryStore.getState().reset(); useSettingsStore.getState().reset(); useToolStore.getState().reset(); await useProfileStore.getState().load(); await appBootstrap(); notify(`已导入 ${result.importedMessages} 条消息和 ${result.importedMemories} 条记忆`, 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '导入失败', 'error') }
    finally { if (importRef.current) importRef.current.value = '' }
  }

  const clearData = async () => {
    setClearing(true)
    try {
      await appApi.data.clear()
      usePartnerStore.getState().reset()
      useProfileStore.getState().reset()
      useChatStore.getState().reset()
      useMemoryStore.getState().reset()
      useSettingsStore.getState().reset()
      useToolStore.getState().reset()
      setHasCompanion(false)
      setModelConfigured(false)
      setClearOpen(false)
      notify('本地数据已清空', 'success')
      navigate('/partner/create', { replace: true })
    }
    catch (error) { notify(error instanceof Error ? error.message : '清空失败', 'error') }
    finally { setClearing(false) }
  }

  const privacyUrl = import.meta.env.VITE_PRIVACY_URL ?? 'https://example.com/privacy'
  const safetyUrl = import.meta.env.VITE_SAFETY_HELP_URL ?? 'https://example.com/help'

  return <div className="standard-page settings-page local-settings-page">
    <PageHeader eyebrow="偏好与管理" title="设置" description="让 Yuni 以你喜欢的方式陪伴你。" />
    <section className="settings-model-section" aria-label="模型配置"><ModelSettingsDialog /></section>

    <div className="settings-columns">
      <section className="settings-section"><div className="settings-heading settings-heading-row"><div><h2><NotebookTabs size={16} /> 长期记忆</h2><p>记住与你有关的事，仅在相关对话中使用。关闭后仍可管理已保存内容。</p></div><Switch checked={settings?.memoryEnabled ?? false} disabled={!settings || saving} onCheckedChange={(checked) => void toggleMemory(checked)} aria-label="长期记忆开关" /></div></section>
      <section className="settings-section"><div className="settings-heading settings-heading-row"><div><h2><Monitor size={16} /> 伴侣主动聊天</h2><p>安静一段时间后主动关心你；连续两轮未获回复后暂停。</p><p>免打扰时段 {settings?.quietHoursStart ?? '23:30'} — {settings?.quietHoursEnd ?? '08:30'}</p></div><Switch checked={settings?.proactiveChatEnabled ?? true} disabled={!settings || saving} onCheckedChange={(checked) => void updateProactive({ proactiveChatEnabled: checked })} aria-label="主动聊天开关" /></div><div className="setting-line setting-action-line"><div className="field-group"><span>主动找你的频率</span><Select label="主动找你的频率" value={String(settings?.proactiveIntervalMinutes ?? 120)} disabled={!settings || saving || !(settings.proactiveChatEnabled ?? true)} onValueChange={value => void updateProactive({ proactiveIntervalMinutes: Number(value) })} options={[{ value: '30', label: '高频黏人', description: '约 30～45 分钟' }, { value: '60', label: '较频繁', description: '约 1 小时' }, { value: '120', label: '日常标准', description: '约 2 小时' }, { value: '240', label: '安静克制', description: '约 4 小时' }]} /></div><Button variant="secondary" disabled={triggeringProactive || !settings?.model.configured} onClick={() => void triggerProactiveNow()}>{triggeringProactive ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />} 现在让她找我一次</Button></div></section>
      <section className="settings-section"><div className="settings-heading"><h2><Monitor size={16} /> Agent 状态</h2><p>本地服务与前端协议信息。</p></div><dl className="system-meta"><div><dt>Agent 版本</dt><dd>{bootstrap?.agentVersion ?? '--'}</dd></div><div><dt>协议版本</dt><dd>{bootstrap?.protocolVersion ?? '--'}</dd></div><div><dt>模型状态</dt><dd className={settings?.model.configured ? 'configured' : ''}>{settings?.model.configured ? '已配置' : '未配置'}</dd></div></dl><Button variant="secondary" onClick={() => void appApi.system.openLogs()}><ScrollText size={15} /> 打开日志目录</Button></section>
    </div>

    <section className="settings-section settings-wide"><div className="settings-heading settings-heading-row"><div><h2><FolderOpen size={16} /> 授权目录</h2><p>管理 Agent 可访问的目录，写入文件仍需单独确认。</p></div><AddRootDialog saving={toolsLoading} onAdd={async (path) => { await addRoot(path); notify('目录已授权', 'success') }} /></div><div className="root-list">{roots.length ? roots.map((root) => <div className="root-row" key={root.id}><span className="status-icon"><FolderOpen size={16} /></span><div><strong>{root.displayName}</strong><code>{root.path}</code></div><Button variant="icon" aria-label={`撤销 ${root.displayName} 授权`} onClick={() => void removeRoot(root.id)}><X size={15} /></Button></div>) : <div className="compact-empty">尚未授权任何目录，文件工具不会运行。</div>}</div></section>

    <section className="settings-section settings-wide"><div className="settings-heading"><h2><FileClock size={16} /> 工具调用记录</h2><p>所有本地文件操作都会留下状态与目标路径。</p></div><div className="tool-history">{calls.length ? calls.slice(0, 8).map((call) => <ToolHistoryRow key={call.id} call={call} />) : <div className="compact-empty">还没有工具调用记录。</div>}</div></section>

    <section className="settings-section settings-wide"><div className="settings-heading"><h2><Database size={16} /> 本地数据</h2><p>导出、导入或删除 SQLite 中的伴侣、聊天和记忆数据。</p></div><div className="data-actions"><Button variant="secondary" onClick={() => void exportData()}><HardDriveDownload size={15} /> 导出备份</Button><Button variant="secondary" onClick={() => importRef.current?.click()}><HardDriveUpload size={15} /> 导入备份</Button><Button variant="danger" onClick={() => setClearOpen(true)}><Trash2 size={15} /> 清空本地数据</Button><input ref={importRef} type="file" accept="application/json,.json" className="visually-hidden" onChange={(event) => void importData(event.target.files?.[0])} /></div></section>

    <section className="settings-section settings-wide settings-links-band"><button onClick={() => void openExternal(privacyUrl)}><ShieldCheck size={16} /><span><strong>隐私说明</strong><small>当前版本 {settings?.privacyVersion ?? 'v1.0'}</small></span><ExternalLink size={14} /></button><button onClick={() => void openExternal(safetyUrl)}><Wrench size={16} /><span><strong>安全帮助</strong><small>查看现实支持和应急资源</small></span><ExternalLink size={14} /></button></section>
    <ConfirmDialog open={clearOpen} onOpenChange={setClearOpen} title="清空所有本地数据？" description="伴侣、聊天、记忆、授权目录和模型配置将被删除。该操作无法撤销。" confirmLabel="全部清空" busy={clearing} onConfirm={clearData} />
  </div>
}

function AddRootDialog({ saving, onAdd }: { saving: boolean; onAdd: (path: string) => Promise<void> }) {
  const [open, setOpen] = useState(false); const [path, setPath] = useState(''); const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
  const [picking, setPicking] = useState(false)
  const busy = saving || adding || picking
  const selectPath = useCallback((value: string | null) => { setPath(value ?? ''); setError('') }, [])
  const submit = async () => {
    if (!path || busy) return
    setAdding(true)
    setError('')
    try { await onAdd(path); setPath(''); setOpen(false) }
    catch (caught) { setError(caught instanceof Error ? caught.message : '添加失败') }
    finally { setAdding(false) }
  }
  const choose = async () => {
    setPicking(true)
    setError('')
    try { const selected = await pickDirectory(); if (selected) setPath(selected) }
    catch (caught) { setError(caught instanceof Error ? caught.message : '无法打开目录选择器') }
    finally { setPicking(false) }
  }
  return <Dialog open={open} onOpenChange={(value) => { if (busy) return; setOpen(value); setPath(''); setError('') }}><DialogTrigger asChild><Button variant="secondary"><Plus size={15} /> 添加目录</Button></DialogTrigger><DialogContent className="directory-dialog" closeDisabled={busy}><DialogTitle>授权本地目录</DialogTitle><DialogDescription>只授权你愿意让 Yuni 读取的目录。写入仍会单独确认。</DialogDescription><div className="dialog-form">{isTauriRuntime() ? <Button variant="secondary" disabled={busy} onClick={() => void choose()}><FolderOpen size={15} />选择文件夹</Button> : open && <DirectoryBrowser disabled={busy} onSelect={selectPath} />}<div className="directory-selection"><span>已选择</span><code>{path || '尚未选择文件夹'}</code></div>{error && <p className="form-error" role="alert">{error}</p>}</div><div className="dialog-actions"><DialogClose asChild><Button variant="ghost" disabled={busy}>取消</Button></DialogClose><Button disabled={busy || !path} onClick={() => void submit()}>{adding ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}允许读取</Button></div></DialogContent></Dialog>
}

function ToolHistoryRow({ call }: { call: ToolCall }) {
  const labels: Record<ToolCall['status'], string> = { awaiting_confirmation: '等待确认', running: '执行中', succeeded: '已完成', failed: '失败', rejected: '已拒绝' }
  return <div className="tool-history-row"><span className={`tool-status status-${call.status}`} /> <div><strong>{call.summary}</strong><code>{call.targetPaths.join(' · ')}</code></div><span>{labels[call.status]}</span></div>
}
