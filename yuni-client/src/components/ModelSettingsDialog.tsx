import { Check, ChevronDown, Database, Eye, EyeOff, Image as ImageIcon, KeyRound, LoaderCircle, MessageSquare, RefreshCw, Trash2, Volume2 } from 'lucide-react'
import { useEffect, useId, useRef, useState, useCallback, type ReactNode } from 'react'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from './ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Switch } from './ui/switch'
import { Select } from './ui/select'
import { VoicePicker } from './VoicePicker'
import { builtinVoiceChoices, sameVoiceEndpoint, voiceService } from './voiceOptions'
import { appApi } from '../services/api'
import { useSettingsStore } from '../stores/settingsStore'
import { useUiStore } from '../stores/uiStore'
import type { ImageConfigInput, ImageConfigSummary, ImageModelSize, ModelConfigInput, ModelConfigSummary, VoiceConfigInput, VoiceConfigSummary, VoiceTriggerMode } from '../types/domain'

export function ModelSettingsDialog() {
  const { settings, saving, saveModel, clearModelKey, saveEmbedding, clearEmbeddingKey, saveImage, clearImageKey, saveVoice, clearVoiceKey } = useSettingsStore()
  const [open, setOpen] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [previewEpoch, setPreviewEpoch] = useState(0)
  const busy = saving || rebuilding
  const optionalConfigured = [
    Boolean(settings?.image?.enabled && settings?.image?.configured),
    Boolean(settings?.voice?.enabled && settings?.voice?.configured),
    Boolean(settings?.embedding?.configured),
  ].filter(Boolean).length
  const statusText = !settings
    ? '加载中'
    : settings.model.configured
      ? optionalConfigured
        ? `已就绪 · 含 ${optionalConfigured} 项扩展`
        : '聊天模型已就绪'
      : '待配置聊天模型'

  return <Dialog open={open} onOpenChange={value => { if (!busy) setOpen(value) }}>
    <DialogTrigger asChild>
      <button type="button" className="model-settings-entry" disabled={!settings}>
        <span className="settings-entry-icon"><KeyRound size={21} /></span>
        <span className="settings-entry-copy"><strong>模型配置</strong><span>必填聊天模型；语音、生图、向量按需选配</span></span>
        <span className={`settings-entry-status${settings?.model.configured ? ' ready' : ''}`}>{statusText}</span>
        <ChevronDown className="settings-entry-chevron" size={18} />
      </button>
    </DialogTrigger>
    <DialogContent className="model-settings-dialog" closeDisabled={busy} onEscapeKeyDown={event => {
      const target = event.target as HTMLElement
      if (busy || document.querySelector('.app-select-content') || target.closest('.model-picker')?.querySelector('[aria-expanded="true"]')) event.preventDefault()
    }} onPointerDownOutside={event => event.preventDefault()}>
      <header className="model-settings-header">
        <span className="model-settings-mark"><KeyRound size={20} /></span>
        <div><DialogTitle>模型配置</DialogTitle><DialogDescription>只需配置「聊天模型」即可开始对话；语音、生图和向量检索均为选填扩展，未启用时不会调用或注入提示词。</DialogDescription></div>
      </header>
      {settings && <Tabs defaultValue="chat" className="model-settings-tabs" onValueChange={() => setPreviewEpoch(value => value + 1)}>
        <TabsList className="model-settings-tab-list" aria-label="模型用途">
          <TabsTrigger value="chat" disabled={busy}><MessageSquare size={15} />聊天模型<span className={`tab-badge${settings.model.configured ? ' is-ready' : ' is-required'}`}>{settings.model.configured ? '已就绪' : '必填'}</span></TabsTrigger>
          <TabsTrigger value="voice" disabled={busy}><Volume2 size={15} />语音合成<span className={`tab-badge${settings.voice?.enabled && settings.voice?.configured ? ' is-ready' : ''}`}>{settings.voice?.enabled && settings.voice?.configured ? '已启用' : '选填'}</span></TabsTrigger>
          <TabsTrigger value="image" disabled={busy}><ImageIcon size={15} />图像生成<span className={`tab-badge${settings.image?.enabled && settings.image?.configured ? ' is-ready' : ''}`}>{settings.image?.enabled && settings.image?.configured ? '已启用' : '选填'}</span></TabsTrigger>
          <TabsTrigger value="embedding" disabled={busy}><Database size={15} />向量检索<span className={`tab-badge${settings.embedding?.configured ? ' is-ready' : ''}`}>{settings.embedding?.configured ? '已启用' : '选填'}</span></TabsTrigger>
        </TabsList>
        <div className="model-settings-panels">
          <TabsContent value="chat" forceMount className="model-settings-panel">
            <div className="model-panel-intro"><h3>聊天模型（唯一必填）</h3><p>用于日常陪伴对话、动作规划和记忆整理。支持一键填入常用服务商预设。</p></div>
            <ModelConfigForm initial={settings.model} saving={saving} onSave={saveModel} onClear={clearModelKey} onClose={() => setOpen(false)} />
          </TabsContent>
          <TabsContent value="voice" forceMount className="model-settings-panel">
            <div className="model-panel-intro"><h3>语音合成（选填扩展）</h3><p>开启后，伴侣可以发送微信式语音消息。未开启时只使用文字对话。</p></div>
            <VoiceConfigForm initial={settings.voice ?? { enabled: false, baseUrl: '', modelName: '', voice: '', triggerMode: 'occasional', configured: false }} mainModel={settings.model} saving={saving} previewEpoch={previewEpoch} onSave={saveVoice} onClear={clearVoiceKey} onClose={() => setOpen(false)} />
          </TabsContent>
          <TabsContent value="image" forceMount className="model-settings-panel">
            <div className="model-panel-intro"><h3>图像生成（选填扩展）</h3><p>开启后，伴侣可以按你的要求发自拍或画插画。未开启时不向模型暴露生图工具。</p></div>
            <ImageConfigForm initial={settings.image ?? { enabled: false, baseUrl: '', modelName: '', defaultSize: '1024x1024', stylePrefix: '', configured: false }} mainModel={settings.model} saving={saving} onSave={saveImage} onClear={clearImageKey} onClose={() => setOpen(false)} />
          </TabsContent>
          <TabsContent value="embedding" forceMount className="model-settings-panel">
            <div className="model-panel-intro"><h3>向量检索 Embedding（选填增强）</h3><p>不配置也能正常使用长期记忆（自动使用本地关键词检索）；配置后可增强同义词与语义联想检索。</p></div>
            <EmbeddingConfigForm initial={settings.embedding ?? { provider: 'openai_compatible', baseUrl: '', modelName: '', configured: false }} mainModel={settings.model} saving={saving} onSave={saveEmbedding} onClear={clearEmbeddingKey} rebuilding={rebuilding} setRebuilding={setRebuilding} onClose={() => setOpen(false)} />
          </TabsContent>
        </div>
      </Tabs>}
    </DialogContent>
  </Dialog>
}

function ApiKeyField({ value, onChange, configured, placeholder, disabled }: { value: string; onChange: (value: string) => void; configured: boolean; placeholder?: string; disabled: boolean }) {
  const id = useId()
  const [visible, setVisible] = useState(false)
  return <div className="field-group api-key-field">
    <div className="api-key-heading"><label htmlFor={id}>API Key</label><span className={configured ? 'key-configured' : ''}>{configured ? <><Check size={12} />已保存</> : '未保存'}</span></div>
    <div className="api-key-control"><input id={id} type={visible ? 'text' : 'password'} autoComplete="off" spellCheck={false} value={value} onChange={event => onChange(event.target.value)} disabled={disabled} placeholder={placeholder ?? (configured ? '留空保留当前 Key' : '输入 API Key')} /><button type="button" disabled={disabled} aria-label={visible ? '隐藏 API Key' : '显示 API Key'} onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button></div>
  </div>
}

function ConfigFooter({ saving, onClose, children }: { saving: boolean; onClose: () => void; children?: ReactNode }) {
  return <div className="model-actions"><div className="model-secondary-actions">{children}</div><div className="model-primary-actions"><Button type="button" variant="secondary" disabled={saving} onClick={onClose}>取消</Button><Button type="submit" disabled={saving}>{saving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}保存配置</Button></div></div>
}

const CHAT_PROVIDER_PRESETS: Array<{ label: string; provider: ModelConfigInput['provider']; baseUrl: string; modelName: string }> = [
  { label: 'DeepSeek 官方', provider: 'openai_compatible', baseUrl: 'https://api.deepseek.com/v1', modelName: 'deepseek-chat' },
  { label: '硅基流动 SiliconFlow', provider: 'openai_compatible', baseUrl: 'https://api.siliconflow.cn/v1', modelName: 'deepseek-ai/DeepSeek-V3' },
  { label: '阿里百炼 DashScope', provider: 'openai_compatible', baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1', modelName: 'qwen-plus' },
  { label: 'OpenAI 官方', provider: 'openai', baseUrl: 'https://api.openai.com/v1', modelName: 'gpt-4o-mini' },
]

const EMBEDDING_PRESETS: Array<{ label: string; baseUrl: string; modelName: string }> = [
  { label: '硅基流动 BGE-M3', baseUrl: 'https://api.siliconflow.cn/v1', modelName: 'BAAI/bge-m3' },
  { label: '阿里百炼 Text-Embedding-V3', baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1', modelName: 'text-embedding-v3' },
  { label: 'OpenAI Embedding-3-Small', baseUrl: 'https://api.openai.com/v1', modelName: 'text-embedding-3-small' },
]

const IMAGE_PRESETS: Array<{ label: string; baseUrl: string; modelName: string }> = [
  { label: '硅基流动 Kolors', baseUrl: 'https://api.siliconflow.cn/v1', modelName: 'Kwai-Kolors/Kolors' },
  { label: 'OpenAI GPT-Image', baseUrl: 'https://api.openai.com/v1', modelName: 'gpt-image-1' },
]

const VOICE_SERVICE_PRESETS: Array<{ label: string; baseUrl: string; modelName: string; voice: string }> = [
  { label: 'MiniMax 高清语音 (推荐)', baseUrl: 'https://api.minimax.cn/v1', modelName: 'speech-2.8-hd', voice: 'female-shaonv' },
  { label: 'MiniMax 极速语音', baseUrl: 'https://api.minimax.cn/v1', modelName: 'speech-2.8-turbo', voice: 'female-shaonv' },
  { label: '硅基流动 CosyVoice2', baseUrl: 'https://api.siliconflow.cn/v1', modelName: 'FunAudioLLM/CosyVoice2-0.5B', voice: 'FunAudioLLM/CosyVoice2-0.5B:anna' },
  { label: 'OpenAI TTS', baseUrl: 'https://api.openai.com/v1', modelName: 'gpt-4o-mini-tts', voice: 'alloy' },
]

function ModelConfigForm({ initial, saving, onSave, onClear, onClose }: { initial: ModelConfigInput & { configured: boolean }; saving: boolean; onSave: (config: ModelConfigInput) => Promise<void>; onClear: () => Promise<void>; onClose: () => void }) {
  const notify = useUiStore((state) => state.notify)
  const [provider, setProvider] = useState(initial.provider)
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl)
  const [modelName, setModelName] = useState(initial.modelName)
  const [apiKey, setApiKey] = useState('')
  const clear = async () => {
    try { await onClear(); setApiKey(''); notify('聊天模型 Key 已删除', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '删除失败', 'error') }
  }
  const submit = async () => {
    if (!baseUrl.trim() || !modelName.trim()) return notify('请填写 Base URL 和模型名', 'error')
    try { await onSave({ provider, baseUrl: baseUrl.trim(), modelName: modelName.trim(), apiKey: apiKey.trim() || undefined }); setApiKey(''); notify('模型配置已保存', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '模型配置保存失败', 'error') }
  }
  return <form className="model-config-form" onSubmit={(event) => { event.preventDefault(); void submit() }}>
    <div className="model-config-grid">
      <div className="field-group">
        <span>一键填入常用服务商</span>
        <div className="voice-presets" role="group" aria-label="聊天模型常用服务商">
          {CHAT_PROVIDER_PRESETS.map(preset => (
            <button
              key={preset.label}
              type="button"
              className={baseUrl.trim().replace(/\/+$/, '') === preset.baseUrl ? 'active' : undefined}
              disabled={saving}
              onClick={() => { setProvider(preset.provider); setBaseUrl(preset.baseUrl); setModelName(preset.modelName) }}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>
      <label className="field-group"><span>接口地址 (Base URL)</span><input value={baseUrl} disabled={saving} spellCheck={false} onChange={(event) => setBaseUrl(event.target.value)} placeholder="https://api.deepseek.com/v1" /></label>
      <ModelPicker key={`${baseUrl}\n${apiKey}\n${initial.configured}`} label="模型名称" placeholder="输入模型 ID，或点击右侧获取列表" value={modelName} onChange={setModelName} disabled={saving} loadModels={() => appApi.settings.listModels({ baseUrl: baseUrl.trim(), apiKey: apiKey.trim() || undefined })} />
      <ApiKeyField value={apiKey} onChange={setApiKey} configured={initial.configured} disabled={saving} />
      <p className="model-storage-note">兼容所有 OpenAI 格式接口。配置仅保存在本机 auth.json 中，请勿分享。</p>
    </div>
    <ConfigFooter saving={saving} onClose={onClose}>{initial.configured && <Button type="button" variant="ghost" className="model-remove-key" disabled={saving} onClick={() => void clear()}><Trash2 size={14} />移除 Key</Button>}</ConfigFooter>
  </form>
}

function EmbeddingConfigForm({ initial, mainModel, saving, onSave, onClear, onClose, rebuilding, setRebuilding }: { initial: ModelConfigInput & { configured: boolean }; mainModel: ModelConfigSummary; saving: boolean; onSave: (config: ModelConfigInput) => Promise<void>; onClear: () => Promise<void>; onClose: () => void; rebuilding: boolean; setRebuilding: (value: boolean) => void }) {
  const notify = useUiStore((state) => state.notify)
  const [enabled, setEnabled] = useState(Boolean(initial.configured || initial.baseUrl || initial.modelName))
  const [customBaseUrl, setCustomBaseUrl] = useState<string | null>(initial.baseUrl ? initial.baseUrl : null)
  const baseUrl = customBaseUrl ?? (initial.baseUrl || mainModel.baseUrl || '')
  const [modelName, setModelName] = useState(initial.modelName)
  const [apiKey, setApiKey] = useState('')
  const sameAsMainUrl = Boolean(baseUrl.trim() && mainModel.baseUrl && baseUrl.trim().replace(/\/+$/, '') === mainModel.baseUrl.trim().replace(/\/+$/, ''))

  const submit = async () => {
    if (!enabled) {
      if (initial.configured) {
        try { await onClear(); setApiKey(''); notify('已关闭向量检索，自动使用本地关键词检索', 'success') }
        catch (error) { notify(error instanceof Error ? error.message : '关闭失败', 'error') }
      } else {
        notify('已保持本地关键词检索模式', 'info')
      }
      return
    }
    if (!baseUrl.trim() || !modelName.trim()) return notify('启用向量检索需要填写接口地址和模型名', 'error')
    try { await onSave({ provider: 'openai_compatible', baseUrl: baseUrl.trim(), modelName: modelName.trim(), apiKey: apiKey.trim() || undefined }); setApiKey(''); notify('Embedding 配置已保存', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : 'Embedding 保存失败', 'error') }
  }
  const clear = async () => {
    try { await onClear(); setApiKey(''); setEnabled(false); notify('Embedding Key 已删除，已切换为本地关键词检索', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '删除失败', 'error') }
  }
  const rebuild = async () => {
    setRebuilding(true)
    try { await appApi.settings.rebuildExpressions(); notify('记忆与表达索引已重置，后台将逐批重建', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '重建失败', 'error') }
    finally { setRebuilding(false) }
  }
  return <form className="model-config-form" onSubmit={(event) => { event.preventDefault(); void submit() }}>
    <div className="model-config-grid">
      <div className="model-enabled-line">
        <div>
          <strong>启用独立向量检索 (Embedding)</strong>
          <p>{enabled ? '已开启语义向量检索增强' : '当前未启用：长期记忆自动使用零配置的本地 BM25 关键词检索'}</p>
        </div>
        <Switch checked={enabled} disabled={saving || rebuilding} onCheckedChange={setEnabled} aria-label="向量检索开关" />
      </div>
      {enabled ? <>
        <div className="field-group">
          <span>常用向量模型预设</span>
          <div className="voice-presets" role="group" aria-label="常用向量模型预设">
            {EMBEDDING_PRESETS.map(preset => (
              <button
                key={preset.label}
                type="button"
                className={modelName.trim() === preset.modelName ? 'active' : undefined}
                disabled={saving || rebuilding}
                onClick={() => { setCustomBaseUrl(preset.baseUrl); setModelName(preset.modelName) }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <label className="field-group"><span>接口地址 (Base URL)</span><input value={baseUrl} onChange={(event) => setCustomBaseUrl(event.target.value)} disabled={saving || rebuilding} placeholder="https://api.siliconflow.cn/v1" /></label>
        <label className="field-group"><span>向量模型名称</span><input value={modelName} onChange={(event) => setModelName(event.target.value)} disabled={saving || rebuilding} placeholder="如 BAAI/bge-m3 或 text-embedding-v3" /></label>
        <ApiKeyField value={apiKey} onChange={setApiKey} configured={initial.configured} disabled={saving || rebuilding} placeholder={initial.configured ? '留空保留当前 Key' : sameAsMainUrl && mainModel.configured ? '留空自动复用聊天模型的 Key' : '输入 API Key'} />
        <div className="model-maintenance"><div><strong>记忆与表达索引</strong><p>更换向量模型后，可重建已有内容的向量索引。</p></div><Button type="button" variant="secondary" disabled={saving || rebuilding || !initial.configured} onClick={() => void rebuild()}>{rebuilding ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}重建索引</Button></div>
        <p className="model-storage-note">独立保存在本机 embedding-auth.json 中。与聊天模型地址相同时，API Key 留空即可自动复用。</p>
      </> : (
        <div className="model-collapsed-hint">
          <p>无需配置此项也能正常使用「长期记忆」与「表达学习」功能。只有当你希望提升同义词模糊匹配效果时，再开启并配置向量模型即可。</p>
        </div>
      )}
    </div>
    <ConfigFooter saving={saving || rebuilding} onClose={onClose}>{initial.configured && <Button type="button" variant="ghost" className="model-remove-key" disabled={saving || rebuilding} onClick={() => void clear()}><Trash2 size={14} />移除 Key</Button>}</ConfigFooter>
  </form>
}

function ImageConfigForm({ initial, mainModel, saving, onSave, onClear, onClose }: { initial: ImageConfigSummary; mainModel: ModelConfigSummary; saving: boolean; onSave: (config: ImageConfigInput) => Promise<void>; onClear: () => Promise<void>; onClose: () => void }) {
  const notify = useUiStore((state) => state.notify)
  const [enabled, setEnabled] = useState(initial.enabled)
  const [customBaseUrl, setCustomBaseUrl] = useState<string | null>(initial.baseUrl ? initial.baseUrl : null)
  const baseUrl = customBaseUrl ?? (initial.baseUrl || mainModel.baseUrl || '')
  const [modelName, setModelName] = useState(initial.modelName)
  const [apiKey, setApiKey] = useState('')
  const [defaultSize, setDefaultSize] = useState<ImageModelSize>(initial.defaultSize ?? '1024x1024')
  const [stylePrefix, setStylePrefix] = useState(initial.stylePrefix ?? '')
  const sameAsMainUrl = Boolean(baseUrl.trim() && mainModel.baseUrl && baseUrl.trim().replace(/\/+$/, '') === mainModel.baseUrl.trim().replace(/\/+$/, ''))

  const submit = async () => {
    if (enabled && (!baseUrl.trim() || !modelName.trim())) {
      return notify('启用生图需要填写 Base URL 和模型名', 'error')
    }
    try {
      await onSave({
        enabled,
        baseUrl: baseUrl.trim(),
        modelName: modelName.trim(),
        apiKey: apiKey.trim() || undefined,
        defaultSize,
        stylePrefix: stylePrefix.trim(),
      })
      setApiKey('')
      notify('生图配置已保存', 'success')
    } catch (error) {
      notify(error instanceof Error ? error.message : '生图配置保存失败', 'error')
    }
  }

  const clear = async () => {
    try {
      await onClear()
      setApiKey('')
      notify('生图 API Key 已删除', 'success')
    } catch (error) {
      notify(error instanceof Error ? error.message : '删除失败', 'error')
    }
  }

  return <form className="model-config-form" onSubmit={(event) => { event.preventDefault(); void submit() }}>
    <div className="model-config-grid">
      <div className="model-enabled-line">
        <div>
          <strong>启用图像生成</strong>
          <p>{enabled ? '允许伴侣在对话中发自拍或按要求画图' : '当前未启用：伴侣不会调用生图工具，也不注入生图提示词'}</p>
        </div>
        <Switch checked={enabled} disabled={saving} onCheckedChange={setEnabled} aria-label="生图功能开关" />
      </div>
      {enabled ? <>
        <div className="field-group">
          <span>常用生图服务预设</span>
          <div className="voice-presets" role="group" aria-label="常用生图服务预设">
            {IMAGE_PRESETS.map(preset => (
              <button
                key={preset.label}
                type="button"
                className={modelName.trim() === preset.modelName ? 'active' : undefined}
                disabled={saving}
                onClick={() => { setCustomBaseUrl(preset.baseUrl); setModelName(preset.modelName) }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <label className="field-group"><span>接口地址 (Base URL)</span><input value={baseUrl} spellCheck={false} onChange={(event) => setCustomBaseUrl(event.target.value)} disabled={saving} placeholder="https://api.example.com/v1" /></label>
        <label className="field-group"><span>生图模型名</span><input value={modelName} spellCheck={false} onChange={(event) => setModelName(event.target.value)} disabled={saving} placeholder="如 Kwai-Kolors/Kolors 或 gpt-image-1" /></label>
        <ApiKeyField value={apiKey} onChange={setApiKey} configured={initial.configured} disabled={saving} placeholder={initial.configured ? '留空保留当前 Key' : sameAsMainUrl && mainModel.configured ? '留空自动复用聊天模型的 Key' : '输入 API Key'} />
        <div className="field-group">
          <span>默认画幅</span>
          <Select label="默认画幅" value={defaultSize} disabled={saving} onValueChange={value => setDefaultSize(value as ImageModelSize)} options={[
            { value: '1024x1024', label: '方形 1:1', description: '1024 × 1024' },
            { value: '768x1024', label: '竖构图 3:4', description: '768 × 1024' },
            { value: '1024x768', label: '横构图 4:3', description: '1024 × 768' },
          ]} />
        </div>
        <label className="field-group image-style-field"><span>全局画风前缀（可选）</span><input value={stylePrefix} maxLength={300} onChange={(event) => setStylePrefix(event.target.value)} disabled={saving} placeholder="可选，如：清透水彩插画风、柔和自然光" /></label>
        <p className="model-storage-note">独立保存在本机 image-auth.json 中。与聊天模型地址相同时，留空可复用聊天模型 Key。</p>
      </> : (
        <div className="model-collapsed-hint">
          <p>如果你不需要伴侣发自拍或画图，保持关闭即可。开启后，若生图服务与聊天模型使用同一平台，API Key 留空即可自动复用。</p>
        </div>
      )}
    </div>
    <ConfigFooter saving={saving} onClose={onClose}>{initial.configured && <Button type="button" variant="ghost" className="model-remove-key" disabled={saving} onClick={() => void clear()}><Trash2 size={14} />移除 Key</Button>}</ConfigFooter>
  </form>
}

function VoiceConfigForm({ initial, mainModel, saving, previewEpoch, onSave, onClear, onClose }: { initial: VoiceConfigSummary; mainModel: ModelConfigSummary; saving: boolean; previewEpoch: number; onSave: (config: VoiceConfigInput) => Promise<void>; onClear: () => Promise<void>; onClose: () => void }) {
  const notify = useUiStore((state) => state.notify)
  const [enabled, setEnabled] = useState(initial.enabled)
  const [customBaseUrl, setCustomBaseUrl] = useState<string | null>(initial.baseUrl ? initial.baseUrl : null)
  const fallbackBaseUrl = initial.baseUrl || mainModel.baseUrl || ''
  const baseUrl = customBaseUrl ?? fallbackBaseUrl
  const [modelName, setModelName] = useState(initial.modelName)
  const isMiniMax = voiceService(baseUrl, modelName) === 'minimax'
  const [apiKey, setApiKey] = useState('')
  const [voice, setVoice] = useState(initial.voice || builtinVoiceChoices(fallbackBaseUrl, initial.modelName)[0]?.id || '')
  const [triggerMode, setTriggerMode] = useState<VoiceTriggerMode>(initial.triggerMode ?? 'occasional')
  const [previewing, setPreviewing] = useState(false)
  const previewRef = useRef<{ audio: HTMLAudioElement; url: string } | null>(null)
  const previewGeneration = useRef(0)
  const previewEpochRef = useRef(previewEpoch)
  const mountedRef = useRef(false)
  const sameAsMainUrl = Boolean(baseUrl.trim() && mainModel.baseUrl && baseUrl.trim().replace(/\/+$/, '') === mainModel.baseUrl.trim().replace(/\/+$/, ''))
  const resolvedVoice = voice.trim()

  const stopPreview = useCallback(() => {
    previewGeneration.current += 1
    if (mountedRef.current) setPreviewing(false)
    const current = previewRef.current
    if (!current) return
    previewRef.current = null
    current.audio.pause()
    current.audio.removeAttribute('src')
    current.audio.load()
    URL.revokeObjectURL(current.url)
  }, [])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      stopPreview()
    }
  }, [stopPreview])

  useEffect(() => {
    previewEpochRef.current = previewEpoch
    stopPreview()
  }, [previewEpoch, baseUrl, modelName, voice, apiKey, enabled, stopPreview])

  const submit = async () => {
    if (enabled && (!baseUrl.trim() || !modelName.trim() || !voice.trim())) return notify('启用语音需要填写 Base URL、模型名和音色', 'error')
    try {
      await onSave({ enabled, baseUrl: baseUrl.trim(), modelName: modelName.trim(), apiKey: apiKey.trim() || undefined, voice: resolvedVoice, triggerMode })
      stopPreview()
      setApiKey('')
      notify('语音配置已保存', 'success')
    } catch (error) {
      notify(error instanceof Error ? error.message : '语音配置保存失败', 'error')
    }
  }

  const preview = async () => {
    if (!baseUrl.trim() || !modelName.trim() || !voice.trim()) return notify('请先填写 Base URL、模型名和音色', 'error')
    stopPreview()
    const generation = ++previewGeneration.current
    const epoch = previewEpochRef.current
    setPreviewing(true)
    try {
      const blob = await appApi.settings.previewVoice({ baseUrl: baseUrl.trim(), modelName: modelName.trim(), apiKey: apiKey.trim() || undefined, voice: voice.trim() })
      if (!mountedRef.current || generation !== previewGeneration.current || epoch !== previewEpochRef.current) return
      const url = URL.createObjectURL(blob)
      const audio = new Audio()
      let failed = false
      const fail = (message: string) => {
        if (failed || generation !== previewGeneration.current) return
        failed = true
        stopPreview()
        if (mountedRef.current) notify(message, 'error')
      }
      audio.addEventListener('ended', stopPreview, { once: true })
      audio.addEventListener('error', () => fail('试听音频无法播放，请检查模型和音色'), { once: true })
      previewRef.current = { audio, url }
      audio.src = url
      try {
        await audio.play()
      } catch (error) {
        fail(error instanceof DOMException && error.name === 'NotAllowedError' ? '浏览器阻止了试听播放，请再次点击试听' : '语音试听失败，请检查音色配置')
      }
    } catch (error) {
      if (mountedRef.current && generation === previewGeneration.current) notify(error instanceof Error ? error.message : '语音试听失败', 'error')
    } finally {
      if (mountedRef.current && generation === previewGeneration.current) setPreviewing(false)
    }
  }

  const clear = async () => {
    stopPreview()
    try { await onClear(); setApiKey(''); notify('语音 API Key 已删除', 'success') }
    catch (error) { notify(error instanceof Error ? error.message : '删除失败', 'error') }
  }

  return <form className="model-config-form" onSubmit={(event) => { event.preventDefault(); void submit() }}>
    <div className="model-config-grid">
      <div className="model-enabled-line">
        <div>
          <strong>启用伴侣语音回复</strong>
          <p>{enabled ? '伴侣可发送微信式语音条消息' : '当前未启用：只使用文字回复，不注入语音提示词'}</p>
        </div>
        <Switch checked={enabled} disabled={saving} onCheckedChange={setEnabled} aria-label="语音功能开关" />
      </div>
      {enabled ? <>
        <div className="field-group">
          <span>一键填入语音服务预设</span>
          <div className="voice-presets" role="group" aria-label="常用语音服务预设">
            {VOICE_SERVICE_PRESETS.map(preset => (
              <button
                key={preset.label}
                type="button"
                className={modelName.trim() === preset.modelName && baseUrl.trim().replace(/\/+$/, '') === preset.baseUrl ? 'active' : undefined}
                disabled={saving}
                onClick={() => {
                  stopPreview()
                  const keepVoice = sameVoiceEndpoint(baseUrl, preset.baseUrl)
                    && voiceService(baseUrl, modelName) === voiceService(preset.baseUrl, preset.modelName)
                    && (isMiniMax || modelName.trim() === preset.modelName)
                  if (!keepVoice || !voice.trim()) setVoice(preset.voice)
                  if (!sameVoiceEndpoint(baseUrl, preset.baseUrl)) setApiKey('')
                  setCustomBaseUrl(preset.baseUrl)
                  setModelName(preset.modelName)
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field-group"><span>发送策略</span><Select label="发送策略" value={triggerMode} disabled={saving} onValueChange={value => setTriggerMode(value as VoiceTriggerMode)} options={[{ value: 'occasional', label: '智能感应 + 偶尔主动发送（推荐）' }, { value: 'explicit_only', label: '仅在我想听声音时发送' }, { value: 'always', label: '尽量每次回复都发语音' }]} /></div>
        <label className="field-group"><span>接口地址 (Base URL)</span><input value={baseUrl} spellCheck={false} onChange={(event) => setCustomBaseUrl(event.target.value)} disabled={saving} placeholder="https://api.minimax.cn/v1" /></label>
        <label className="field-group"><span>语音模型名</span><input value={modelName} spellCheck={false} onChange={(event) => setModelName(event.target.value)} disabled={saving} placeholder={isMiniMax ? '如 speech-2.8-hd 或 speech-2.8-turbo' : '如 gpt-4o-mini-tts 或 FunAudioLLM/CosyVoice2-0.5B'} /></label>
        <ApiKeyField value={apiKey} onChange={setApiKey} configured={initial.configured && sameVoiceEndpoint(baseUrl, initial.baseUrl)} disabled={saving} placeholder={initial.configured && sameVoiceEndpoint(baseUrl, initial.baseUrl) ? '留空保留当前 Key' : sameAsMainUrl && mainModel.configured ? '留空自动复用聊天模型的 Key' : '输入 API Key'} />
        <VoicePicker
          baseUrl={baseUrl} model={modelName} apiKey={apiKey} value={voice}
          onChange={next => { stopPreview(); setVoice(next) }}
          savedVoice={initial.voice} savedBaseUrl={initial.baseUrl} disabled={saving}
        />
        <Button type="button" variant="secondary" className="voice-preview-btn" disabled={saving || previewing || !voice.trim() || !modelName.trim() || !baseUrl.trim()} onClick={() => void preview()}>{previewing ? <LoaderCircle className="spin" size={14} /> : <Volume2 size={14} />} 试听音色</Button>
        <p className="model-storage-note">独立保存在本机 voice-auth.json 中。与聊天模型地址相同时，留空可复用聊天模型 Key。</p>
      </> : (
        <div className="model-collapsed-hint">
          <p>如果你只需要纯文字陪伴，保持关闭即可。随时开启后可一键选择 MiniMax 或 OpenAI 兼容语音服务。</p>
        </div>
      )}
    </div>
    <ConfigFooter saving={saving} onClose={onClose}>{initial.configured && <Button type="button" variant="ghost" className="model-remove-key" disabled={saving} onClick={() => void clear()}><Trash2 size={14} />移除 Key</Button>}</ConfigFooter>
  </form>
}

function ModelPicker({ label, placeholder, value, onChange, loadModels, disabled }: { label: string; placeholder?: string; value: string; onChange: (value: string) => void; loadModels: () => Promise<string[]>; disabled: boolean }) {
  const id = useId()
  const [models, setModels] = useState<string[]>([])
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [active, setActive] = useState(0)
  const [above, setAbove] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const request = useRef({ generation: 0, active: true })
  const options = models.filter(model => model.toLowerCase().includes(filter.toLowerCase()))
  const placeMenu = () => {
    const rect = root.current?.querySelector('.model-picker-control')?.getBoundingClientRect()
    setAbove(Boolean(rect && rect.bottom + 250 > window.innerHeight && rect.top > 250))
  }
  const reveal = () => { placeMenu(); setOpen(true) }
  useEffect(() => {
    const state = request.current
    state.active = true
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', close)
    return () => { document.removeEventListener('pointerdown', close); state.active = false; state.generation++ }
  }, [])
  useEffect(() => {
    if (!open) return
    const menu = document.getElementById(`${id}-list`), option = document.getElementById(`${id}-option-${active}`)
    if (!menu || !option) return
    if (option.offsetTop < menu.scrollTop) menu.scrollTop = option.offsetTop
    else if (option.offsetTop + option.offsetHeight > menu.scrollTop + menu.clientHeight) menu.scrollTop = option.offsetTop + option.offsetHeight - menu.clientHeight
  }, [active, id, open])
  useEffect(() => {
    if (!open) return
    const reposition = () => {
      const rect = root.current?.querySelector('.model-picker-control')?.getBoundingClientRect()
      setAbove(Boolean(rect && rect.bottom + 250 > window.innerHeight && rect.top > 250))
    }
    window.addEventListener('resize', reposition)
    return () => window.removeEventListener('resize', reposition)
  }, [open])
  const fetchModels = async () => {
    const state = request.current, current = ++state.generation
    setLoading(true); setError(''); setOpen(false)
    try {
      const result = await loadModels()
      if (current !== state.generation || !state.active) return
      setModels(result); setFilter(''); setActive(0); reveal()
    } catch (caught) {
      if (current === state.generation && state.active) setError(caught instanceof Error ? caught.message : '获取模型列表失败')
    } finally { if (current === state.generation && state.active) setLoading(false) }
  }
  const select = (model: string) => { onChange(model); setFilter(''); setOpen(false) }
  return <div className="field-group model-picker" ref={root} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false) }}>
    <div className="model-picker-heading"><label htmlFor={id}>{label}</label><Button type="button" variant="icon" className="model-fetch" disabled={loading || disabled} title="获取模型列表" aria-label={`获取${label}列表`} onClick={() => void fetchModels()}>{loading ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}<span>{loading ? '获取中' : '获取列表'}</span></Button></div>
    <div className="model-picker-control">
      <input id={id} value={value} placeholder={placeholder} disabled={disabled} role="combobox" aria-expanded={open} aria-controls={`${id}-list`} aria-autocomplete="list" aria-activedescendant={open && options.length ? `${id}-option-${active}` : undefined} onChange={event => { onChange(event.target.value); setFilter(event.target.value); setActive(0); setOpen(models.length > 0) }} onKeyDown={event => {
        if (event.key === 'Escape' && open) { event.preventDefault(); event.stopPropagation(); setOpen(false); return }
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); if (!open) { setFilter(''); setActive(0); reveal() } else setActive(index => options.length ? (index + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length : 0) }
        if (event.key === 'Enter' && open) { event.preventDefault(); if (options[active]) select(options[active]) }
      }} />
      <button type="button" className="model-picker-toggle" title="展开模型列表" aria-label={`展开${label}列表`} disabled={disabled || !models.length} onClick={() => { setFilter(''); setActive(0); if (open) setOpen(false); else reveal() }}><ChevronDown size={15} /></button>
      {open && <div id={`${id}-list`} className={`provider-select-menu model-picker-menu${above ? ' above' : ''}`} role="listbox" aria-label={`${label}列表`}>{options.length ? options.map((model, index) => <button key={model} id={`${id}-option-${index}`} type="button" role="option" aria-selected={value === model} title={model} className={index === active ? 'active' : ''} onMouseEnter={() => setActive(index)} onClick={() => select(model)}><span>{model}</span>{model === value && <Check size={14} />}</button>) : <div className="model-picker-empty">没有匹配的模型</div>}</div>}
    </div>
      { error && <span className="form-error" role="alert">{error}</span> }
  </div>
}
