import { LoaderCircle, RefreshCw } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { appApi } from '../services/api'
import { Button } from './ui/button'
import { Select } from './ui/select'
import { builtinVoiceChoices, sameVoiceEndpoint, voiceService, type VoiceChoice } from './voiceOptions'

const CUSTOM_VOICE = '__yuni_custom_voice__'

interface VoicePickerProps {
  baseUrl: string
  model: string
  apiKey: string
  value: string
  onChange: (voice: string) => void
  savedVoice: string
  savedBaseUrl: string
  disabled: boolean
}

export function VoicePicker(props: VoicePickerProps) {
  return <VoicePickerContent key={`${props.baseUrl}\n${props.model}\n${props.apiKey}`} {...props} />
}

function VoicePickerContent({ baseUrl, model, apiKey, value, onChange, savedVoice, savedBaseUrl, disabled }: VoicePickerProps) {
  const inputId = useId()
  const [accountVoices, setAccountVoices] = useState<VoiceChoice[]>([])
  const [custom, setCustom] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [loaded, setLoaded] = useState(false)
  const generation = useRef(0)
  const service = voiceService(baseUrl, model)

  useEffect(() => () => { generation.current += 1 }, [])

  const choices = new Map(builtinVoiceChoices(baseUrl, model).map(voice => [voice.id, voice]))
  for (const voice of accountVoices) {
    // The provider sometimes returns only IDs; retain our friendly built-in names.
    choices.set(voice.id, { ...voice, name: voice.name === voice.id ? choices.get(voice.id)?.name ?? voice.name : voice.name })
  }
  const counts = { system: 0, cloned: 0, generated: 0 }
  const options = [...choices.values()].map(voice => {
    counts[voice.kind] += 1
    const category = voice.kind === 'cloned' ? '克隆音色' : voice.kind === 'generated' ? '设计音色' : '系统音色'
    return {
      value: voice.id,
      label: voice.name === voice.id
        ? voice.id === savedVoice ? '当前已保存的音色' : `${category} ${counts[voice.kind]}`
        : voice.name,
      description: voice.name === voice.id ? voice.id : category,
    }
  })
  if (savedVoice && sameVoiceEndpoint(baseUrl, savedBaseUrl) && !choices.has(savedVoice)) {
    options.unshift({ value: savedVoice, label: '当前已保存的音色', description: '保留已有配置，可直接试听' })
  }
  const isListed = options.some(option => option.value === value)
  const showCustom = custom || !isListed
  const selected = showCustom ? CUSTOM_VOICE : value

  const fetchVoices = async () => {
    const current = ++generation.current
    setLoading(true)
    setError('')
    try {
      const voices = await appApi.settings.listVoices({ baseUrl: baseUrl.trim(), apiKey: apiKey.trim() || undefined })
      if (current !== generation.current) return
      setAccountVoices(voices)
      setLoaded(true)
    } catch (error) {
      if (current === generation.current) setError(error instanceof Error ? error.message : '音色列表获取失败，请稍后重试')
    } finally {
      if (current === generation.current) setLoading(false)
    }
  }

  return <div className="field-group voice-picker">
    <div className="voice-picker-heading">
      <span>选择音色</span>
      {service === 'minimax' && <Button type="button" variant="ghost" className="voice-catalog-fetch" disabled={disabled || loading} onClick={() => void fetchVoices()}>
        {loading ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}
        {loading ? '获取中…' : loaded ? '刷新账号音色' : '获取账号音色'}
      </Button>}
    </div>
    <Select label="选择音色" value={selected} disabled={disabled} onValueChange={choice => {
      if (choice === CUSTOM_VOICE) {
        setCustom(true)
        if (isListed && value !== savedVoice) onChange('')
      } else {
        setCustom(false)
        onChange(choice)
      }
    }} options={[...options, { value: CUSTOM_VOICE, label: '自定义 / 克隆音色…', description: '已有音色 ID 时使用' }]} />
    {showCustom && <label className="voice-custom-field" htmlFor={inputId}>
      <span>自定义音色 ID</span>
      <input id={inputId} value={value} maxLength={120} spellCheck={false} disabled={disabled} onChange={event => onChange(event.target.value)} placeholder="粘贴服务商提供的音色 ID" />
    </label>}
    <p className="voice-host-hint">{showCustom
      ? '填写语音服务商提供的音色 ID；克隆音色需由你自己的账号创建。'
      : '不用记音色 ID，直接选择后点击「试听音色」，听过再决定。'}</p>
    {service === 'minimax' && <p className="voice-host-hint">填写 API Key 后，可获取该账号的完整音色列表，包含系统音色和已有克隆音色。</p>}
    {service === 'custom' && <p className="voice-host-hint">此模型尚无内置音色列表，请使用服务商提供的音色 ID。</p>}
    {loaded && <p className="voice-catalog-status" role="status">已获取 {accountVoices.length} 个账号音色{accountVoices.length ? '' : '，可继续使用内置音色或自定义音色'}。</p>}
    {error && <span className="form-error" role="alert">{error}。仍可选择内置音色。</span>}
  </div>
}
