import { Check, ImagePlus, LoaderCircle, Save } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { CompanionAvatar } from '../components/CompanionAvatar'
import { AvatarCropDialog } from '../components/AvatarCropDialog'
import { PageHeader } from '../components/PageHeader'
import { Button } from '../components/ui/button'
import { cn } from '../lib/utils'
import { appApi } from '../services/api'
import { useProfileStore } from '../stores/profileStore'
import { useUiStore } from '../stores/uiStore'
import type { AvatarPreset, UserProfile, UserProfileDraft } from '../types/domain'

function toDraft(profile: UserProfile): UserProfileDraft {
  const { nickname, avatarType, avatarPresetId, avatarFileId, avatarUrl } = profile
  return { nickname, avatarType, avatarPresetId, avatarFileId, avatarUrl }
}

export function MyPage() {
  const { load, save } = useProfileStore()
  const notify = useUiStore((state) => state.notify)
  const [draft, setDraft] = useState<UserProfileDraft | null>(null)
  const [baseline, setBaseline] = useState<UserProfileDraft | null>(null)
  const [presets, setPresets] = useState<AvatarPreset[]>([])
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [cropFile, setCropFile] = useState<File | null>(null)
  const [attempt, setAttempt] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    let active = true
    const prepare = async () => {
      setDraft(null)
      setError('')
      try {
        const [, avatars] = await Promise.all([load(), appApi.partner.getPresets()])
        const { profile, error: loadError } = useProfileStore.getState()
        if (!profile || loadError) throw new Error(loadError ?? '个人资料加载失败')
        if (active) { const next = toDraft(profile); setDraft(next); setBaseline(next); setPresets(avatars) }
      } catch (cause) { if (active) setError(cause instanceof Error ? cause.message : '个人资料加载失败') }
    }
    void prepare()
    return () => { active = false }
  }, [attempt, load])
  const update = (patch: Partial<UserProfileDraft>) => { setDraft((current) => current ? { ...current, ...patch } : current); setError('') }
  const dirty = draft !== null && JSON.stringify(draft) !== JSON.stringify(baseline)
  const submit = async () => {
    if (!draft || saving || uploading) return
    if (!draft.nickname.trim()) return setError('请填写昵称')
    setSaving(true)
    setError('')
    try {
      const profile = await save({ ...draft, nickname: draft.nickname.trim() })
      const next = toDraft(profile)
      setDraft(next); setBaseline(next)
      notify('个人资料已更新', 'success')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '个人资料保存失败') }
    finally { setSaving(false) }
  }
  const uploadAvatar = async (file: File) => {
    setUploading(true); setError('')
    try { const avatar = await appApi.profile.uploadAvatar(file); update({ ...avatar, avatarType: 'uploaded', avatarPresetId: null }) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '头像上传失败'); throw cause }
    finally { setUploading(false) }
  }
  return <div className="standard-page partner-edit-page my-page">
    <PageHeader eyebrow="个人资料" title="我的" actions={<>
      <Button variant="secondary" disabled={!dirty || saving || uploading} onClick={() => { setDraft(baseline); setError('') }}>取消修改</Button>
      <Button type="submit" form="my-profile-form" disabled={!draft || !dirty || saving || uploading}>{saving || uploading ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}保存修改</Button>
    </>} />
    {!draft ? error ? <div className="empty-state" role="alert"><strong>{error}</strong><Button variant="secondary" onClick={() => setAttempt((value) => value + 1)}>重新加载</Button></div> : <div className="inline-loading" aria-label="正在加载个人资料"><span /><span /><span /></div> :
      <form id="my-profile-form" className="partner-edit-form my-profile-form" onSubmit={(event) => { event.preventDefault(); void submit() }}>
        <fieldset disabled={saving || uploading}>
          <section className="partner-edit-section">
            <h2>个人资料</h2>
            <div className="avatar-config">
              <div className="avatar-preview-large"><CompanionAvatar label={draft.nickname || '我'} src={draft.avatarUrl} /><span className="avatar-preview-label">预览</span></div>
              <div className="avatar-options"><span className="field-label">头像</span><div className="avatar-choice-row">
                {presets.map((preset) => <button type="button" key={preset.id} className={cn('avatar-choice', draft.avatarPresetId === preset.id && 'active')} aria-label={preset.name} aria-pressed={draft.avatarPresetId === preset.id} onClick={() => update({ avatarType: 'preset', avatarPresetId: preset.id, avatarFileId: null, avatarUrl: preset.url })}><CompanionAvatar label={preset.name} src={preset.url} color={preset.color} />{draft.avatarPresetId === preset.id && <i><Check size={11} /></i>}</button>)}
                <button type="button" className={cn('avatar-choice upload-choice', draft.avatarType === 'uploaded' && 'active')} aria-label="上传我的头像" title="上传我的头像" onClick={() => inputRef.current?.click()}><ImagePlus size={19} /></button>
              </div></div>
            </div>
            <label className="field-group"><span>昵称</span><input autoComplete="nickname" maxLength={20} value={draft.nickname} onChange={(event) => update({ nickname: event.target.value })} /></label>
            <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" aria-label="我的头像文件" onChange={(event) => { const file = event.target.files?.[0]; event.currentTarget.value = ''; if (file) setCropFile(file) }} />
          </section>
        </fieldset>
        {error && <p className="form-error" role="alert">{error}</p>}
      </form>}
    <AvatarCropDialog file={cropFile} onClose={() => setCropFile(null)} onConfirm={uploadAvatar} />
  </div>
}
