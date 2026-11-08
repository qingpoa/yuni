import { LoaderCircle, Save } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PageHeader } from '../components/PageHeader'
import { AvatarCropDialog } from '../components/AvatarCropDialog'
import { PartnerBasicFields, PartnerBoundaryFields, PartnerCharacterFields } from '../components/PartnerFormFields'
import { Button } from '../components/ui/button'
import { usePartnerStore } from '../stores/partnerStore'
import { useUiStore } from '../stores/uiStore'
import type { CreatePartnerDraft } from '../types/domain'

export function EditPartnerPage() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const { draft, options, presets, saving, error, load, loadConfiguration, setDraft, usePartnerAsDraft: resetDraftFromPartner, updatePartner, uploadAvatar } = usePartnerStore()
  const notify = useUiStore((state) => state.notify)
  const [baseline, setBaseline] = useState<CreatePartnerDraft | null>(null)
  const [loadError, setLoadError] = useState('')
  const [validationError, setValidationError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [cropFile, setCropFile] = useState<File | null>(null)

  useEffect(() => {
    let active = true
    const prepare = async () => {
      setBaseline(null)
      setLoadError('')
      try {
        await loadConfiguration()
        await load()
        if (!active) return
        const current = usePartnerStore.getState()
        if (current.error || !current.partner) throw new Error(current.error ?? '伴侣资料加载失败')
        resetDraftFromPartner()
        setBaseline(usePartnerStore.getState().draft)
      } catch (cause) {
        if (active) setLoadError(cause instanceof Error ? cause.message : '伴侣资料加载失败')
      }
    }
    void prepare()
    return () => { active = false }
  }, [attempt, load, loadConfiguration, resetDraftFromPartner])

  const save = async () => {
    if (!baseline || saving) return
    if (!draft.name.trim()) return setValidationError('请填写伴侣名字')
    if (!draft.userNickname.trim()) return setValidationError('请填写伴侣对你的称呼')
    setValidationError('')
    setDraft({ name: draft.name.trim(), userNickname: draft.userNickname.trim(), boundaryText: draft.boundaryText.trim(), customDescription: draft.customDescription.trim() })
    try {
      await updatePartner()
      notify('伴侣资料已更新', 'success')
      navigate('/profile')
    } catch (cause) {
      notify(cause instanceof Error ? cause.message : '伴侣资料保存失败', 'error')
    }
  }
  const uploadCroppedAvatar = async (file: File) => {
    await uploadAvatar(file)
    notify('头像预览已更新', 'success')
  }
  const dirty = baseline !== null && JSON.stringify(draft) !== JSON.stringify(baseline)

  return <div className="standard-page partner-edit-page">
    <PageHeader eyebrow="伴侣资料" title="编辑伴侣" actions={<>
      <Button variant="secondary" disabled={saving} onClick={() => navigate('/profile')}>取消</Button>
      <Button className="partner-save" type="submit" form="partner-edit-form" disabled={!dirty || saving}>{saving ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}保存修改</Button>
    </>} />
    {loadError ? <div className="empty-state" role="alert"><strong>{loadError}</strong><Button variant="secondary" onClick={() => setAttempt((value) => value + 1)}>重新加载</Button></div> : !baseline || !options ? <div className="inline-loading" aria-label="正在加载伴侣资料"><span /><span /><span /></div> :
      <form id="partner-edit-form" className="partner-edit-form" onSubmit={(event) => { event.preventDefault(); void save() }}>
        <fieldset disabled={saving}>
          <section className="partner-edit-section" aria-labelledby="partner-basic-heading"><h2 id="partner-basic-heading">基础资料</h2><PartnerBasicFields draft={draft} presets={presets} activeColor={presets.find((item) => item.id === draft.avatarPresetId)?.color} setDraft={setDraft} openFile={() => inputRef.current?.click()} /></section>
          <section className="partner-edit-section" aria-labelledby="partner-character-heading"><h2 id="partner-character-heading">人物设定</h2><PartnerCharacterFields draft={draft} options={options} setDraft={setDraft} /></section>
          <section className="partner-edit-section" aria-labelledby="partner-boundary-heading"><h2 id="partner-boundary-heading">称呼与边界</h2><PartnerBoundaryFields draft={draft} setDraft={setDraft} /></section>
          <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" aria-label="伴侣头像文件" onChange={(event) => { const file = event.target.files?.[0]; event.currentTarget.value = ''; if (file) setCropFile(file) }} />
        </fieldset>
        {(validationError || error) && <p className="form-error" role="alert">{validationError || error}</p>}
      </form>}
    <AvatarCropDialog file={cropFile} onClose={() => setCropFile(null)} onConfirm={uploadCroppedAvatar} />
  </div>
}
