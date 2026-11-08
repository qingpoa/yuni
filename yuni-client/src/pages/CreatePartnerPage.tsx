import { motion } from 'framer-motion'
import { ArrowLeft, ArrowRight, Check, Heart, ImagePlus, LoaderCircle, ShieldCheck, Sparkles } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { WindowChrome } from '../components/AppShell'
import { AvatarCropDialog } from '../components/AvatarCropDialog'
import { PartnerBasicFields, PartnerBoundaryFields, PartnerCharacterFields } from '../components/PartnerFormFields'
import { Button } from '../components/ui/button'
import { cn } from '../lib/utils'
import { useAppStore } from '../stores/appStore'
import { usePartnerStore } from '../stores/partnerStore'
import { useUiStore } from '../stores/uiStore'

const steps = [
  { label: '基础资料', icon: ImagePlus },
  { label: '人物设定', icon: Sparkles },
  { label: '边界确认', icon: ShieldCheck },
]

export function CreatePartnerPage() {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [step, setStep] = useState(0)
  const [confirmed, setConfirmed] = useState(false)
  const [validationError, setValidationError] = useState('')
  const [cropFile, setCropFile] = useState<File | null>(null)
  const setHasCompanion = useAppStore((state) => state.setHasCompanion)
  const { draft, options, presets, saving, error, loadConfiguration, setDraft, createPartner, uploadAvatar } = usePartnerStore()
  const notify = useUiStore((state) => state.notify)

  useEffect(() => { void loadConfiguration() }, [loadConfiguration])

  const validateStep = () => {
    if (step === 0 && !draft.name.trim()) return '请填写伴侣名字'
    if (step === 0 && draft.name.trim().length > 20) return '伴侣名字不能超过 20 个字符'
    if (step === 0 && !draft.gender) return '请选择伴侣性别'
    if (step === 2 && !draft.userNickname.trim()) return '请填写伴侣对你的称呼'
    if (step === 2 && draft.userNickname.trim().length > 20) return '称呼不能超过 20 个字符'
    if (step === 2 && !confirmed) return '请先确认安全陪伴约定'
    return ''
  }

  const next = () => {
    const message = validateStep()
    if (message) return setValidationError(message)
    setValidationError('')
    setStep(Math.min(2, step + 1))
  }

  const save = async () => {
    if (!draft.gender) { setStep(0); return setValidationError('请选择伴侣性别') }
    const message = validateStep()
    if (message) return setValidationError(message)
    setDraft({ name: draft.name.trim(), userNickname: draft.userNickname.trim(), boundaryText: draft.boundaryText.trim() })
    try {
      await createPartner()
      setHasCompanion(true)
      notify('伴侣创建完成', 'success')
      navigate('/chat')
    } catch {
      // Store keeps the draft and exposes the error in the page.
    }
  }

  const uploadCroppedAvatar = async (file: File) => {
    await uploadAvatar(file)
    notify('头像预览已更新', 'success')
  }

  const activePreset = presets.find((item) => item.id === draft.avatarPresetId)

  return (
    <div className="onboarding-canvas">
      <div className="onboarding-window">
        <WindowChrome />
        <div className="onboarding-layout">
          <aside className="onboarding-aside">
            <div>
              <span className="brand-glyph brand-glyph-large"><Heart size={18} fill="currentColor" /></span>
              <span className="eyebrow">创建伴侣</span>
              <h1>让相处从了解开始</h1>
              <p>这些设定可以随时修改。第一版只创建一位伴侣，让关系保持简单而连续。</p>
            </div>
            <div className="step-list">
              {steps.map(({ label, icon: Icon }, index) => (
                <button key={label} className={cn('step-item', step === index && 'active', step > index && 'done')} onClick={() => { setStep(index); setValidationError('') }}>
                  <span>{step > index ? <Check size={16} /> : <Icon size={16} />}</span>
                  <div><strong>0{index + 1}</strong><p>{label}</p></div>
                </button>
              ))}
            </div>
            <Link to="/settings" className="back-link"><ArrowLeft size={15} /> 暂时离开</Link>
          </aside>

          <main className="onboarding-main">
            <div className="step-progress"><span style={{ width: `${((step + 1) / steps.length) * 100}%` }} /></div>
            <motion.div key={step} className="onboarding-step" initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.18 }}>
              {step === 0 && <><div className="section-heading"><span>第一步</span><h2>TA 是怎样的人？</h2><p>先从名字和一个适合 TA 的头像开始。</p></div><PartnerBasicFields draft={draft} presets={presets} activeColor={activePreset?.color} setDraft={(patch) => { setDraft(patch); setValidationError('') }} openFile={() => inputRef.current?.click()} /></>}
              {step === 1 && options && (
                <>
                  <div className="section-heading"><span>第二步</span><h2>TA 是怎样的性格？</h2></div>
                  <PartnerCharacterFields draft={draft} options={options} setDraft={setDraft} />
                </>
              )}
              {step === 2 && (
                <>
                  <div className="section-heading"><span>第三步</span><h2>把称呼和边界说清楚</h2><p>告诉伴侣如何称呼你，以及什么样的陪伴更舒服。</p></div>
                  <PartnerBoundaryFields draft={draft} setDraft={setDraft} />
                  <div className="safety-note"><ShieldCheck size={20} /><div><strong>安全陪伴约定</strong><p>Yuni 不会提供危险建议，也不会鼓励你远离现实关系。遇到高风险表达时，会认真引导你寻求现实帮助。</p></div></div>
                  <label className="confirm-line"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>我已了解以上说明，并同意以安全、尊重的方式开始这段关系。</span></label>
                </>
              )}
            </motion.div>

            {(validationError || error) && <p className="form-error wizard-error" role="alert">{validationError || error}</p>}
            <footer className="onboarding-actions">
              <Button variant="ghost" disabled={step === 0 || saving} onClick={() => { setStep(Math.max(0, step - 1)); setValidationError('') }}><ArrowLeft size={16} /> 上一步</Button>
              {step < 2 ? <Button disabled={saving} onClick={next}>下一步 <ArrowRight size={16} /></Button> : <Button disabled={saving} onClick={() => void save()}>{saving ? <><LoaderCircle className="spin" size={16} /> 保存中</> : <>创建并开始聊天 <Heart size={16} fill="currentColor" /></>}</Button>}
            </footer>
            <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" aria-label="伴侣头像文件" onChange={(event) => { const file = event.target.files?.[0]; event.currentTarget.value = ''; if (file) setCropFile(file) }} />
          </main>
        </div>
      </div>
      <AvatarCropDialog file={cropFile} onClose={() => setCropFile(null)} onConfirm={uploadCroppedAvatar} />
    </div>
  )
}

