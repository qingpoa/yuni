import { Check, ImagePlus } from 'lucide-react'
import { CompanionAvatar } from './CompanionAvatar'
import { cn } from '../lib/utils'
import type { AvatarPreset, CreatePartnerDraft, PartnerOptions } from '../types/domain'

type DraftFields = { draft: CreatePartnerDraft; setDraft: (patch: Partial<CreatePartnerDraft>) => void }

export function PartnerBasicFields({ draft, presets, activeColor, setDraft, openFile }: DraftFields & {
  presets: AvatarPreset[]; activeColor?: string; openFile: () => void
}) {
  return <>
    <div className="avatar-config">
      <div className="avatar-preview-large"><CompanionAvatar label="当前伴侣" color={activeColor} src={draft.avatarUrl} /><span className="avatar-preview-label">预览</span></div>
      <div className="avatar-options"><span className="field-label">选择默认头像</span><div className="avatar-choice-row">
        {presets.map((preset) => <button type="button" key={preset.id} className={cn('avatar-choice', draft.avatarPresetId === preset.id && 'active')} onClick={() => setDraft({ avatarType: 'preset', avatarPresetId: preset.id, avatarFileId: null, avatarUrl: preset.url })} aria-label={preset.name} aria-pressed={draft.avatarPresetId === preset.id}>
          <CompanionAvatar label={preset.name} color={preset.color} src={preset.url} />{draft.avatarPresetId === preset.id && <i><Check size={11} /></i>}
        </button>)}
        <button type="button" className={cn('avatar-choice upload-choice', draft.avatarType === 'uploaded' && 'active')} onClick={openFile} aria-label="上传头像"><ImagePlus size={19} /></button>
      </div></div>
    </div>
    <div className="form-grid"><label className="field-group"><span>伴侣名字</span><input maxLength={20} value={draft.name} onChange={(event) => setDraft({ name: event.target.value })} placeholder="给 TA 一个名字" /></label></div>
    <div className="field-group partner-gender-field"><span id="partner-gender-label">伴侣性别</span><div className="gender-options" role="radiogroup" aria-labelledby="partner-gender-label">
      {(['male', 'female'] as const).map((gender) => <label key={gender} className={cn('gender-option', draft.gender === gender && 'active')}>
        <input type="radio" name="partner-gender" value={gender} checked={draft.gender === gender} onChange={() => setDraft({ gender })} />
        <span>{gender === 'male' ? '男' : '女'}</span>
      </label>)}
    </div></div>
  </>
}

export function PartnerCharacterFields({ draft, options, setDraft }: DraftFields & {
  options: PartnerOptions;
}) {
  return <>
    <label className="field-group"><span>自定义人物描述</span><textarea aria-label="自定义人物描述" aria-describedby="partner-description-count" rows={7} maxLength={2000} value={draft.customDescription} onChange={(event) => setDraft({ customDescription: event.target.value })} placeholder="例如：嘴硬心软，喜欢冷幽默；平时话不多，聊到电影时会很健谈。" /><small id="partner-description-count">{draft.customDescription.length} / 2000</small></label>
    <div className="choice-section"><span className="field-label">亲密表达</span><div className="intimacy-options">
      {options.intimacyLevels.map((item) => <label key={item.value} className={cn('intimacy-option', draft.intimacyLevel === item.value && 'active')}>
        <input type="radio" name="intimacy" checked={draft.intimacyLevel === item.value} onChange={() => setDraft({ intimacyLevel: item.value })} />
        <span><strong>{item.label}</strong><small>{item.value === 'reserved' ? '给彼此更多空间' : item.value === 'gentle_natural' ? '亲近但不过分黏腻' : '更主动表达在意'}</small></span>
      </label>)}
    </div></div>
  </>
}

export function PartnerBoundaryFields({ draft, setDraft }: DraftFields) {
  return <div className="partner-boundary-fields">
    <label className="field-group"><span>TA 对你的称呼</span><input maxLength={20} value={draft.userNickname} onChange={(event) => setDraft({ userNickname: event.target.value })} placeholder="例如：小予" /></label>
    <label className="field-group"><span>边界说明</span><textarea rows={5} maxLength={500} value={draft.boundaryText} onChange={(event) => setDraft({ boundaryText: event.target.value })} placeholder="例如：难过时先陪我一会儿，再问我是否需要建议。" /><small>{draft.boundaryText.length} / 500</small></label>
  </div>
}

