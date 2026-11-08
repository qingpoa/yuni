import { Check, Heart, Pencil, ShieldCheck, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { CompanionAvatar } from '../components/CompanionAvatar'
import { PageHeader } from '../components/PageHeader'
import { Button } from '../components/ui/button'
import { usePartnerStore } from '../stores/partnerStore'

const labels: Record<string, string> = {
  reserved: '克制', gentle_natural: '温柔自然', warm_close: '亲近温暖',
}

export function ProfilePage() {
  const { partner, loading, load } = usePartnerStore()
  const [now] = useState(Date.now)
  useEffect(() => { if (!partner) void load() }, [load, partner])

  if (loading || !partner) return <div className="inline-loading"><span /><span /><span /></div>
  const days = Math.max(1, Math.floor((now - new Date(partner.createdAt).getTime()) / 86_400_000))

  return (
    <div className="standard-page profile-page">
      <PageHeader eyebrow="伴侣资料" title={`你和${partner.name}`} description="这些设定决定 TA 如何称呼你、理解你，以及怎样陪在你身边。" actions={<Link to="/profile/edit" className="button-link"><Button variant="secondary"><Pencil size={15} /> 编辑资料</Button></Link>} />
      <section className="profile-intro">
        <div className="profile-portrait glass-surface"><CompanionAvatar className="companion-avatar-profile" label={partner.name} src={partner.avatarUrl} /><span className="profile-presence-label">在线</span></div>
        <div className="profile-identity">
          <div className="relationship-line"><span><Heart size={13} fill="currentColor" /> 稳定恋人</span><small>相识 {days} 天</small></div>
          <h2>{partner.name}</h2>
        </div>
      </section>
      <div className="profile-details">
        <section className="detail-section">
          <div className="detail-heading"><Sparkles size={18} /><div><h3>相处约定</h3></div></div>
          <dl className="detail-list">
            <div><dt>性别</dt><dd>{partner.gender === 'male' ? '男' : partner.gender === 'female' ? '女' : '未设置'}</dd></div>
            <div><dt>TA 对你的称呼</dt><dd>{partner.userNickname}</dd></div>
            <div><dt>亲密表达</dt><dd>{labels[partner.intimacyLevel] ?? partner.intimacyLevel}</dd></div>
          </dl>
        </section>
        <section className="detail-section">
          <div className="detail-heading"><ShieldCheck size={18} /><div><h3>关系边界</h3><p>{partner.name}会在每次聊天中尊重这项约定</p></div></div>
          <blockquote>{partner.boundaryText || '暂未设置额外边界。你可以随时编辑伴侣资料进行补充。'}</blockquote>
          <div className="boundary-confirm"><Check size={15} /><span>已应用到后续对话</span></div>
        </section>
      </div>
      <section className="profile-description detail-section"><h3>人物设定</h3><blockquote>{partner.customDescription || '尚未填写人物设定。'}</blockquote></section>
    </div>
  )
}
