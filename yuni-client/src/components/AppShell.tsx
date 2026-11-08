import { AnimatePresence, motion } from 'framer-motion'
import { Heart, MessageCircle, Minus, NotebookTabs, Settings, Square, UserRound, UserRoundPen, X } from 'lucide-react'
import { useEffect } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { cn } from '../lib/utils'
import { windowAction } from '../services/desktop'
import { useAppStore } from '../stores/appStore'
import { usePartnerStore } from '../stores/partnerStore'
import { useSettingsStore } from '../stores/settingsStore'
import { useProfileStore } from '../stores/profileStore'
import { CompanionAvatar } from './CompanionAvatar'
import { ToolConfirmationDialog } from './ToolConfirmationDialog'
import { Tooltip } from './ui/tooltip'

const companionNav = [
  { to: '/chat', label: '聊天', icon: MessageCircle },
  { to: '/profile', label: '伴侣资料', icon: UserRound },
  { to: '/memory', label: '记忆中心', icon: NotebookTabs },
]

export function WindowChrome({ compact = false }: { compact?: boolean }) {
  return <header className={cn('window-chrome', compact && 'window-chrome-overlay')} data-tauri-drag-region>
    <div className="window-brand" data-tauri-drag-region><span className="brand-glyph"><Heart size={13} fill="currentColor" /></span><span>Yuni</span></div>
    <div className="window-actions" aria-label="窗口控制">
      <Tooltip label="最小化"><button aria-label="最小化" onClick={() => void windowAction('minimize')}><Minus size={14} /></button></Tooltip>
      <Tooltip label="最大化"><button aria-label="最大化" onClick={() => void windowAction('toggleMaximize')}><Square size={11} /></button></Tooltip>
      <Tooltip label="关闭"><button className="close-window" aria-label="关闭" onClick={() => void windowAction('close')}><X size={14} /></button></Tooltip>
    </div>
  </header>
}

export function AppShell() {
  const location = useLocation()
  const bootstrap = useAppStore((state) => state.bootstrapData)
  const { partner, load } = usePartnerStore()
  const loadSettings = useSettingsStore((state) => state.load)
  const loadProfile = useProfileStore((state) => state.load)

  useEffect(() => { if (bootstrap?.hasCompanion) void load(); void loadSettings() }, [bootstrap?.hasCompanion, load, loadSettings])
  useEffect(() => { void loadProfile() }, [loadProfile])
  const companionName = partner?.name ?? '尚未创建'

  return <div className="desktop-canvas"><div className="desktop-app"><WindowChrome /><div className="app-body">
    <aside className="sidebar glass-surface">
      <div className="companion-summary"><div className="avatar-shell avatar-shell-lg"><CompanionAvatar label={companionName} src={partner?.avatarUrl} /></div><div><strong>{companionName}</strong></div></div>
      <nav className="sidebar-nav" aria-label="主导航">
        {bootstrap?.hasCompanion && companionNav.map(({ to, label, icon: Icon }) => <NavLink key={to} to={to} className={({ isActive }) => cn('nav-item', isActive && 'active')}><Icon size={18} strokeWidth={1.8} /><span>{label}</span></NavLink>)}
        {!bootstrap?.hasCompanion && <NavLink to="/partner/create" className="nav-item"><Heart size={18} strokeWidth={1.8} /><span>创建伴侣</span></NavLink>}
        <NavLink to="/me" className={({ isActive }) => cn('nav-item', isActive && 'active')}><UserRoundPen size={18} strokeWidth={1.8} /><span>我的</span></NavLink>
        <NavLink to="/settings" className={({ isActive }) => cn('nav-item', isActive && 'active')}><Settings size={18} strokeWidth={1.8} /><span>设置</span></NavLink>
      </nav>
    </aside>
    <main className="workspace"><AnimatePresence mode="wait"><motion.div key={location.pathname} className="page-motion" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}><Outlet /></motion.div></AnimatePresence></main>
  </div></div><ToolConfirmationDialog /></div>
}
