import { lazy, Suspense, useEffect, type ReactNode } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { ToastViewport } from './components/ToastViewport'
import { appApi } from './services/api'
import { useAppStore } from './stores/appStore'
import { useChatStore } from './stores/chatStore'
import { usePartnerStore } from './stores/partnerStore'
import { useUiStore } from './stores/uiStore'

const ChatPage = lazy(() => import('./pages/ChatPage').then((module) => ({ default: module.ChatPage })))
const CreatePartnerPage = lazy(() => import('./pages/CreatePartnerPage').then((module) => ({ default: module.CreatePartnerPage })))
const EditPartnerPage = lazy(() => import('./pages/EditPartnerPage').then((module) => ({ default: module.EditPartnerPage })))
const MemoryPage = lazy(() => import('./pages/MemoryPage').then((module) => ({ default: module.MemoryPage })))
const MyPage = lazy(() => import('./pages/MyPage').then((module) => ({ default: module.MyPage })))
const ProfilePage = lazy(() => import('./pages/ProfilePage').then((module) => ({ default: module.ProfilePage })))
const SettingsPage = lazy(() => import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })))

function LoadingScreen() {
  return <div className="route-loading" aria-label="正在连接本地服务"><span /><span /><span /><p>正在启动本地 Agent</p></div>
}

function RouteGate({ children, companion = 'any' }: { children: ReactNode; companion?: 'any' | 'required' | 'missing' }) {
  const { status, bootstrapData } = useAppStore()
  if (['booting_tauri', 'starting_agent', 'loading_bootstrap'].includes(status)) return <LoadingScreen />
  if (status === 'failed' || status === 'degraded') return <StartupError />
  if (companion === 'required' && !bootstrapData?.hasCompanion) return <Navigate to="/partner/create" replace />
  if (companion === 'missing' && bootstrapData?.hasCompanion) return <Navigate to="/chat" replace />
  return children
}

function StartupError() {
  const { bootstrap, error } = useAppStore()
  return <div className="startup-error glass-surface"><h1>本地服务没有准备好</h1><p>{error ?? 'Yuni 无法连接本地 Agent。你可以重新启动，或打开日志目录查看详情。'}</p><div className="startup-actions"><button onClick={() => void bootstrap()}>重新启动</button><button className="secondary" onClick={() => void appApi.system.openLogs()}>打开日志目录</button></div></div>
}

export default function App() {
  const { bootstrap, status, bootstrapData } = useAppStore()
  useEffect(() => { void bootstrap() }, [bootstrap])
  const notify = useUiStore((state) => state.notify)
  useEffect(() => {
    if (status !== 'ready') return
    useChatStore.getState().startEvents(notification => {
      const path = window.location.hash.startsWith('#/') ? window.location.hash.slice(1).split('?')[0] : window.location.pathname
      if (notification.kind === 'message' && path === '/chat' && document.visibilityState === 'visible') return
      const title = notification.kind === 'reminder' ? notification.reminder.title : usePartnerStore.getState().partner?.name ?? '伴侣'
      const body = notification.kind === 'reminder' ? notification.reminder.note || '约定的时间到了。' : notification.message.content
      if ('Notification' in window && Notification.permission === 'granted') {
        try { new Notification(title, { body }); return } catch { /* Fall back to an in-app notification. */ }
      }
      notify(`${title}：${body}`, 'info')
    })
    return () => useChatStore.getState().stopEvents()
  }, [notify, status])
  const defaultRoute = bootstrapData?.hasCompanion ? '/chat' : '/partner/create'

  return <><Suspense fallback={<LoadingScreen />}><Routes>
    <Route path="/partner/create" element={<RouteGate companion="missing"><CreatePartnerPage /></RouteGate>} />
    <Route element={<RouteGate><AppShell /></RouteGate>}>
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="/me" element={<MyPage />} />
      <Route path="/chat" element={<RouteGate companion="required"><ChatPage /></RouteGate>} />
      <Route path="/profile" element={<RouteGate companion="required"><ProfilePage /></RouteGate>} />
      <Route path="/profile/edit" element={<RouteGate companion="required"><EditPartnerPage /></RouteGate>} />
      <Route path="/memory" element={<RouteGate companion="required"><MemoryPage /></RouteGate>} />
    </Route>
    <Route path="*" element={['booting_tauri', 'starting_agent', 'loading_bootstrap'].includes(status) ? <LoadingScreen /> : <Navigate to={defaultRoute} replace />} />
  </Routes></Suspense><ToastViewport /></>
}
