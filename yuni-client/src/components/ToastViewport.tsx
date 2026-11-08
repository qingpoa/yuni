import { CheckCircle2, CircleAlert, Info, X } from 'lucide-react'
import { useUiStore } from '../stores/uiStore'

export function ToastViewport() {
  const { toasts, dismiss } = useUiStore()
  return (
    <div className="toast-viewport" aria-live="polite">
      {toasts.map((toast) => {
        const Icon = toast.tone === 'success' ? CheckCircle2 : toast.tone === 'error' ? CircleAlert : Info
        return (
          <div key={toast.id} className={`toast toast-${toast.tone}`}>
            <Icon size={16} />
            <span>{toast.message}</span>
            <button onClick={() => dismiss(toast.id)} aria-label="关闭提示"><X size={14} /></button>
          </div>
        )
      })}
    </div>
  )
}
