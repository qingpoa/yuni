import { create } from 'zustand'

export interface ToastItem {
  id: number
  message: string
  tone: 'success' | 'error' | 'info'
}

interface UiState {
  toasts: ToastItem[]
  notify: (message: string, tone?: ToastItem['tone']) => void
  dismiss: (id: number) => void
}

export const useUiStore = create<UiState>((set, get) => ({
  toasts: [],
  notify: (message, tone = 'info') => {
    const id = Date.now()
    set({ toasts: [...get().toasts, { id, message, tone }] })
    window.setTimeout(() => get().dismiss(id), 3200)
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((toast) => toast.id !== id) }),
}))
