export async function windowAction(action: 'minimize' | 'toggleMaximize' | 'close') {
  if (!('__TAURI_INTERNALS__' in window)) return
  const { getCurrentWindow } = await import('@tauri-apps/api/window')
  await getCurrentWindow()[action]()
}

export async function openExternal(url: string) {
  if ('__TAURI_INTERNALS__' in window) {
    const { openUrl } = await import('@tauri-apps/plugin-opener')
    await openUrl(url)
    return
  }
  window.open(url, '_blank', 'noopener,noreferrer')
}

export const isTauriRuntime = () => '__TAURI_INTERNALS__' in window

export async function pickDirectory() {
  if (!isTauriRuntime()) return null
  const { open } = await import('@tauri-apps/plugin-dialog')
  const selected = await open({ directory: true, multiple: false, title: '选择 Yuni 可访问的目录' })
  return typeof selected === 'string' ? selected : null
}
