import { Check, LoaderCircle, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useState } from 'react'
import Cropper, { type Area } from 'react-easy-crop'
import { cropAvatar, loadAvatarImage, validateAvatarFile } from '../lib/avatarCrop'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'
import { Tooltip } from './ui/tooltip'

export function AvatarCropDialog({ file, onClose, onConfirm }: {
  file: File | null; onClose: () => void; onConfirm: (file: File) => Promise<void>
}) {
  return file ? <AvatarCropSession key={`${file.name}:${file.size}:${file.lastModified}`} file={file} onClose={onClose} onConfirm={onConfirm} /> : null
}

function AvatarCropSession({ file, onClose, onConfirm }: {
  file: File; onClose: () => void; onConfirm: (file: File) => Promise<void>
}) {
  const [source, setSource] = useState('')
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [crop, setCrop] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [area, setArea] = useState<Area | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    let url = ''
    void Promise.resolve().then(async () => {
      if (!active) return
      validateAvatarFile(file)
      url = URL.createObjectURL(file)
      const loaded = await loadAvatarImage(url)
      if (active) { setImage(loaded); setSource(url) }
    }).catch((cause) => { if (active) setError(cause instanceof Error ? cause.message : '图片读取失败') })
    return () => { active = false; if (url) URL.revokeObjectURL(url) }
  }, [file])
  const changeZoom = (next: number) => setZoom(Math.min(3, Math.max(1, next)))
  const confirm = async () => {
    if (!image || !area || busy) return
    setBusy(true); setError('')
    try { const cropped = await cropAvatar(image, area, file.name); await onConfirm(cropped); onClose() }
    catch (cause) { setError(cause instanceof Error ? cause.message : '头像上传失败，请重试') }
    finally { setBusy(false) }
  }
  return <Dialog open onOpenChange={(open) => { if (!open && !busy) onClose() }}>
    <DialogContent className="avatar-crop-dialog" closeDisabled={busy}>
      <DialogTitle>裁剪头像</DialogTitle>
      <DialogDescription className="visually-hidden">头像裁剪</DialogDescription>
      <div className="avatar-crop-viewport" aria-label="头像裁剪区域">
        {source ? <Cropper image={source} crop={crop} zoom={zoom} minZoom={1} maxZoom={3} aspect={1} cropShape="round" showGrid={false} onCropChange={setCrop} onZoomChange={changeZoom} onCropComplete={(_, pixels) => setArea(pixels)} keyboardStep={10} style={{ containerStyle: { pointerEvents: busy ? 'none' : 'auto' } }} /> : !error && <LoaderCircle size={24} className="spin" aria-label="正在读取图片" />}
      </div>
      <div className="avatar-crop-controls">
        <Tooltip label="缩小"><Button variant="icon" aria-label="缩小" disabled={!source || busy || zoom <= 1} onClick={() => changeZoom(zoom - 0.1)}><ZoomOut size={18} /></Button></Tooltip>
        <input type="range" aria-label="缩放" min={1} max={3} step={0.01} value={zoom} disabled={!source || busy} onChange={(event) => changeZoom(Number(event.target.value))} />
        <Tooltip label="放大"><Button variant="icon" aria-label="放大" disabled={!source || busy || zoom >= 3} onClick={() => changeZoom(zoom + 0.1)}><ZoomIn size={18} /></Button></Tooltip>
        <Tooltip label="重置裁剪"><Button variant="icon" aria-label="重置裁剪" disabled={!source || busy} onClick={() => { setCrop({ x: 0, y: 0 }); setZoom(1) }}><RotateCcw size={17} /></Button></Tooltip>
      </div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="dialog-actions"><Button variant="secondary" disabled={busy} onClick={onClose}>取消</Button><Button disabled={!image || !area || busy} onClick={() => void confirm()}>{busy ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}裁剪并上传</Button></div>
    </DialogContent>
  </Dialog>
}
