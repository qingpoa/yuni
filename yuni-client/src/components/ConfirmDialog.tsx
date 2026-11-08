import { Button } from './ui/button'
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel = '确认',
  busy = false,
  onConfirm,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  confirmLabel?: string
  busy?: boolean
  onConfirm: () => void | Promise<void>
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="confirm-dialog">
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
        <div className="dialog-actions">
          <DialogClose asChild><Button variant="ghost" disabled={busy}>取消</Button></DialogClose>
          <Button variant="danger" disabled={busy} onClick={onConfirm}>{busy ? '处理中...' : confirmLabel}</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
