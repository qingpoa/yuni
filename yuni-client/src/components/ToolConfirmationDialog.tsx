import { Wrench } from 'lucide-react'
import { useToolStore } from '../stores/toolStore'
import { Button } from './ui/button'
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog'

// Mounted once in AppShell so a confirmation request is visible on every page,
// not only while the chat page is open.
export function ToolConfirmationDialog() {
  const { pending, deciding, decide } = useToolStore()
  return <Dialog open={Boolean(pending)}><DialogContent className="tool-confirm-dialog"><DialogTitle>允许这次本地操作？</DialogTitle><DialogDescription>Yuni 只会执行本次操作，不会获得永久写入权限。</DialogDescription>{pending && <div className="tool-confirm-body"><div className="tool-summary"><Wrench size={17} /><div><strong>{pending.summary}</strong><span>{pending.risk === 'destructive' ? '高风险操作' : '写入操作'}</span></div></div>{pending.targetPaths.length > 0 && <div><span className="field-label">目标路径</span>{pending.targetPaths.map((path) => <code key={path}>{path}</code>)}</div>}{pending.diffPreview && <div><span className="field-label">修改预览</span><pre>{pending.diffPreview}</pre></div>}</div>}<div className="dialog-actions"><Button variant="ghost" disabled={deciding} onClick={() => void decide('reject')}>拒绝</Button><Button disabled={deciding} onClick={() => void decide('approve')}>允许一次</Button></div></DialogContent></Dialog>
}
