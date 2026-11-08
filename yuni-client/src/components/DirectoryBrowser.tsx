import { ArrowUp, ChevronRight, FolderOpen, LoaderCircle, Monitor, RefreshCw } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { appApi } from '../services/api'
import type { DirectoryListing } from '../types/domain'
import { Button } from './ui/button'
import { Tooltip } from './ui/tooltip'

export function DirectoryBrowser({ disabled, onSelect }: { disabled: boolean; onSelect: (path: string | null) => void }) {
  const [listing, setListing] = useState<DirectoryListing | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [requestedPath, setRequestedPath] = useState<string | null>(null)
  const generation = useRef(0)
  const browse = useCallback(async (path: string | null) => {
    const current = ++generation.current
    setLoading(true)
    setError('')
    setRequestedPath(path)
    onSelect(null)
    try {
      const result = await appApi.directories.browse(path)
      if (current !== generation.current) return
      setListing(result)
      onSelect(result.path)
    } catch (caught) {
      if (current === generation.current) setError(caught instanceof Error ? caught.message : '目录加载失败')
    } finally {
      if (current === generation.current) setLoading(false)
    }
  }, [onSelect])

  const invalidate = useCallback(() => { generation.current++ }, [])
  useEffect(() => {
    let disposed = false
    void Promise.resolve().then(() => { if (!disposed) void browse(null) })
    return () => { disposed = true; invalidate() }
  }, [browse, invalidate])

  return <div className="directory-browser" aria-busy={loading}>
    <div className="directory-toolbar">
      <Tooltip label="此电脑"><Button variant="icon" aria-label="此电脑" disabled={disabled || loading} onClick={() => void browse(null)}><Monitor size={16} /></Button></Tooltip>
      <Tooltip label="上一级"><Button variant="icon" aria-label="上一级" disabled={disabled || loading || !listing?.path} onClick={() => void browse(listing?.parentPath ?? null)}><ArrowUp size={16} /></Button></Tooltip>
      <span className="directory-path" title={listing?.path ?? '此电脑'}>{listing?.path ?? '此电脑'}</span>
    </div>
    <div className="directory-entries">
      {loading ? <div className="directory-empty"><LoaderCircle className="spin" size={18} /><span>正在加载</span></div>
        : error ? <div className="directory-empty" role="alert"><span>{error}</span><Button variant="secondary" disabled={disabled} onClick={() => void browse(requestedPath)}><RefreshCw size={14} />重试</Button></div>
          : listing?.items.length ? listing.items.map(item => <button type="button" className="directory-entry" key={item.path} disabled={disabled} onClick={() => void browse(item.path)}><FolderOpen size={18} /><span>{item.name}</span><ChevronRight size={15} /></button>)
            : <div className="directory-empty">没有子文件夹</div>}
    </div>
  </div>
}
