import { Sparkles } from 'lucide-react'
import type { CSSProperties } from 'react'
import { cn } from '../lib/utils'

export function CompanionAvatar({
  className,
  label = '伴侣',
  color,
  src,
}: {
  className?: string
  label?: string
  color?: string
  src?: string
}) {
  if (src && !src.startsWith('mock-avatar://')) {
    return <img className={cn('companion-avatar-image', className)} src={src} alt={`${label}的头像`} />
  }

  const presetColor = color ?? ({
    'mock-avatar://mist': '#819587',
    'mock-avatar://river': '#657b83',
    'mock-avatar://sunset': '#a0837d',
    'mock-avatar://night': '#7b788d',
  }[src ?? ''] || undefined)

  return (
    <span
      className={cn('companion-avatar-placeholder', className)}
      style={presetColor ? ({ '--avatar-color': presetColor } as CSSProperties) : undefined}
      role="img"
      aria-label={`${label}的默认头像`}
    >
      <Sparkles aria-hidden="true" />
    </span>
  )
}
