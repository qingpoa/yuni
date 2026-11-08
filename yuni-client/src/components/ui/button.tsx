import type { ButtonHTMLAttributes } from 'react'
import { cn } from '../../lib/utils'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'icon'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
}

export function Button({ className, variant = 'primary', ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'ui-button',
        variant === 'primary' && 'ui-button-primary',
        variant === 'secondary' && 'ui-button-secondary',
        variant === 'ghost' && 'ui-button-ghost',
        variant === 'danger' && 'ui-button-danger',
        variant === 'icon' && 'ui-button-icon',
        className,
      )}
      {...props}
    />
  )
}
