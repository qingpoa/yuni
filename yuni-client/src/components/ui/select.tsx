import * as SelectPrimitive from '@radix-ui/react-select'
import { Check, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'

interface SelectOption {
  value: string
  label: string
  description?: string
}

interface SelectProps {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  label: string
  disabled?: boolean
}

export function Select({ value, onValueChange, options, label, disabled }: SelectProps) {
  const [open, setOpen] = useState(false)
  return <SelectPrimitive.Root open={open && !disabled} onOpenChange={setOpen} value={value} onValueChange={onValueChange} disabled={disabled}>
    <SelectPrimitive.Trigger className="app-select-trigger" aria-label={label}>
      <SelectPrimitive.Value />
      <SelectPrimitive.Icon className="app-select-chevron"><ChevronDown size={16} /></SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
    <SelectPrimitive.Portal>
      <SelectPrimitive.Content className="app-select-content" position="popper" sideOffset={6} collisionPadding={12} onEscapeKeyDown={event => { event.preventDefault(); setOpen(false) }}>
        <SelectPrimitive.ScrollUpButton className="app-select-scroll"><ChevronUp size={14} /></SelectPrimitive.ScrollUpButton>
        <SelectPrimitive.Viewport className="app-select-viewport">
          {options.map(option => <SelectPrimitive.Item className="app-select-item" key={option.value} value={option.value} textValue={option.label}>
            <SelectPrimitive.ItemText><span className="app-select-label">{option.label}</span>{option.description && <span className="app-select-description">{option.description}</span>}</SelectPrimitive.ItemText>
            <SelectPrimitive.ItemIndicator className="app-select-check"><Check size={15} /></SelectPrimitive.ItemIndicator>
          </SelectPrimitive.Item>)}
        </SelectPrimitive.Viewport>
        <SelectPrimitive.ScrollDownButton className="app-select-scroll"><ChevronDown size={14} /></SelectPrimitive.ScrollDownButton>
      </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
  </SelectPrimitive.Root>
}
