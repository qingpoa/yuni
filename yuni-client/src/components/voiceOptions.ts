// Provider voice IDs are implementation details; the settings UI presents names.
import type { VoiceOption } from '../types/domain'

export type VoiceChoice = VoiceOption

const MINIMAX_HOSTS = new Set(['api.minimax.cn', 'api.minimaxi.com', 'api.minimax.io', 'api-bj.minimaxi.com'])

export function voiceService(baseUrl: string, model: string): 'minimax' | 'cosyvoice' | 'openai' | 'custom' {
  try {
    if (MINIMAX_HOSTS.has(new URL(baseUrl).hostname.toLowerCase())) return 'minimax'
  } catch { /* An incomplete address can still be edited. */ }
  if (/^(?:FunAudioLLM\/)?CosyVoice2-0\.5B$/i.test(model.trim())) return 'cosyvoice'
  if (/^(?:gpt-4o-mini-tts(?:-[\d-]+)?|tts-1(?:-hd)?)$/i.test(model.trim())) return 'openai'
  return 'custom'
}

export function builtinVoiceChoices(baseUrl: string, model: string): VoiceChoice[] {
  const system = (id: string, name: string): VoiceChoice => ({ id, name, kind: 'system' })
  switch (voiceService(baseUrl, model)) {
    case 'minimax':
      return [
        system('female-shaonv', '少女音色'),
        system('female-yujie', '御姐音色'),
        system('female-tianmei', '甜美女声'),
        system('male-qn-qingse', '青涩青年'),
      ]
    case 'cosyvoice':
      return [
        ['anna', 'Anna · 沉稳女声'], ['bella', 'Bella · 激情女声'],
        ['claire', 'Claire · 温柔女声'], ['diana', 'Diana · 欢快女声'],
        ['alex', 'Alex · 沉稳男声'], ['benjamin', 'Benjamin · 低沉男声'],
        ['charles', 'Charles · 磁性男声'], ['david', 'David · 欢快男声'],
      ].map(([id, name]) => system(`${model.trim()}:${id}`, name))
    case 'openai': {
      const names = /^tts-1/i.test(model.trim())
        ? ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
        : ['alloy', 'ash', 'ballad', 'coral', 'echo', 'fable', 'nova', 'onyx', 'sage', 'shimmer', 'verse', 'marin', 'cedar']
      return names.map(id => system(id, id.charAt(0).toUpperCase() + id.slice(1)))
    }
    default:
      return []
  }
}

export function sameVoiceEndpoint(left: string, right: string): boolean {
  return Boolean(left.trim() && right.trim() && left.trim().replace(/\/+$/, '') === right.trim().replace(/\/+$/, ''))
}
