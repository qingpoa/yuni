export async function* parseSse(response: Response): AsyncGenerator<{ id: string | null; event: string; data: unknown }> {
  if (!response.ok) throw new Error(`Agent 请求失败（${response.status}）`)
  if (!response.body) throw new Error('Agent 未返回事件流')
  const reader = response.body.getReader(), decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const frames = buffer.split(/\r?\n\r?\n/)
      buffer = frames.pop() ?? ''
      for (const frame of frames) {
        let event = 'message', id: string | null = null
        const data: string[] = []
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          if (line.startsWith('id:')) id = line.slice(3).trim()
          if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
        }
        if (!data.length) continue
        const raw = data.join('\n')
        let parsed: unknown
        try { parsed = JSON.parse(raw) } catch { parsed = raw }
        yield { id, event, data: parsed }
      }
      if (done) break
    }
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
