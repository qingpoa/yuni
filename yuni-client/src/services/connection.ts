export interface AgentConnection { baseUrl: string; session: string }

declare global { interface Window { __YUNI_AGENT__?: AgentConnection } }

let connection: AgentConnection | null = null
export function setAgentConnection(next: AgentConnection) { connection = next }
export function getAgentConnection() {
  const development = import.meta.env.DEV && import.meta.env.VITE_AGENT_BASE_URL && import.meta.env.VITE_AGENT_SESSION
    ? { baseUrl: import.meta.env.VITE_AGENT_BASE_URL, session: import.meta.env.VITE_AGENT_SESSION }
    : null
  const current = connection ?? window.__YUNI_AGENT__ ?? development
  if (!current) throw new Error('本地 Agent 尚未连接')
  return current
}
