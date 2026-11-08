import type { AppApi } from './appApi'
import { localApi } from './localApi'
import { mockApi } from './mockApi'

export const appApi: AppApi = import.meta.env.VITE_APP_API === 'mock' ? mockApi : localApi
