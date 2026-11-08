import { create } from 'zustand'
import { appApi } from '../services/api'
import type { UserProfile, UserProfileDraft } from '../types/domain'

interface ProfileState {
  profile: UserProfile | null
  loading: boolean
  error: string | null
  load: () => Promise<void>
  save: (draft: UserProfileDraft) => Promise<UserProfile>
  reset: () => void
}

export const useProfileStore = create<ProfileState>((set) => ({
  profile: null, loading: false, error: null,
  load: async () => {
    set({ loading: true, error: null })
    try { set({ profile: await appApi.profile.get() }) }
    catch (cause) { set({ error: cause instanceof Error ? cause.message : '个人资料加载失败' }) }
    finally { set({ loading: false }) }
  },
  save: async (draft) => {
    const profile = await appApi.profile.update(draft)
    set({ profile, error: null })
    return profile
  },
  reset: () => set({ profile: null, loading: false, error: null }),
}))
