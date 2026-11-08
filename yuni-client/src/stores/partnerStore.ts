import { create } from 'zustand'
import { appApi } from '../services/api'
import type { AvatarPreset, CreatePartnerDraft, PartnerOptions, PartnerProfile } from '../types/domain'

export const emptyPartnerDraft: CreatePartnerDraft = { name: '', gender: null, avatarType: 'preset', avatarPresetId: 'mist', avatarFileId: null, avatarUrl: 'mock-avatar://mist', userNickname: '', personalityTags: [], interactionStyles: [], intimacyLevel: 'gentle_natural', boundaryText: '', customDescription: '' }

interface PartnerState {
  partner: PartnerProfile | null
  draft: CreatePartnerDraft
  options: PartnerOptions | null
  presets: AvatarPreset[]
  loading: boolean
  saving: boolean
  error: string | null
  load: () => Promise<void>
  loadConfiguration: () => Promise<void>
  setDraft: (patch: Partial<CreatePartnerDraft>) => void
  usePartnerAsDraft: () => void
  createPartner: () => Promise<PartnerProfile>
  updatePartner: () => Promise<PartnerProfile>
  uploadAvatar: (file: File) => Promise<void>
  reset: () => void
}

export const usePartnerStore = create<PartnerState>((set, get) => ({
  partner: null, draft: emptyPartnerDraft, options: null, presets: [], loading: false, saving: false, error: null,
  load: async () => { set({ loading: true, error: null }); try { set({ partner: await appApi.partner.getCurrent() }) } catch (error) { set({ error: error instanceof Error ? error.message : '伴侣资料加载失败' }) } finally { set({ loading: false }) } },
  loadConfiguration: async () => { if (get().options && get().presets.length) return; const [options, presets] = await Promise.all([appApi.partner.getOptions(), appApi.partner.getPresets()]); set({ options, presets }) },
  setDraft: (patch) => set({ draft: { ...get().draft, ...patch }, error: null }),
  usePartnerAsDraft: () => { const partner = get().partner; if (!partner) return; const { id: _id, relationshipStatus: _relationshipStatus, createdAt: _createdAt, updatedAt: _updatedAt, ...draft } = partner; void _id; void _relationshipStatus; void _createdAt; void _updatedAt; set({ draft: { ...draft, gender: partner.gender ?? null, customDescription: partner.customDescription ?? '' } }) },
  createPartner: async () => { set({ saving: true, error: null }); try { const partner = await appApi.partner.create(get().draft); set({ partner, draft: emptyPartnerDraft }); return partner } catch (error) { set({ error: error instanceof Error ? error.message : '创建失败' }); throw error } finally { set({ saving: false }) } },
  updatePartner: async () => { set({ saving: true, error: null }); try { const partner = await appApi.partner.update(get().draft); set({ partner }); return partner } catch (error) { set({ error: error instanceof Error ? error.message : '保存失败' }); throw error } finally { set({ saving: false }) } },
  uploadAvatar: async (file) => { set({ saving: true, error: null }); try { const uploaded = await appApi.partner.uploadAvatar(file); set({ draft: { ...get().draft, avatarType: 'uploaded', avatarPresetId: null, ...uploaded } }) } catch (error) { set({ error: error instanceof Error ? error.message : '头像上传失败' }); throw error } finally { set({ saving: false }) } },
  reset: () => set({ partner: null, draft: emptyPartnerDraft, error: null }),
}))
