import type { AllowedRoot, AvatarPreset, BootstrapData, ChatMessage, ChatStreamEvent, CreatePartnerDraft, CursorPage, DataImportResult, DirectoryListing, ImageConfigInput, MemoryItem, MemoryType, MemoryFilter, MemoryDraft, MemoryPatch, MemoryEntityItem, MemoryCandidateItem, ModelConfigInput, ModelListInput, PartnerOptions, PartnerProfile, ProactiveSettingsInput, Reminder, Settings, ToolCall, ToolDecision, UserProfile, UserProfileDraft, VoiceConfigInput, VoicePreviewInput, VoiceListInput, VoiceOption } from '../types/domain'

import type { AppEventFrame, MessageAccepted, StickerOption } from '../types/domain'

export interface AppApi {
  events: { stream(afterId: number | null, signal: AbortSignal): AsyncGenerator<AppEventFrame> }
  bootstrap(): Promise<BootstrapData>
  profile: {
    get(): Promise<UserProfile>
    update(draft: UserProfileDraft): Promise<UserProfile>
    uploadAvatar(file: File): Promise<{ avatarFileId: number; avatarUrl: string }>
  }
  partner: {
    getCurrent(): Promise<PartnerProfile | null>
    getOptions(): Promise<PartnerOptions>
    getPresets(): Promise<AvatarPreset[]>
    create(draft: CreatePartnerDraft): Promise<PartnerProfile>
    update(draft: CreatePartnerDraft): Promise<PartnerProfile>
    uploadAvatar(file: File): Promise<{ avatarFileId: number; avatarUrl: string }>
  }
  chat: {
    stickers(): Promise<StickerOption[]>
    stickerContent(stickerId: number): Promise<Blob>
    imageContent(imageId: string): Promise<Blob>
    voiceContent(voiceId: string): Promise<Blob>
    submit(content: string, clientMessageId: string, signal?: AbortSignal, stickerId?: number): Promise<MessageAccepted>
    submitRetry(messageId: number, signal?: AbortSignal): Promise<MessageAccepted>
    history(cursor: string | null, limit?: number): Promise<CursorPage<ChatMessage> & { proactiveCursor?: number }>
    stream(content: string, clientMessageId: string, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent>
    retry(messageId: number, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent>
    pollProactive(afterId: number | null): Promise<{ items: ChatMessage[]; cursor: number }>
    triggerProactive(): Promise<ChatMessage>
  }
  memory: {
    create(draft: MemoryDraft): Promise<MemoryItem>
    entities(): Promise<MemoryEntityItem[]>
    candidates(): Promise<MemoryCandidateItem[]>
    decideCandidate(id: number, decision: 'approve' | 'reject'): Promise<void>
    history(id: number): Promise<MemoryItem[]>
    list(type: MemoryType | '', cursor: string | null, limit?: number, filter?: MemoryFilter): Promise<CursorPage<MemoryItem>>
    update(id: number, patch: MemoryPatch): Promise<MemoryItem>
    resolveConflict(id: number, content: string, options?: { action?: 'keep' | 'adopt' | 'coexist'; qualifier?: string; otherQualifier?: string }): Promise<MemoryItem>
    remove(id: number): Promise<void>
    clear(): Promise<void>
  }
  settings: {
    get(): Promise<Settings>
    setMemoryEnabled(enabled: boolean): Promise<Settings>
    saveProactive(config: ProactiveSettingsInput): Promise<Settings>
    saveModel(config: ModelConfigInput): Promise<Settings>
    clearModelKey(): Promise<Settings>
    listModels(config: ModelListInput): Promise<string[]>
    saveEmbedding(config: ModelConfigInput): Promise<Settings>
    clearEmbeddingKey(): Promise<Settings>
    saveImage(config: ImageConfigInput): Promise<Settings>
    clearImageKey(): Promise<Settings>
    saveVoice(config: VoiceConfigInput): Promise<Settings>
    clearVoiceKey(): Promise<Settings>
    previewVoice(config: VoicePreviewInput): Promise<Blob>
    listVoices(config: VoiceListInput): Promise<VoiceOption[]>
    rebuildExpressions(): Promise<void>
  }
  roots: { list(): Promise<AllowedRoot[]>; add(path: string): Promise<AllowedRoot>; remove(id: number): Promise<void> }
  directories: { browse(path: string | null): Promise<DirectoryListing> }
  tools: { list(): Promise<ToolCall[]>; decide(callId: string, decision: ToolDecision): Promise<ToolCall> }
  data: { export(): Promise<Blob>; import(file: File): Promise<DataImportResult>; clear(): Promise<void> }
  reminders: { due(): Promise<Reminder[]>; markDelivered(id: number): Promise<void> }
  system: { openLogs(): Promise<void> }
}
