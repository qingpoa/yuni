export interface ApiResponse<T> {
  code: number
  message: string
  data: T | null
}

export interface BootstrapData {
  initialized: boolean
  hasCompanion: boolean
  memoryEnabled: boolean
  modelConfigured: boolean
  agentVersion: string
  protocolVersion: number
}

export type AvatarType = 'preset' | 'uploaded'
export type PartnerGender = 'male' | 'female'

export interface PartnerProfile {
  id: number
  name: string
  gender: PartnerGender | null
  avatarType: AvatarType
  avatarPresetId: string | null
  avatarFileId: number | null
  avatarUrl: string
  userNickname: string
  relationshipStatus: 'stable_partner'
  personalityTags: string[]
  interactionStyles: string[]
  intimacyLevel: string
  boundaryText: string
  customDescription: string
  createdAt: string
  updatedAt: string
}

export interface CreatePartnerDraft {
  name: string
  gender: PartnerGender | null
  avatarType: AvatarType
  avatarPresetId: string | null
  avatarFileId: number | null
  avatarUrl: string
  userNickname: string
  personalityTags: string[]
  interactionStyles: string[]
  intimacyLevel: string
  boundaryText: string
  customDescription: string
}

export interface UserProfile {
  id: number
  nickname: string
  avatarType: AvatarType
  avatarPresetId: string | null
  avatarFileId: number | null
  avatarUrl: string
  createdAt: string
  updatedAt: string
}

export type UserProfileDraft = Pick<UserProfile, 'nickname' | 'avatarType' | 'avatarPresetId' | 'avatarFileId' | 'avatarUrl'>

export interface SelectOption { value: string; label: string }

export interface PartnerOptions {
  personalityTags: SelectOption[]
  interactionStyles: SelectOption[]
  intimacyLevels: SelectOption[]
}

export interface AvatarPreset { id: string; name: string; url: string; color: string }
export interface HelpResource { label: string; url: string }
export interface StickerOption { id: number; description: string; imageUrl: string }
export interface GeneratedImageAttachment { id: string; url: string; description: string }

export type VoiceTriggerMode = 'explicit_only' | 'occasional' | 'always'

export interface VoiceConfigSummary {
  enabled: boolean
  baseUrl: string
  modelName: string
  voice: string
  triggerMode: VoiceTriggerMode
  configured: boolean
}

export interface VoiceConfigInput {
  enabled: boolean
  baseUrl: string
  modelName: string
  apiKey?: string
  voice: string
  triggerMode: VoiceTriggerMode
}

export interface VoiceListInput {
  baseUrl: string
  apiKey?: string
}

export interface VoiceOption {
  id: string
  name: string
  kind: 'system' | 'cloned' | 'generated'
}

export interface VoicePreviewInput {
  baseUrl: string
  modelName: string
  apiKey?: string
  voice: string
  text?: string
}

export interface GeneratedVoiceAttachment {
  id: string
  url: string
  duration: number
}

export interface ChatMessage {
  localId: string
  id: number | null
  clientMessageId: string | null
  role: 'user' | 'assistant'
  content: string
  replyParts?: string[]
  sticker?: StickerOption | null
  image?: GeneratedImageAttachment | null
  voice?: GeneratedVoiceAttachment | null
  visibleParts?: number
  status: 'pending' | 'streaming' | 'succeeded' | 'failed'
  messageType: 'normal' | 'safety'
  helpResources: HelpResource[]
  parentMessageId: number | null
  errorCode: number | null
  errorMessage: string | null
  createdAt: string
}

export type MemoryType = 'user_fact' | 'user_preference' | 'important_event' | 'relationship_preference' | 'emotion_pattern' | 'boundary'

export type MemoryKind = 'fact' | 'event' | 'ongoing_state'
export interface MemoryFilter { kind?: MemoryKind; status?: string; entityId?: number; q?: string }
export interface MemoryTimeMeta { planned_start?: string | null; planned_end?: string | null; actual_start?: string | null; actual_end?: string | null; start_window_end?: string | null; end_window_start?: string | null }
export interface MemoryEntityItem { id: number; canonicalName: string; entityType: string; relationToUser: string | null; aliases: string[]; memoryCount: number }
export interface MemoryCandidateItem { id: number; title: string; content: string; reason: string | null; kind: MemoryKind }
export interface MemoryDraft {
  type: MemoryType; title: string; content: string; importance: 1 | 2 | 3 | 4 | 5
  subjectKey: string; kind: MemoryKind; qualifier?: string; subjectPerson?: string
  domain?: 'real_world' | 'conversation' | 'fictional'; sceneKey?: string | null
  eventStatus?: 'planned' | 'in_progress' | 'completed' | 'cancelled' | null
  outcome?: string | null; timeText?: string | null; timePrecision?: 'exact' | 'day' | 'window' | 'recurring' | 'unknown'
  validFrom?: string | null; validUntil?: string | null; timeMeta?: MemoryTimeMeta
}
export type MemoryPatch = Pick<MemoryDraft, 'type' | 'title' | 'content' | 'importance'> & Partial<Omit<MemoryDraft, 'subjectKey' | 'subjectPerson' | 'domain' | 'sceneKey'>> & { mode?: 'correct' | 'transition' }

export interface MemoryItem {
  id: number
  type: MemoryType
  title: string
  content: string
  sourceMessageId: number | null
  sourceLabel: string
  sourceExcerpt: string | null
  importance: 1 | 2 | 3 | 4 | 5
  embeddingStatus: 'pending' | 'indexed' | 'failed'
  status: 'active' | 'historical' | 'conflicted' | 'retracted' | 'deleted'
  conflictContent: string | null
  kind?: MemoryKind
  domain?: 'real_world' | 'conversation' | 'fictional'
  sceneKey?: string | null
  subjectEntityId?: number | null
  subjectKey?: string
  qualifier?: string
  eventStatus?: 'planned' | 'in_progress' | 'completed' | 'cancelled' | null
  outcome?: string | null
  timeText?: string | null
  timePrecision?: string
  validFrom?: string | null
  validUntil?: string | null
  freshUntil?: string | null
  timeMeta?: MemoryTimeMeta
  entityIds?: number[]
  supersededById?: number | null
  conflictWithId?: number | null
  sources?: Array<{ messageId: number; evidenceText: string; messageSentAt: string | null }>
  createdAt: string
  updatedAt: string
}

export interface CursorPage<T> { items: T[]; nextCursor: string | null }

export interface ModelConfigSummary {
  provider: 'openai_compatible' | 'openai'
  baseUrl: string
  modelName: string
  configured: boolean
}

export interface ModelConfigInput {
  provider: ModelConfigSummary['provider']
  baseUrl: string
  modelName: string
  apiKey?: string
}

export interface ModelListInput { baseUrl?: string; apiKey?: string }

export type ImageModelSize = '1024x1024' | '768x1024' | '1024x768'

export interface ImageConfigSummary {
  enabled: boolean
  baseUrl: string
  modelName: string
  defaultSize: ImageModelSize
  stylePrefix: string
  configured: boolean
}

export interface ImageConfigInput {
  enabled: boolean
  baseUrl: string
  modelName: string
  apiKey?: string
  defaultSize: ImageModelSize
  stylePrefix: string
}

export interface ProactiveSettingsInput {
  proactiveChatEnabled: boolean
  proactiveIntervalMinutes: number
  quietHoursStart: string
  quietHoursEnd: string
}

export interface Settings {
  memoryEnabled: boolean
  proactiveChatEnabled?: boolean
  proactiveIntervalMinutes?: number
  quietHoursStart?: string
  quietHoursEnd?: string
  privacyVersion: string
  model: ModelConfigSummary
  embedding?: ModelConfigSummary
  image?: ImageConfigSummary
  voice?: VoiceConfigSummary
}

export interface AllowedRoot { id: number; path: string; displayName: string; createdAt: string }
export interface DirectoryListing { path: string | null; parentPath: string | null; items: Array<{ name: string; path: string }> }
export type ToolRisk = 'read' | 'write' | 'destructive'
export type ToolStatus = 'awaiting_confirmation' | 'running' | 'succeeded' | 'failed' | 'rejected'

export interface ToolCall {
  id: string
  toolName: string
  risk: ToolRisk
  status: ToolStatus
  summary: string
  targetPaths: string[]
  diffPreview: string | null
  errorMessage: string | null
  createdAt: string
  completedAt: string | null
}

export type ToolDecision = 'approve' | 'reject'

export type ChatStreamEvent =
  | { type: 'message_created'; messageId: number }
  | { type: 'batch_started'; userMessageIds: number[]; parentMessageId: number }
  | { type: 'delta'; text: string }
  | { type: 'tool_confirmation_required'; toolCall: ToolCall }
  | { type: 'tool_started'; toolCall: ToolCall }
  | { type: 'tool_completed'; toolCall: ToolCall }
  | { type: 'reply_completed'; assistantMessageId: number; content: string; replyParts: string[]; sticker?: StickerOption | null; image?: GeneratedImageAttachment | null; voice?: GeneratedVoiceAttachment | null; messageType: 'normal' | 'safety'; helpResources: HelpResource[]; parentMessageId?: number | null; createdAt?: string }
  | { type: 'completed'; assistantMessageId?: number; content?: string; replyParts?: string[]; sticker?: StickerOption | null; image?: GeneratedImageAttachment | null; voice?: GeneratedVoiceAttachment | null; messageType?: 'normal' | 'safety'; helpResources?: HelpResource[]; replyCount?: number; parentMessageId?: number | null; createdAt?: string }
  | { type: 'error'; code: number; message: string }

export interface DataImportResult { hasCompanion: boolean; importedMessages: number; importedMemories: number }
export interface Reminder { id: number; title: string; note: string; remindAt: string }

export interface MessageAccepted { message: ChatMessage; turnStatus: string }
export type AppStreamEvent =
  | { type: 'ready'; cursor: number; reset: boolean; messages: ChatMessage[]; activeUserMessageIds: number[]; toolCalls: ToolCall[]; reminders: Reminder[] }
  | { type: 'user_message_created'; message: ChatMessage }
  | { type: 'turn_queued'; userMessageIds: number[]; parentMessageId: number }
  | { type: 'reminder_due'; reminder: Reminder }
  | { type: 'state_reset'; reason: string }
  | (ChatStreamEvent & { userMessageIds?: number[]; parentMessageId?: number | null; proactive?: boolean })
export interface AppEventFrame { id: number | null; event: AppStreamEvent }

export class ApiError extends Error {
  code: number
  constructor(code: number, message: string) { super(message); this.code = code }
}
