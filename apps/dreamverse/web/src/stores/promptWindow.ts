import { createManagedStore, type ManagedStore } from './createManagedStore';

function normalizePromptList(
  nextPrompts: unknown,
): string[] {
  if (!Array.isArray(nextPrompts)) {
    return [];
  }

  return nextPrompts
    .map((prompt) =>
      typeof prompt === 'string' ? prompt.trim() : '',
    )
    .filter((prompt) => prompt.length > 0);
}

export interface PromptWindowState {
  simpleMode: boolean;
  editableMode: boolean;
  storyPresets: Record<string, unknown>[];
  selectedPresetId: string;
  selectedPreset: Record<string, unknown> | null;
  previewSegments: string[];
  maxCuratedPromptCount: number;
  curatedPromptLimit: number;
  outboundCuratedPrompts: string[];
  seedPrompts: string[];
  currentPromptWindowPrompts: string[];
  outboundSessionPrompts: string[];
  canJoinSession: boolean;
  editableSegments: string[];
  sanitizedEditableSegments: string[];
  editableCanJoin: boolean;
  editableDirty: boolean;
  customPresetId: string;
  customPresetLabel: string;
  simplePromptOptions: Record<string, unknown>[];
  selectedSimplePrompt: Record<string, unknown> | null;
  selectedSimplePromptId: string;
  simplePromptDraft: string;
  simplePromptsLoading: boolean;
  simplePromptsError: string;
  simpleInputImageName: string;
  simpleInputImageMimeType: string;
  simpleInputImageDataUrl: string;
  simpleInputImageLoading: boolean;
  simpleInputImageError: string;
}

function derivePromptWindowState(
  state: PromptWindowState,
): PromptWindowState {
  const storyPresets = Array.isArray(state.storyPresets)
    ? state.storyPresets
    : [];
  const simplePromptOptions = Array.isArray(state.simplePromptOptions)
    ? state.simplePromptOptions
    : [];

  let selectedPresetId =
    typeof state.selectedPresetId === 'string'
      ? state.selectedPresetId
      : '';
  if (
    selectedPresetId &&
    storyPresets.length > 0 &&
    !storyPresets.some((preset) => preset.id === selectedPresetId)
  ) {
    selectedPresetId = storyPresets[0].id as string;
  }
  const selectedPreset =
    storyPresets.find((preset) => preset.id === selectedPresetId) ||
    null;

  let selectedSimplePromptId =
    typeof state.selectedSimplePromptId === 'string'
      ? state.selectedSimplePromptId
      : '';
  if (
    selectedSimplePromptId &&
    !simplePromptOptions.some(
      (prompt) => prompt.id === selectedSimplePromptId,
    )
  ) {
    selectedSimplePromptId = '';
  }
  const selectedSimplePrompt =
    simplePromptOptions.find(
      (prompt) => prompt.id === selectedSimplePromptId,
    ) || null;

  const editableSegments = Array.isArray(state.editableSegments)
    ? state.editableSegments
    : [];
  const sanitizedEditableSegments =
    normalizePromptList(editableSegments);
  const editableCanJoin = sanitizedEditableSegments.length >= 2;

  const previewSegments = state.editableMode
    ? sanitizedEditableSegments
    : normalizePromptList(
        (selectedPreset as Record<string, unknown> | null)
          ?.segment_prompts,
      );

  const maxCuratedPromptCount = previewSegments.length;
  let curatedPromptLimit = Number.parseInt(
    state.curatedPromptLimit as unknown as string,
    10,
  );
  if (!Number.isFinite(curatedPromptLimit)) {
    curatedPromptLimit = 0;
  }
  if (maxCuratedPromptCount === 0) {
    curatedPromptLimit = 0;
  } else if (
    curatedPromptLimit < 1 ||
    curatedPromptLimit > maxCuratedPromptCount
  ) {
    curatedPromptLimit = maxCuratedPromptCount;
  }

  const outboundCuratedPrompts =
    curatedPromptLimit > 0
      ? previewSegments.slice(0, curatedPromptLimit)
      : [];
  const seedPrompts = normalizePromptList(state.seedPrompts);
  const outboundSessionPrompts = state.simpleMode
    ? normalizePromptList([state.simplePromptDraft])
    : outboundCuratedPrompts;
  const currentPromptWindowPrompts =
    seedPrompts.length > 0 ? seedPrompts : outboundSessionPrompts;
  const canJoinSession = state.simpleMode
    ? outboundSessionPrompts.length > 0
    : state.editableMode
      ? editableCanJoin
      : Boolean(selectedPreset);

  return {
    ...state,
    storyPresets,
    selectedPresetId,
    selectedPreset,
    simplePromptOptions,
    selectedSimplePromptId,
    selectedSimplePrompt,
    editableSegments,
    sanitizedEditableSegments,
    editableCanJoin,
    previewSegments,
    maxCuratedPromptCount,
    curatedPromptLimit,
    seedPrompts,
    outboundCuratedPrompts,
    outboundSessionPrompts,
    currentPromptWindowPrompts,
    canJoinSession,
  } as unknown as PromptWindowState;
}

const DEFAULT_PROMPT_WINDOW_STATE: PromptWindowState = {
  simpleMode: false,
  editableMode: false,
  storyPresets: [],
  selectedPresetId: '',
  selectedPreset: null,
  previewSegments: [],
  maxCuratedPromptCount: 0,
  curatedPromptLimit: 0,
  outboundCuratedPrompts: [],
  seedPrompts: [],
  currentPromptWindowPrompts: [],
  outboundSessionPrompts: [],
  canJoinSession: false,
  editableSegments: [],
  sanitizedEditableSegments: [],
  editableCanJoin: false,
  editableDirty: false,
  customPresetId: '',
  customPresetLabel: '',
  simplePromptOptions: [],
  selectedSimplePrompt: null,
  selectedSimplePromptId: '',
  simplePromptDraft: '',
  simplePromptsLoading: false,
  simplePromptsError: '',
  simpleInputImageName: '',
  simpleInputImageMimeType: '',
  simpleInputImageDataUrl: '',
  simpleInputImageLoading: false,
  simpleInputImageError: '',
};

export interface SimpleInputImageArgs {
  name?: string;
  mimeType?: string;
  dataUrl?: string;
}

export interface SeedEditableOptions {
  defaultCustomPresetId?: string;
  defaultCustomPresetLabel?: string;
}

export interface ReplacePromptWindowOptions {
  syncEditable?: boolean;
}

export type PromptWindowStore = ManagedStore<PromptWindowState> & {
  reset: (overrides?: Partial<PromptWindowState>) => PromptWindowState;
  resetSessionState: () => PromptWindowState;
  setModeContext: (
    modeContext: Partial<PromptWindowState>,
  ) => PromptWindowState;
  setStoryPresets: (
    nextStoryPresets: Record<string, unknown>[],
  ) => PromptWindowState;
  appendStoryPreset: (
    preset: Record<string, unknown>,
  ) => PromptWindowState;
  seedEditableFromPreset: (
    preset: Record<string, unknown> | null,
    options?: SeedEditableOptions,
  ) => PromptWindowState;
  setSelectedPresetId: (
    selectedPresetId: string,
  ) => PromptWindowState;
  setCuratedPromptLimit: (
    curatedPromptLimit: number,
  ) => PromptWindowState;
  setSeedPrompts: (nextPrompts: unknown) => PromptWindowState;
  replacePromptWindow: (
    nextPrompts: unknown,
    options?: ReplacePromptWindowOptions,
  ) => PromptWindowState;
  addEditableSegment: () => PromptWindowState;
  removeEditableSegment: (index: number) => PromptWindowState;
  updateEditableSegment: (
    index: number,
    value: string,
  ) => PromptWindowState;
  setCustomPresetId: (
    customPresetId: string,
  ) => PromptWindowState;
  setCustomPresetLabel: (
    customPresetLabel: string,
  ) => PromptWindowState;
  setSimplePromptOptions: (
    simplePromptOptions: Record<string, unknown>[],
  ) => PromptWindowState;
  applySelectedSimplePrompt: (
    promptId: string,
    options?: { overwriteDraft?: boolean },
  ) => PromptWindowState;
  setSimplePromptDraft: (
    simplePromptDraft: string,
  ) => PromptWindowState;
  setSimplePromptsLoading: (
    simplePromptsLoading: boolean,
  ) => PromptWindowState;
  setSimplePromptsError: (
    simplePromptsError: string,
  ) => PromptWindowState;
  setSimpleInputImage: (
    args?: SimpleInputImageArgs,
  ) => PromptWindowState;
  clearSimpleInputImage: () => PromptWindowState;
  setSimpleInputImageLoading: (
    simpleInputImageLoading: boolean,
  ) => PromptWindowState;
  setSimpleInputImageError: (
    simpleInputImageError: string,
  ) => PromptWindowState;
};

export function createPromptWindowStore(
  initialState: Partial<PromptWindowState> = {},
): PromptWindowStore {
  const store = createManagedStore<PromptWindowState>(
    {
      ...DEFAULT_PROMPT_WINDOW_STATE,
      ...initialState,
    } as PromptWindowState,
    derivePromptWindowState,
  );

  return {
    ...store,
    reset(overrides: Partial<PromptWindowState> = {}) {
      return store.set({
        ...DEFAULT_PROMPT_WINDOW_STATE,
        ...initialState,
        ...overrides,
      } as PromptWindowState);
    },
    resetSessionState() {
      return store.patch({
        seedPrompts: [],
        simpleInputImageLoading: false,
        simpleInputImageError: '',
      });
    },
    setModeContext(modeContext: Partial<PromptWindowState>) {
      return store.patch(modeContext);
    },
    setStoryPresets(nextStoryPresets: Record<string, unknown>[]) {
      return store.patch({
        storyPresets: Array.isArray(nextStoryPresets)
          ? nextStoryPresets
          : [],
      });
    },
    appendStoryPreset(preset: Record<string, unknown>) {
      return store.update((state) => ({
        ...state,
        storyPresets: [...state.storyPresets, preset],
      }));
    },
    seedEditableFromPreset(
      preset: Record<string, unknown> | null,
      {
        defaultCustomPresetId = '',
        defaultCustomPresetLabel = '',
      }: SeedEditableOptions = {},
    ) {
      return store.update((state) => {
        if (
          preset &&
          Array.isArray(preset.segment_prompts) &&
          preset.segment_prompts.length > 0
        ) {
          return {
            ...state,
            editableSegments: [
              ...(preset.segment_prompts as string[]),
            ],
            customPresetId: `${preset.id}_custom`,
            customPresetLabel: `${preset.label} (Custom)`,
            editableDirty: false,
          };
        }

        return {
          ...state,
          editableSegments: ['', ''],
          customPresetId:
            state.customPresetId.trim() || defaultCustomPresetId,
          customPresetLabel:
            state.customPresetLabel.trim() || defaultCustomPresetLabel,
          editableDirty: false,
        };
      });
    },
    setSelectedPresetId(selectedPresetId: string) {
      return store.patch({ selectedPresetId });
    },
    setCuratedPromptLimit(curatedPromptLimit: number) {
      return store.patch({ curatedPromptLimit });
    },
    setSeedPrompts(nextPrompts: unknown) {
      return store.patch({
        seedPrompts: normalizePromptList(nextPrompts),
      });
    },
    replacePromptWindow(
      nextPrompts: unknown,
      { syncEditable = false }: ReplacePromptWindowOptions = {},
    ) {
      const normalized = normalizePromptList(nextPrompts);
      return store.update((state) => ({
        ...state,
        seedPrompts: normalized,
        editableSegments:
          syncEditable &&
          state.editableMode &&
          normalized.length > 0
            ? [...normalized]
            : state.editableSegments,
        editableDirty:
          syncEditable &&
          state.editableMode &&
          normalized.length > 0
            ? false
            : state.editableDirty,
      }));
    },
    addEditableSegment() {
      return store.update((state) => ({
        ...state,
        editableSegments: [...state.editableSegments, ''],
        editableDirty: true,
      }));
    },
    removeEditableSegment(index: number) {
      return store.update((state) => {
        const nextSegments = state.editableSegments.filter(
          (_, i) => i !== index,
        );
        return {
          ...state,
          editableSegments:
            nextSegments.length > 0 ? nextSegments : [''],
          editableDirty: true,
        };
      });
    },
    updateEditableSegment(index: number, value: string) {
      return store.update((state) => ({
        ...state,
        editableSegments: state.editableSegments.map((segment, i) =>
          i === index ? value : segment,
        ),
        editableDirty: true,
      }));
    },
    setCustomPresetId(customPresetId: string) {
      return store.patch({
        customPresetId,
        editableDirty: true,
      });
    },
    setCustomPresetLabel(customPresetLabel: string) {
      return store.patch({
        customPresetLabel,
        editableDirty: true,
      });
    },
    setSimplePromptOptions(
      simplePromptOptions: Record<string, unknown>[],
    ) {
      return store.patch({ simplePromptOptions });
    },
    applySelectedSimplePrompt(
      promptId: string,
      { overwriteDraft = true }: { overwriteDraft?: boolean } = {},
    ) {
      return store.update((state) => {
        const selectedPrompt =
          state.simplePromptOptions.find(
            (item) => item.id === promptId,
          ) || null;
        return {
          ...state,
          selectedSimplePromptId:
            (selectedPrompt?.id as string) || '',
          simplePromptDraft:
            selectedPrompt && overwriteDraft
              ? (selectedPrompt.prompt as string)
              : state.simplePromptDraft,
        };
      });
    },
    setSimplePromptDraft(simplePromptDraft: string) {
      return store.patch({ simplePromptDraft });
    },
    setSimplePromptsLoading(simplePromptsLoading: boolean) {
      return store.patch({ simplePromptsLoading });
    },
    setSimplePromptsError(simplePromptsError: string) {
      return store.patch({ simplePromptsError });
    },
    setSimpleInputImage({
      name = '',
      mimeType = '',
      dataUrl = '',
    }: SimpleInputImageArgs = {}) {
      return store.patch({
        simpleInputImageName: name,
        simpleInputImageMimeType: mimeType,
        simpleInputImageDataUrl: dataUrl,
        simpleInputImageError: '',
      });
    },
    clearSimpleInputImage() {
      return store.patch({
        simpleInputImageName: '',
        simpleInputImageMimeType: '',
        simpleInputImageDataUrl: '',
        simpleInputImageError: '',
      });
    },
    setSimpleInputImageLoading(simpleInputImageLoading: boolean) {
      return store.patch({ simpleInputImageLoading });
    },
    setSimpleInputImageError(simpleInputImageError: string) {
      return store.patch({ simpleInputImageError });
    },
  };
}

export { DEFAULT_PROMPT_WINDOW_STATE, normalizePromptList };
