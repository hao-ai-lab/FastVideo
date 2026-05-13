import { createManagedStore, type ManagedStore } from './createManagedStore';

export interface UiState {
  isMonitorRoute: boolean;
  demoMode: boolean;
  simpleMode: boolean;
  devtoolsMode: boolean;
  editableMode: boolean;
  appendingPromptWindow: boolean;
  appendPromptWindowStatus: string;
  appendPromptWindowError: string;
  promptConfigEditorOpen: boolean;
  promptConfigLoading: boolean;
  promptConfigSaving: boolean;
  promptConfigLoaded: boolean;
  promptConfigError: string;
  promptConfigStatus: string;
  nextSegmentPromptEditorOpen: boolean;
  autoExtensionPromptEditorOpen: boolean;
  rewriteWindowPromptEditorOpen: boolean;
  nextSegmentSystemPromptDraft: string;
  autoExtensionSystemPromptDraft: string;
  rewriteWindowSystemPromptDraft: string;
}

const DEFAULT_UI_STATE: UiState = {
  isMonitorRoute: false,
  demoMode: false,
  simpleMode: false,
  devtoolsMode: false,
  editableMode: false,
  appendingPromptWindow: false,
  appendPromptWindowStatus: '',
  appendPromptWindowError: '',
  promptConfigEditorOpen: false,
  promptConfigLoading: false,
  promptConfigSaving: false,
  promptConfigLoaded: false,
  promptConfigError: '',
  promptConfigStatus: '',
  nextSegmentPromptEditorOpen: false,
  autoExtensionPromptEditorOpen: false,
  rewriteWindowPromptEditorOpen: true,
  nextSegmentSystemPromptDraft: '',
  autoExtensionSystemPromptDraft: '',
  rewriteWindowSystemPromptDraft: '',
};

export type UiStore = ManagedStore<UiState> & {
  reset: (overrides?: Partial<UiState>) => UiState;
  resetSessionState: () => UiState;
};

export function createUiStore(
  initialState: Partial<UiState> = {},
): UiStore {
  const store = createManagedStore<UiState>({
    ...DEFAULT_UI_STATE,
    ...initialState,
  } as UiState);

  return {
    ...store,
    reset(overrides: Partial<UiState> = {}) {
      return store.set({
        ...DEFAULT_UI_STATE,
        ...initialState,
        ...overrides,
      } as UiState);
    },
    resetSessionState() {
      return store.patch({
        appendingPromptWindow: false,
        appendPromptWindowStatus: '',
        appendPromptWindowError: '',
        promptConfigEditorOpen: false,
        promptConfigLoading: false,
        promptConfigSaving: false,
        promptConfigLoaded: false,
        promptConfigError: '',
        promptConfigStatus: '',
        nextSegmentPromptEditorOpen: false,
        autoExtensionPromptEditorOpen: false,
        rewriteWindowPromptEditorOpen: true,
        nextSegmentSystemPromptDraft: '',
        autoExtensionSystemPromptDraft: '',
        rewriteWindowSystemPromptDraft: '',
      });
    },
  };
}

export { DEFAULT_UI_STATE };
