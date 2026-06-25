export function applySessionUiMessage<T extends object>(
  state: T,
  data: any,
): T {
  if (!data || typeof data !== 'object') {
    return state;
  }

  if (data.type === 'queue_status') {
    return {
      ...state,
      queuePosition: data.position,
    } as T;
  }

  if (data.type === 'auto_extension_updated') {
    return {
      ...state,
      autoExtensionEnabled: Boolean(data.enabled),
    } as T;
  }

  if (data.type === 'gpu_assigned') {
    return {
      ...state,
      gpuAssigned: true,
      queuePosition: 0,
      sessionTimeout: data.session_timeout,
      timeLeft: data.session_timeout,
    } as T;
  }

  if (data.type === 'session_timeout') {
    return {
      ...state,
      gpuAssigned: false,
      timeLeft: 0,
    } as T;
  }

  return state;
}
