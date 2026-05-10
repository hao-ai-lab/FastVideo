export function parseEditableQuery(
  search: string | null | undefined,
): boolean | null {
  const params = new URLSearchParams(search || '');
  const queryValue = params.get('editable');
  if (queryValue === 'true') return true;
  if (queryValue === 'false') return false;
  return null;
}

interface ResolveEditableModeOptions {
  isDev?: boolean;
  envValue?: string;
  search?: string;
}

export function resolveEditableMode({
  isDev = false,
  envValue = '',
  search = '',
}: ResolveEditableModeOptions = {}): boolean {
  const envEnabled = String(envValue || '').toLowerCase() === 'true';
  const baseMode = envEnabled;

  if (!isDev) {
    return baseMode;
  }

  const urlOverride = parseEditableQuery(search);
  if (urlOverride !== null) {
    return urlOverride;
  }

  return baseMode;
}
