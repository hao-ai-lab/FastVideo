export function parseDevtoolsQuery(
  search: string | null | undefined,
): boolean | null {
  const params = new URLSearchParams(search || '');
  if (!params.has('devtools')) {
    return null;
  }

  const rawValue = String(params.get('devtools') || '').trim().toLowerCase();
  if (
    rawValue === ''
    || rawValue === '1'
    || rawValue === 'true'
    || rawValue === 'yes'
    || rawValue === 'on'
  ) {
    return true;
  }
  if (
    rawValue === '0'
    || rawValue === 'false'
    || rawValue === 'no'
    || rawValue === 'off'
  ) {
    return false;
  }
  return null;
}

interface ResolveDevtoolsModeOptions {
  buildEnabled?: boolean;
  search?: string;
}

export function resolveDevtoolsMode({
  buildEnabled = false,
  search = '',
}: ResolveDevtoolsModeOptions = {}): boolean {
  if (!buildEnabled) {
    return false;
  }

  return parseDevtoolsQuery(search) === true;
}
