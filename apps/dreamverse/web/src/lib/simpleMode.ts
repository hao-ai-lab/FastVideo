export function parseSimpleQuery(
  search: string | null | undefined,
): boolean | null {
  const params = new URLSearchParams(search || '');
  if (!params.has('simple')) return null;

  const value = String(params.get('simple') || '').trim().toLowerCase();
  if (value === '' || value === '1' || value === 'true' || value === 'yes' || value === 'on') {
    return true;
  }
  if (value === '0' || value === 'false' || value === 'no' || value === 'off') {
    return false;
  }
  return null;
}

interface ResolveSimpleModeOptions {
  search?: string;
}

export function resolveSimpleMode({
  search = '',
}: ResolveSimpleModeOptions = {}): boolean {
  return parseSimpleQuery(search) === true;
}
