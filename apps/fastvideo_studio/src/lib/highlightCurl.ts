// SPDX-License-Identifier: Apache-2.0

export type CurlTokenKind = 'plain' | 'keyword' | 'flag' | 'variable' | 'string' | 'key' | 'number' | 'literal' | 'comment';
export interface CurlToken { kind: CurlTokenKind; text: string }

// This deliberately covers the generated shell/JSON examples, not arbitrary
// shell programs. Shell-escaped apostrophes stay inside their quoted word.
const SHELL_TOKENS = /#[^\n]*|'(?:[^']|'"'"')*'|"(?:\\.|[^"\\])*"|--?[a-zA-Z][\w-]*|\$(?:\{[\w]+\}|[\w]+)|\b[A-Z][A-Z_0-9]*(?==)|\b(?:export|curl|jq)\b/g;
const JSON_TOKENS = /"(?:\\.|'"'"'|[^"\\])*"|\$[A-Z][A-Z_0-9]*|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g;

function tokenize(text: string, pattern: RegExp, json: boolean): CurlToken[] {
  const tokens: CurlToken[] = [];
  let previous = 0;
  for (const match of text.matchAll(pattern)) {
    const start = match.index!;
    if (start > previous) tokens.push({ kind: 'plain', text: text.slice(previous, start) });
    const token = match[0];
    if (!json && token.startsWith("'{")) {
      tokens.push({ kind: 'plain', text: "'" }, ...tokenize(token.slice(1, -1), JSON_TOKENS, true), { kind: 'plain', text: "'" });
    } else {
      let kind: CurlTokenKind;
      if (token.startsWith('#')) kind = 'comment';
      else if (json && token.startsWith('"')) kind = /^\s*:/.test(text.slice(start + token.length)) ? 'key' : 'string';
      else if (token.startsWith('$') || /^[A-Z][A-Z_0-9]*$/.test(token) || /^"\$\{?[A-Z]/.test(token)) kind = 'variable';
      else if (token.startsWith('"') || token.startsWith("'")) kind = 'string';
      else if (json) kind = /^(true|false|null)$/.test(token) ? 'literal' : 'number';
      else kind = token.startsWith('-') ? 'flag' : 'keyword';
      tokens.push({ kind, text: token });
    }
    previous = start + token.length;
  }
  if (previous < text.length) tokens.push({ kind: 'plain', text: text.slice(previous) });
  return tokens;
}

/** Tokens always concatenate to the original bytes; rendering never parses HTML. */
export function highlightCurl(command: string): CurlToken[] {
  return tokenize(command, SHELL_TOKENS, false);
}
