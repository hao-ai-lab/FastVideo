import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const css = readFileSync(join(process.cwd(), 'src/app/globals.css'), 'utf8');

function token(block: string, name: string): string {
  const match = block.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!match) throw new Error(`Missing --${name} token`);
  return match[1];
}

function luminance(hex: string): number {
  const channels = hex
    .slice(1)
    .match(/.{2}/g)!
    .map((channel) => parseInt(channel, 16) / 255)
    .map((channel) =>
      channel <= 0.04045
        ? channel / 12.92
        : ((channel + 0.055) / 1.055) ** 2.4,
    );
  return (
    0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
  );
}

function contrast(first: string, second: string): number {
  const firstLuminance = luminance(first);
  const secondLuminance = luminance(second);
  return (
    (Math.max(firstLuminance, secondLuminance) + 0.05) /
    (Math.min(firstLuminance, secondLuminance) + 0.05)
  );
}

describe('global focus styles', () => {
  it('keeps focus tokens above 3:1 against both page themes', () => {
    const light = css.match(/:root\s*{([\s\S]*?)\n}/)?.[1] ?? '';
    const dark = css.match(/\.dark\s*{([\s\S]*?)\n}/)?.[1] ?? '';

    expect(contrast(token(light, 'ring'), token(light, 'background'))).toBeGreaterThanOrEqual(
      3,
    );
    expect(contrast(token(dark, 'ring'), token(dark, 'background'))).toBeGreaterThanOrEqual(
      3,
    );
  });

  it('applies a non-animated three-pixel outline to focus-visible controls', () => {
    expect(css).toContain('):focus-visible {');
    expect(css).toContain('outline: 3px solid var(--ring) !important;');
    expect(css).toContain('outline-offset: 2px !important;');
  });
});
