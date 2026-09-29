import { describe, expect, it } from 'vitest';

import { applyGradeInPlace, isNeutralGrade, NEUTRAL_GRADE } from './grading';

function solidRgba(rgb: [number, number, number], count = 4, alpha = 255): Uint8ClampedArray {
  const data = new Uint8ClampedArray(count * 4);
  for (let i = 0; i < count; i++) {
    data[i * 4] = rgb[0];
    data[i * 4 + 1] = rgb[1];
    data[i * 4 + 2] = rgb[2];
    data[i * 4 + 3] = alpha;
  }
  return data;
}

function meanRgb(data: Uint8ClampedArray): [number, number, number] {
  let r = 0, g = 0, b = 0;
  const n = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
  }
  return [r / n, g / n, b / n];
}

describe('isNeutralGrade', () => {
  it('is true only for brightness 0, contrast 1, saturation 1', () => {
    expect(isNeutralGrade(0, 1, 1)).toBe(true);
    expect(isNeutralGrade(NEUTRAL_GRADE.brightness, NEUTRAL_GRADE.contrast, NEUTRAL_GRADE.saturation)).toBe(true);
    expect(isNeutralGrade(1, 1, 1)).toBe(false);
    expect(isNeutralGrade(0, 1.1, 1)).toBe(false);
    expect(isNeutralGrade(0, 1, 0.9)).toBe(false);
  });
});

describe('applyGradeInPlace', () => {
  it('leaves neutral values as a no-op', () => {
    const data = solidRgba([200, 100, 50]);
    const before = Array.from(data);
    applyGradeInPlace(data, 0, 1, 1);
    expect(Array.from(data)).toEqual(before);
  });

  it('leaves alpha untouched', () => {
    const data = solidRgba([200, 100, 50], 1, 137);
    applyGradeInPlace(data, 50, 1.5, 0.5);
    expect(data[3]).toBe(137);
  });

  it('positive brightness lightens every channel', () => {
    const data = solidRgba([200, 100, 50]);
    const [r0, g0, b0] = meanRgb(data);
    applyGradeInPlace(data, 50, 1, 1);
    const [r1, g1, b1] = meanRgb(data);
    expect(r1).toBeGreaterThan(r0);
    expect(g1).toBeGreaterThan(g0);
    expect(b1).toBeGreaterThan(b0);
  });

  it('negative brightness darkens', () => {
    const data = solidRgba([200, 100, 50]);
    const [r0, g0, b0] = meanRgb(data);
    applyGradeInPlace(data, -50, 1, 1);
    const [r1, g1, b1] = meanRgb(data);
    expect(r1).toBeLessThan(r0);
    expect(g1).toBeLessThan(g0);
    expect(b1).toBeLessThan(b0);
  });

  it('clamps brightness at the top of the range instead of wrapping', () => {
    const data = solidRgba([240, 240, 240]);
    applyGradeInPlace(data, 100, 1, 1);
    expect(Array.from(data.subarray(0, 3))).toEqual([255, 255, 255]);
  });

  it('clamps brightness at the bottom of the range instead of wrapping', () => {
    const data = solidRgba([10, 10, 10]);
    applyGradeInPlace(data, -100, 1, 1);
    expect(Array.from(data.subarray(0, 3))).toEqual([0, 0, 0]);
  });

  it('zero saturation makes every channel converge on the luma', () => {
    const data = solidRgba([200, 100, 50]);
    applyGradeInPlace(data, 0, 1, 0);
    const [r, g, b] = meanRgb(data);
    expect(Math.abs(r - g)).toBeLessThan(2);
    expect(Math.abs(g - b)).toBeLessThan(2);
  });

  it('saturation above one widens the channel spread', () => {
    const base = solidRgba([200, 100, 50]);
    const [baseR, , baseB] = meanRgb(base);
    const data = solidRgba([200, 100, 50]);
    applyGradeInPlace(data, 0, 1, 2);
    const [r, , b] = meanRgb(data);
    expect(r - b).toBeGreaterThan(baseR - baseB);
  });

  it('high contrast pushes values away from mid-gray, in both directions', () => {
    const base = solidRgba([200, 100, 50]);
    const [baseR, baseG, baseB] = meanRgb(base);
    const data = solidRgba([200, 100, 50]);
    applyGradeInPlace(data, 0, 2, 1);
    const [r, g, b] = meanRgb(data);
    expect(r).toBeGreaterThan(baseR); // 200 sits above 128, pushed further up
    expect(g).toBeLessThan(baseG); // 100 sits below 128, pushed further down
    expect(b).toBeLessThan(baseB); // 50 sits below 128, pushed further down
  });
});
