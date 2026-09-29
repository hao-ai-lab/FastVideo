/**
 * Client-side mirror of trim.py's `_apply_grade`, so the "before/after" preview in
 * the Edit video dialog can show the effect of a color grade instantly, without a
 * server round trip. Same bounds, same order of operations (saturation, then
 * contrast, then brightness). Not pixel-exact with the server's numpy path --
 * `Uint8ClampedArray` rounds to the nearest integer on assignment where numpy
 * truncates -- but the difference is well under a visible unit per channel, and
 * this is only ever a preview; the server's render (via `POST .../trim`) is what
 * actually gets saved.
 */

export const BRIGHTNESS_BOUNDS = [-100, 100] as const;
export const CONTRAST_BOUNDS = [0, 3] as const;
export const SATURATION_BOUNDS = [0, 3] as const;

export const NEUTRAL_GRADE = { brightness: 0, contrast: 1, saturation: 1 } as const;

// Perceptual luma weights, matching trim.py's _LUMA_WEIGHTS.
const LUMA_R = 0.2126;
const LUMA_G = 0.7152;
const LUMA_B = 0.0722;

/** True for the values that leave every pixel unchanged -- mirrors trim.py's `graded` flag. */
export function isNeutralGrade(brightness: number, contrast: number, saturation: number): boolean {
  return brightness === NEUTRAL_GRADE.brightness && contrast === NEUTRAL_GRADE.contrast && saturation === NEUTRAL_GRADE.saturation;
}

/**
 * Adjusts an RGBA buffer (as in `ImageData.data`) in place; alpha is left alone.
 * Skipping the call entirely when `isNeutralGrade` is true (as callers should)
 * avoids the per-pixel work for the common no-op case.
 */
export function applyGradeInPlace(data: Uint8ClampedArray, brightness: number, contrast: number, saturation: number): void {
  for (let i = 0; i < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    if (saturation !== 1) {
      const luma = r * LUMA_R + g * LUMA_G + b * LUMA_B;
      r = luma + (r - luma) * saturation;
      g = luma + (g - luma) * saturation;
      b = luma + (b - luma) * saturation;
    }
    if (contrast !== 1) {
      r = (r - 128) * contrast + 128;
      g = (g - 128) * contrast + 128;
      b = (b - 128) * contrast + 128;
    }
    if (brightness !== 0) {
      r += brightness;
      g += brightness;
      b += brightness;
    }

    // Assigning into a Uint8ClampedArray clamps to [0, 255] and rounds on its own.
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
  }
}
