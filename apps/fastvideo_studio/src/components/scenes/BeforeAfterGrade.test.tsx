import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BeforeAfterGrade } from './BeforeAfterGrade';

// jsdom has no real video decoder, canvas 2D context, or `ImageData` constructor,
// so all three are stubbed: the video always "has" an 8x6 frame ready, the canvas
// records what was drawn/read/written instead of actually rasterizing anything,
// and ImageData is a plain data holder (its real job -- backing a canvas -- is
// already covered by the getContext stub).
if (typeof globalThis.ImageData === 'undefined') {
  class FakeImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
    constructor(data: Uint8ClampedArray, width: number, height: number) {
      this.data = data;
      this.width = width;
      this.height = height;
    }
  }
  // @ts-expect-error -- test-only polyfill for a browser global jsdom doesn't provide
  globalThis.ImageData = FakeImageData;
}

function stubVideoFrame() {
  Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { configurable: true, get: () => 4 });
  Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { configurable: true, get: () => 8 });
  Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { configurable: true, get: () => 6 });
}

function stubCanvasContext() {
  const data = new Uint8ClampedArray(8 * 6 * 4).fill(100);
  const putImageData = vi.fn();
  const drawImage = vi.fn();
  const getImageData = vi.fn(() => ({ width: 8, height: 6, data }));
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage,
    getImageData,
    putImageData,
  } as unknown as CanvasRenderingContext2D);
  return { drawImage, getImageData, putImageData, data };
}

describe('BeforeAfterGrade', () => {
  beforeEach(() => {
    stubVideoFrame();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows the "load a frame" hint until a video frame is captured', () => {
    const ref = { current: null };
    render(<BeforeAfterGrade videoRef={ref} brightness={0} contrast={1} saturation={1} />);
    expect(screen.getByText(/Play or seek the preview above/)).toBeInTheDocument();
  });

  it('captures a frame once the video has data, and draws it into both canvases', () => {
    const { getImageData, putImageData } = stubCanvasContext();
    const video = document.createElement('video');
    document.body.appendChild(video);
    const ref = { current: video };

    render(<BeforeAfterGrade videoRef={ref} brightness={0} contrast={1} saturation={1} />);
    fireEvent.loadedData(video);

    expect(getImageData).toHaveBeenCalled();
    expect(putImageData).toHaveBeenCalled();
    expect(screen.queryByText(/Play or seek the preview above/)).not.toBeInTheDocument();
    document.body.removeChild(video);
  });

  it('re-grades the "after" canvas when the color values change, without touching the video again', () => {
    const { putImageData } = stubCanvasContext();
    const video = document.createElement('video');
    document.body.appendChild(video);
    const ref = { current: video };

    const { rerender } = render(<BeforeAfterGrade videoRef={ref} brightness={0} contrast={1} saturation={1} />);
    fireEvent.loadedData(video);
    const callsAfterCapture = putImageData.mock.calls.length;

    rerender(<BeforeAfterGrade videoRef={ref} brightness={50} contrast={1} saturation={1} />);
    expect(putImageData.mock.calls.length).toBeGreaterThan(callsAfterCapture);

    // Neutral values were drawn as-is; the brightened redraw's pixel buffer must differ.
    const neutralData = putImageData.mock.calls[0][0].data as Uint8ClampedArray;
    const brightenedData = putImageData.mock.calls[putImageData.mock.calls.length - 1][0].data as Uint8ClampedArray;
    expect(brightenedData[0]).toBeGreaterThan(neutralData[0]);

    document.body.removeChild(video);
  });

  it('"Use current frame" re-captures the video at its current position', () => {
    const { drawImage } = stubCanvasContext();
    const video = document.createElement('video');
    document.body.appendChild(video);
    const ref = { current: video };

    render(<BeforeAfterGrade videoRef={ref} brightness={0} contrast={1} saturation={1} />);
    fireEvent.loadedData(video);
    const callsAfterAutoCapture = drawImage.mock.calls.length;

    fireEvent.click(screen.getByRole('button', { name: 'Use current frame' }));
    expect(drawImage.mock.calls.length).toBeGreaterThan(callsAfterAutoCapture);

    document.body.removeChild(video);
  });
});
