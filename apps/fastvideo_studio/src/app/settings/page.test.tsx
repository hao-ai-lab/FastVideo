import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { HeaderActionsProvider } from '@/components/HeaderActionsContext';
import { resetToDefaults, updateOption } from '@/stores/defaultOptions';

import SettingsPage from './page';

vi.mock('@/lib/api', () => ({
  getModels: vi.fn().mockResolvedValue([]),
  getSettings: vi.fn().mockResolvedValue({}),
  updateSettings: vi.fn().mockResolvedValue({}),
}));

// Keep the real store (so `useStore` resolves) but spy on the mutators.
vi.mock('@/stores/defaultOptions', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@/stores/defaultOptions')>();
  return {
    ...actual,
    updateOption: vi.fn(),
    resetToDefaults: vi.fn(),
  };
});

function renderPage() {
  return render(
    <HeaderActionsProvider>
      <SettingsPage />
    </HeaderActionsProvider>,
  );
}

describe('Settings page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calls updateOption when a toggle changes', () => {
    renderPage();
    fireEvent.click(
      screen.getByRole('switch', { name: 'Auto Start Job on Create' }),
    );
    expect(updateOption).toHaveBeenCalledWith('autoStartJob', true);
  });

  it('calls updateOption when a slider changes', () => {
    renderPage();
    // First slider in DOM order is "Frames".
    const slider = screen.getAllByRole('slider')[0];
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(updateOption).toHaveBeenCalledWith('numFrames', expect.any(Number));
  });

  it('calls resetToDefaults when Reset to Defaults is clicked', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Reset to Defaults' }));
    expect(resetToDefaults).toHaveBeenCalledTimes(1);
  });
});
