import * as React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { NumberRow, OptionSection, SliderRow } from './form-rows';

function renderSlider(overrides: Partial<React.ComponentProps<typeof SliderRow>> = {}) {
  const onChange = vi.fn();
  const props = {
    id: 'frames', label: 'Frames', min: 1, max: 500, step: 1,
    value: 81, onChange, ...overrides,
  };
  const result = render(<SliderRow {...props} />);
  return { ...result, props, onChange, input: screen.getByRole('spinbutton') };
}

describe('precise numeric controls', () => {
  it('allows typing an exact frame count and commits once on blur', async () => {
    const user = userEvent.setup();
    const { input, onChange } = renderSlider();
    await user.clear(input);
    await user.type(input, '60');
    expect(onChange).not.toHaveBeenCalled();
    await user.tab();
    expect(onChange).toHaveBeenCalledExactlyOnceWith(60);
  });

  it('restores an empty field without committing NaN or zero', async () => {
    const user = userEvent.setup();
    const { input, onChange } = renderSlider();
    await user.clear(input);
    await user.tab();
    expect(input).toHaveValue(81);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('clamps out-of-range values and snaps dimensions to the existing step', () => {
    const { input, onChange } = renderSlider({ min: 64, max: 1080, step: 16, value: 480 });
    fireEvent.change(input, { target: { value: '600' } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith(608);
    fireEvent.change(input, { target: { value: '2000' } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith(1072);
    fireEvent.change(input, { target: { value: '-1' } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenLastCalledWith(64);
  });

  it('avoids floating-point noise when committing a fractional step', () => {
    const { input, onChange } = renderSlider({ min: 0, max: 1, step: 0.05, value: 0 });
    fireEvent.change(input, { target: { value: '0.3' } });
    fireEvent.blur(input);
    expect(onChange).toHaveBeenCalledExactlyOnceWith(0.3);
  });

  it('commits Enter without submitting and lets Escape discard edits', async () => {
    const user = userEvent.setup();
    const onCommit = vi.fn();
    const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault());
    render(
      <form onSubmit={onSubmit}>
        <NumberRow id="seed" label="Seed" value={1024} onChange={onCommit} min={0} />
      </form>,
    );
    const input = screen.getByRole('spinbutton');
    await user.clear(input);
    await user.type(input, '42{Enter}');
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(42);
    expect(onSubmit).not.toHaveBeenCalled();
    await user.clear(input);
    await user.type(input, '99{Escape}');
    expect(input).toHaveValue(1024);
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  it('synchronizes external resets and preserves slider keyboard commits', () => {
    const { input, onChange, props, rerender } = renderSlider();
    fireEvent.change(input, { target: { value: '123' } });
    rerender(<SliderRow {...props} value={30} />);
    expect(input).toHaveValue(30);
    fireEvent.keyDown(screen.getByRole('slider'), { key: 'ArrowRight' });
    expect(onChange).toHaveBeenCalledExactlyOnceWith(31);
  });

  it('keeps numeric steppers accessible and disables both controls', () => {
    const { input } = renderSlider({ disabled: true });
    expect(input).toHaveAccessibleName('Frames value');
    expect(input).toHaveAttribute('min', '1');
    expect(input).toHaveAttribute('max', '500');
    expect(input).toHaveAttribute('step', '1');
    expect(input).toBeDisabled();
    expect(screen.getByRole('slider')).toHaveAttribute('data-disabled');
  });

  it('keeps automatic parallelism readable without hiding the numeric value', () => {
    const { input } = renderSlider({ label: 'TP Size', min: -1, max: 8, value: -1, format: (v) => v === -1 ? 'Auto' : String(v) });
    expect(input).toHaveValue(-1);
    expect(screen.getByText('Auto')).toBeInTheDocument();
  });
});

describe('option sections', () => {
  it('shows common options initially and keeps advanced options collapsed', () => {
    render(
      <>
        <OptionSection title="Output" description="Frames and resolution" defaultOpen>
          Common controls
        </OptionSection>
        <OptionSection title="Memory" description="CPU offloading">
          Advanced controls
        </OptionSection>
      </>,
    );
    expect(screen.getByText('Output').closest('details')).toHaveAttribute('open');
    expect(screen.getByText('Memory').closest('details')).not.toHaveAttribute('open');
  });
});
