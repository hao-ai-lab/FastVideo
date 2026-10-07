import * as React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { Button } from './button';
import { Input } from './input';
import { NativeSelect } from './native-select';
import { Slider } from './slider';
import { Switch } from './switch';

describe('shared control accessibility', () => {
  it('forwards refs and click handlers to the asChild button', async () => {
    const user = userEvent.setup();
    const ref = React.createRef<HTMLButtonElement>();
    const onClick = vi.fn();
    render(
      <Button asChild ref={ref} onClick={onClick}>
        <button type="button">Slotted action</button>
      </Button>,
    );

    const button = screen.getByRole('button', { name: 'Slotted action' });
    expect(screen.getAllByRole('button')).toHaveLength(1);
    expect(ref.current).toBe(button);
    await user.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(button).toHaveFocus();
  });

  it('keeps button, input, and select targets at least 44px tall', () => {
    render(
      <>
        <Button size="sm">Small action</Button>
        <Input aria-label="Text value" />
        <NativeSelect aria-label="Choice" defaultValue="one">
          <option value="one">One</option>
        </NativeSelect>
      </>,
    );

    expect(screen.getByRole('button', { name: 'Small action' })).toHaveClass(
      'h-11',
    );
    expect(screen.getByRole('textbox', { name: 'Text value' })).toHaveClass(
      'h-11',
    );
    expect(screen.getByRole('combobox', { name: 'Choice' })).toHaveClass(
      'h-11',
    );
  });

  it('uses 44px switch and slider interaction surfaces', () => {
    render(
      <>
        <Switch aria-label="Enabled" />
        <Slider aria-label="Amount" defaultValue={[50]} />
      </>,
    );

    expect(screen.getByRole('switch', { name: 'Enabled' })).toHaveClass(
      'size-11',
    );
    expect(screen.getByRole('slider', { name: 'Amount' })).toHaveClass(
      'size-11',
    );
  });
});
