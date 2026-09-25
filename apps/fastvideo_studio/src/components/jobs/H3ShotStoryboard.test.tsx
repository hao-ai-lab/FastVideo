import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { H3ShotStoryboard } from './H3ShotStoryboard';
import { CAMERA_MOVEMENTS, SHOT_TYPES, makeShot, newLine, type Shot, type ShotInit } from '@/lib/h3Shots';

function shot(init: ShotInit): Shot {
  return makeShot(init);
}

describe('H3ShotStoryboard', () => {
  it('lays shots out in order with type, movement, subjects, and dialogue', () => {
    const shots = [
      shot({ type: 'establishing', description: 'Wide rooftop.', subjectIds: ['<Subject 1>', '<Subject 2>'] }),
      shot({ type: 'close-up', movement: 'handheld', lines: [newLine({ speaker: 'S2', text: 'I had no choice.' }), newLine({ speaker: 'S1', text: 'Then go.' })], subjectIds: ['<Subject 2>'] }),
    ];
    render(<H3ShotStoryboard shots={shots} selectedId={shots[1].id} onSelect={() => {}} onAdd={() => {}} />);

    expect(screen.getByText('Establishing')).toBeInTheDocument();
    expect(screen.getByText('Close-up')).toBeInTheDocument();
    expect(screen.getByText('Handheld')).toBeInTheDocument();
    expect(screen.getByText(/S2: “I had no choice.” S1: “Then go.”/)).toBeInTheDocument();
    expect(screen.getAllByText('Subj 2')).toHaveLength(2);

    const panels = screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-current') || b.textContent?.match(/^\d/));
    expect(panels[0]).toHaveTextContent(/^1Establishing/);
    expect(screen.getByRole('button', { name: /Close-up/ })).toHaveAttribute('aria-current', 'true');
  });

  it('selects a shot on click and adds via the trailing tile', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onAdd = vi.fn();
    const shots = [shot({ description: 'First.' }), shot({ description: 'Second.' })];
    render(<H3ShotStoryboard shots={shots} selectedId={null} onSelect={onSelect} onAdd={onAdd} />);

    await user.click(screen.getByRole('button', { name: /Second\./ }));
    expect(onSelect).toHaveBeenCalledWith(shots[1].id);
    await user.click(screen.getByRole('button', { name: /Add shot/ }));
    expect(onAdd).toHaveBeenCalled();
  });

  it('renders every shot type and camera movement without crashing', () => {
    const shots = SHOT_TYPES.flatMap((t) =>
      CAMERA_MOVEMENTS.map((m) => shot({ type: t.id, movement: m.id, subjectIds: ['<Subject 1>'] })),
    );
    const { container } = render(<H3ShotStoryboard shots={shots} selectedId={null} onSelect={() => {}} onAdd={() => {}} />);
    expect(container.querySelectorAll('svg')).toHaveLength(SHOT_TYPES.length * CAMERA_MOVEMENTS.length);
  });
});
