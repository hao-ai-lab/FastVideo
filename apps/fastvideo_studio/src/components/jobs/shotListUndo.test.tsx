import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { H3ShotListEditor } from './H3ShotListEditor';

const TWO_SHOTS = '[Shot 1] Wide shot. A dock.\n[Shot 2] At 00:02.000, the shot cuts to a close-up shot. A face.';
const WITH_LINE = '[Shot 1] Wide shot. (S1) <d>[English] Hi.</d>';

type User = ReturnType<typeof userEvent.setup>;

function setup(detailed: string, extra: { open?: boolean } = {}) {
  const onApply = vi.fn();
  const ui = (open: boolean) => (
    <H3ShotListEditor
      open={open}
      subjectDefinitions="<Subject 1> is a man."
      initialDetailedDescription={detailed}
      initialRetentionAnalysis=""
      onApply={onApply}
      onClose={() => {}}
    />
  );
  const view = render(ui(extra.open ?? true));
  return { onApply, view, ui };
}

const undoButton = () => screen.getByRole('button', { name: /Undo/ });
const redoButton = () => screen.getByRole('button', { name: /Redo/ });
const applied = (onApply: ReturnType<typeof vi.fn>) => onApply.mock.calls.at(-1)?.[0] as string;
const apply = (user: User) => user.click(screen.getByRole('button', { name: 'Apply' }));
const lineBox = () => screen.getByPlaceholderText('What the speaker says') as HTMLTextAreaElement;
const ctrl = (key: string) => `{Control>}${key}{/Control}`;

describe('shot list undo/redo', () => {
  it('starts with nothing to undo or redo', () => {
    setup(TWO_SHOTS);
    expect(undoButton()).toBeDisabled();
    expect(redoButton()).toBeDisabled();
  });

  it('undoes and redoes moving a shot', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);

    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    expect(undoButton()).toBeEnabled();

    await user.click(undoButton());
    expect(redoButton()).toBeEnabled();
    await apply(user);
    expect(applied(onApply)).toBe(TWO_SHOTS);
  });

  it('redo reapplies what was undone', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    await user.click(undoButton());
    await user.click(redoButton());
    await apply(user);
    expect(applied(onApply).startsWith('[Shot 1] Close-up shot. A face.')).toBe(true);
  });

  it('undoes removing a shot, dialogue and all', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(`${WITH_LINE}\n[Shot 2] At 00:02.500, the shot cuts to a wide shot. B.`);
    await user.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
    expect(screen.queryByPlaceholderText('What the speaker says')).not.toBeInTheDocument();

    await user.click(undoButton());
    expect(lineBox().value).toBe('Hi.');
    await apply(user);
    expect(applied(onApply)).toContain('(S1) <d>[English] Hi.</d>');
  });

  it('undoes an add, a type change, a subject toggle, and a cut time in turn', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);

    await user.click(screen.getAllByRole('button', { name: /Add shot/ })[0]);
    await user.selectOptions(screen.getAllByLabelText('Shot type')[0], 'pov');
    await user.click(screen.getAllByRole('button', { name: '<Subject 1>' })[0]);
    const cut = screen.getAllByPlaceholderText(/\d\d:\d\d\.\d{3}/)[0] as HTMLInputElement;
    await user.type(cut, '5{Enter}');

    // four discrete steps, undone one at a time
    for (let i = 0; i < 4; i += 1) await user.click(undoButton());
    expect(undoButton()).toBeDisabled();
    await apply(user);
    expect(applied(onApply)).toBe(TWO_SHOTS);
  });

  it('committing a cut time that has not changed is not a step', async () => {
    const user = userEvent.setup();
    setup(TWO_SHOTS);
    const cut = screen.getAllByPlaceholderText(/\d\d:\d\d\.\d{3}/)[0] as HTMLInputElement;
    await user.type(cut, '5{Enter}');
    await user.click(undoButton()); // the click blurs the box, which re-commits 00:05.000
    expect(undoButton()).toBeDisabled();
  });

  it('a burst of typing is one step', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(WITH_LINE);
    await user.type(lineBox(), ' there my friend');
    expect(lineBox().value).toBe('Hi. there my friend');

    await user.click(undoButton());
    expect(lineBox().value).toBe('Hi.');
    expect(undoButton()).toBeDisabled();
    await apply(user);
    expect(applied(onApply)).toContain('<d>[English] Hi.</d>');
  });

  it('a tag inserted from the palette is its own step, separate from typing', async () => {
    const user = userEvent.setup();
    setup(WITH_LINE);
    await user.type(lineBox(), ' there');
    lineBox().setSelectionRange(lineBox().value.length, lineBox().value.length);
    await user.click(screen.getByRole('button', { name: /<pause>/ }));
    expect(lineBox().value).toBe('Hi. there <pause>');

    await user.click(undoButton());
    expect(lineBox().value).toBe('Hi. there');
    await user.click(undoButton());
    expect(lineBox().value).toBe('Hi.');
  });

  it('undoes edits to the description and the scene notes', async () => {
    const user = userEvent.setup();
    const { onApply } = setup('[Shot 1] Wide shot. A dock.');
    await user.type(screen.getByPlaceholderText(/Live-action, cinematic/), 'Noir.');
    await user.type(screen.getByPlaceholderText("What's happening in this shot"), ' At dawn.');

    await user.click(undoButton());
    await user.click(undoButton());
    expect(undoButton()).toBeDisabled();
    await apply(user);
    expect(applied(onApply)).toBe('[Shot 1] Wide shot. A dock.');
  });

  it('typing in different boxes makes separate steps', async () => {
    const user = userEvent.setup();
    setup('[Shot 1] Wide shot. A dock.');
    await user.type(screen.getByPlaceholderText(/Live-action, cinematic/), 'Noir.');
    await user.type(screen.getByPlaceholderText("What's happening in this shot"), ' At dawn.');

    await user.click(undoButton());
    expect((screen.getByPlaceholderText("What's happening in this shot") as HTMLTextAreaElement).value).toBe('A dock.');
    expect((screen.getByPlaceholderText(/Live-action, cinematic/) as HTMLTextAreaElement).value).toBe('Noir.');
  });

  it('a new edit clears what could be redone', async () => {
    const user = userEvent.setup();
    setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    await user.click(undoButton());
    expect(redoButton()).toBeEnabled();
    await user.click(screen.getAllByRole('button', { name: /Add shot/ })[0]);
    expect(redoButton()).toBeDisabled();
  });
});

describe('shot list undo/redo shortcuts', () => {
  it('Ctrl+Z from inside a textbox undoes the whole list, not just that box', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]); // move a shot
    // focus a different textbox, then undo from there
    const box = screen.getAllByPlaceholderText("What's happening in this shot")[0] as HTMLTextAreaElement;
    await user.click(box);
    await user.keyboard(ctrl('z'));
    expect(undoButton()).toBeDisabled();
    await apply(user);
    expect(applied(onApply)).toBe(TWO_SHOTS);
  });

  it('redoes with Ctrl+Shift+Z and with Ctrl+Y', async () => {
    const user = userEvent.setup();
    setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    const box = screen.getAllByPlaceholderText("What's happening in this shot")[0];
    await user.click(box);

    await user.keyboard(ctrl('z'));
    expect(redoButton()).toBeEnabled();
    await user.keyboard('{Control>}{Shift>}z{/Shift}{/Control}');
    expect(redoButton()).toBeDisabled();

    await user.keyboard(ctrl('z'));
    await user.keyboard(ctrl('y'));
    expect(redoButton()).toBeDisabled();
    expect(undoButton()).toBeEnabled();
  });

  it('Cmd+Z works too', async () => {
    const user = userEvent.setup();
    setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    await user.click(screen.getAllByPlaceholderText("What's happening in this shot")[0]);
    await user.keyboard('{Meta>}z{/Meta}');
    expect(undoButton()).toBeDisabled();
  });

  it('leaves the script import box to its own undo', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    expect(undoButton()).toBeEnabled();

    await user.click(screen.getByRole('button', { name: 'Import script…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import script' });
    const box = within(dialog).getByLabelText('Script');
    await user.click(box);
    await user.keyboard(ctrl('z'));
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    // the shot list's history was untouched: the move is still there to undo
    await user.click(await screen.findByRole('button', { name: /Undo/ }));
    await apply(user);
    expect(applied(onApply)).toBe(TWO_SHOTS);
  });
});

describe('shot list history across imports and reopening', () => {
  it('loading a script clip is one undoable step', async () => {
    const user = userEvent.setup();
    const { onApply } = setup(TWO_SHOTS);
    await user.click(screen.getByRole('button', { name: 'Import script…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import script' });
    await user.click(within(dialog).getByLabelText('Script'));
    await user.paste('CAPTAIN: Sit down and hear me out.\n\nCLERK: Yes, sir.');
    await user.click(within(dialog).getByRole('button', { name: 'Load clip' }));
    await screen.findAllByPlaceholderText('What the speaker says');

    await user.click(undoButton());
    expect(screen.queryByPlaceholderText('What the speaker says')).not.toBeInTheDocument();
    await apply(user);
    expect(applied(onApply)).toBe(TWO_SHOTS);
  });

  it('opening the editor again starts a fresh history', async () => {
    const user = userEvent.setup();
    const { view, ui } = setup(TWO_SHOTS);
    await user.click(screen.getAllByRole('button', { name: '↑' })[1]);
    expect(undoButton()).toBeEnabled();

    view.rerender(ui(false));
    view.rerender(ui(true));
    expect(await screen.findByRole('button', { name: /Undo/ })).toBeDisabled();
  });
});
