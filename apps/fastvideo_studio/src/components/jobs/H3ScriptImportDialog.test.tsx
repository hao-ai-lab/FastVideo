import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { H3ShotListEditor } from './H3ShotListEditor';

// Original test material -- not from any real screenplay.
const SCRIPT = [
  'CAPTAIN: Sit down. (EXHALES) We need to talk about the ledger.',
  '',
  'CLERK: Yes, sir.',
  '',
  'CAPTAIN: Three crates are missing and nobody on this dock saw a thing.',
].join('\n');

const SUBJECTS = '<Subject 1> is the captain.\n<Subject 2> is the clerk.';

function renderEditor(detailed = '', onApply = vi.fn()) {
  render(
    <H3ShotListEditor
      open
      subjectDefinitions={SUBJECTS}
      initialDetailedDescription={detailed}
      initialRetentionAnalysis=""
      onApply={onApply}
      onClose={() => {}}
    />,
  );
  return onApply;
}

async function importScript(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Import script…' }));
  const dialog = await screen.findByRole('dialog', { name: 'Import script' });
  await user.click(within(dialog).getByLabelText('Script'));
  await user.paste(SCRIPT);
  return dialog;
}

describe('script import', () => {
  it('maps speakers to subjects by default and previews the clip', async () => {
    const user = userEvent.setup();
    renderEditor();
    const dialog = await importScript(user);

    expect(within(dialog).getByLabelText('CAPTAIN subject')).toHaveValue('<Subject 1>');
    expect(within(dialog).getByLabelText('CLERK subject')).toHaveValue('<Subject 2>');
    expect(within(dialog).getByLabelText('CAPTAIN speaker tag')).toHaveValue('S1');
    expect(within(dialog).getByLabelText('CLERK speaker tag')).toHaveValue('S2');
    expect(within(dialog).getByRole('option', { name: /Clip 1 of \d+/ })).toBeInTheDocument();
    expect(within(dialog).getByText('Establishing')).toBeInTheDocument();
  });

  it('loads the chosen clip into the shot list and writes it on apply', async () => {
    const user = userEvent.setup();
    const onApply = renderEditor();
    const dialog = await importScript(user);

    await user.click(within(dialog).getByRole('button', { name: 'Load clip' }));

    const boxes = (await screen.findAllByPlaceholderText('What the speaker says')) as HTMLTextAreaElement[];
    expect(boxes[0].value).toBe('Sit down. <exhale> We need to talk about the ledger.');
    expect(boxes[1].value).toBe('Yes, sir.');

    await user.click(screen.getByRole('button', { name: 'Apply' }));
    const [detailed, retention] = onApply.mock.calls[0];
    expect(detailed).toContain('[Shot 1] Establishing shot.');
    expect(detailed).toContain('(S1) <d>[English] Sit down. <exhale> We need to talk about the ledger.</d> (S2) <d>[English] Yes, sir.</d>');
    expect(retention).toContain('<Subject 1> (appears in');
    expect(retention).toContain('<Subject 2> (appears in');
  });

  it('warns that loading replaces the shots already in the list', async () => {
    const user = userEvent.setup();
    renderEditor('[Shot 1] Wide shot. A dock.\n\n[Shot 2] Close-up shot. A face.');
    const dialog = await importScript(user);
    expect(within(dialog).getByText('Replaces the 2 shots currently in the list.')).toBeInTheDocument();
  });

  it('reloading the same clip after edits restores the imported text', async () => {
    const user = userEvent.setup();
    renderEditor();
    let dialog = await importScript(user);
    await user.click(within(dialog).getByRole('button', { name: 'Load clip' }));

    let boxes = (await screen.findAllByPlaceholderText('What the speaker says')) as HTMLTextAreaElement[];
    await user.type(boxes[1], ' EDITED');
    expect(boxes[1].value).toBe('Yes, sir. EDITED');

    await user.click(screen.getByRole('button', { name: 'Import script…' }));
    dialog = await screen.findByRole('dialog', { name: 'Import script' });
    // Loading a clip advances the picker to the next one; choose clip 1 again.
    await user.selectOptions(within(dialog).getByLabelText('Clip to load'), '0');
    await user.click(within(dialog).getByRole('button', { name: 'Load clip' }));

    boxes = (await screen.findAllByPlaceholderText('What the speaker says')) as HTMLTextAreaElement[];
    expect(boxes[1].value).toBe('Yes, sir.');
  });

  it('advances to the next clip after loading one', async () => {
    const user = userEvent.setup();
    renderEditor();
    let dialog = await importScript(user);
    const picker = within(dialog).getByLabelText('Clip to load') as HTMLSelectElement;
    expect(picker.options.length).toBeGreaterThan(1);
    await user.click(within(dialog).getByRole('button', { name: 'Load clip' }));

    await user.click(screen.getByRole('button', { name: 'Import script…' }));
    dialog = await screen.findByRole('dialog', { name: 'Import script' });
    expect((within(dialog).getByLabelText('Clip to load') as HTMLSelectElement).value).toBe('1');
  });

  it('explains when the text has no speaker turns', async () => {
    const user = userEvent.setup();
    renderEditor();
    await user.click(screen.getByRole('button', { name: 'Import script…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Import script' });
    await user.click(within(dialog).getByLabelText('Script'));
    await user.paste('just some prose');
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/No "NAME: line" turns found/);
    expect(within(dialog).getByRole('button', { name: 'Load clip' })).toBeDisabled();
  });
});
