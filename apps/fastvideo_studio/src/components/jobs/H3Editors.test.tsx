import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { H3DialogueEditor } from './H3DialogueEditor';
import { H3ShotListEditor } from './H3ShotListEditor';

const SUBJECTS = '<Subject 1> is the courier.\n<Subject 2> is the rival.';
const DETAILED = [
  '[Shot 1] Wide shot. A rooftop at night.',
  '[Shot 2] Close-up shot, handheld. He turns. (S2) <d>[English] I had no choice.</d>',
].join('\n\n');

describe('H3DialogueEditor (shared palette + undo)', () => {
  it('inserts a tag at the cursor, undoes it, and applies the edited text', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    render(<H3DialogueEditor open initialValue="Hello there" onApply={onApply} onClose={() => {}} />);

    const textarea = screen.getByDisplayValue('Hello there') as HTMLTextAreaElement;
    textarea.focus();
    textarea.setSelectionRange(5, 5);
    await user.click(screen.getByRole('button', { name: /<pause>/ }));
    expect(textarea.value).toBe('Hello <pause> there');

    await user.click(screen.getByRole('button', { name: /Undo/ }));
    expect(textarea.value).toBe('Hello there');

    await user.click(screen.getByRole('button', { name: 'Nervous' }));
    expect(textarea.value).toContain('<uh> <stutter>');

    await user.click(screen.getByRole('button', { name: 'Apply to field' }));
    expect(onApply).toHaveBeenCalledWith(expect.stringContaining('<uh> <stutter>'));
  });
});

describe('H3ShotListEditor dialogue tooling', () => {
  function renderEditor(onApply = vi.fn()) {
    render(
      <H3ShotListEditor
        open
        subjectDefinitions={SUBJECTS}
        initialDetailedDescription={DETAILED}
        initialRetentionAnalysis=""
        onApply={onApply}
        onClose={() => {}}
      />,
    );
    return onApply;
  }

  it('disables the palette until a shot dialogue is focused', () => {
    renderEditor();
    expect(screen.getByText(/Click into a shot's dialogue/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /<pause>/ })).toBeDisabled();
  });

  it('inserts a palette tag into the focused shot dialogue and writes it on apply', async () => {
    const user = userEvent.setup();
    const onApply = renderEditor();

    const dialogueBoxes = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    expect(dialogueBoxes).toHaveLength(1);
    expect(dialogueBoxes[0].value).toBe('I had no choice.');

    await user.click(dialogueBoxes[0]);
    dialogueBoxes[0].setSelectionRange(1, 1);
    expect(screen.getByText('Tags insert into Shot 2 dialogue')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /<pause>/ }));
    expect(dialogueBoxes[0].value).toBe('I <pause> had no choice.');

    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0]).toContain('<d>[English] I <pause> had no choice.</d>');
  });

  it('adds and removes dialogue lines, writing every line on apply', async () => {
    const user = userEvent.setup();
    const onApply = renderEditor();

    await user.click(screen.getAllByRole('button', { name: '+ Add line' })[1]);
    const boxes = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    await user.type(boxes[1], 'Then go.');
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toContain(
      '(S2) <d>[English] I had no choice.</d> (S2) <d>[English] Then go.</d>',
    );
  });

  it('shows the styled preview once a shot dialogue contains tags', async () => {
    const user = userEvent.setup();
    renderEditor();
    const boxes = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    await user.click(boxes[0]);
    await user.click(screen.getByRole('button', { name: /<long pause>/ }));

    const shot2 = boxes[0].closest('[id^="shot-card-"]') as HTMLElement;
    expect(within(shot2).getByText('long pause')).toBeInTheDocument();
  });
});

describe('H3ShotListEditor ordered body', () => {
  function renderWith(detailed: string) {
    const onApply = vi.fn();
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

  it('shows hand-written dialogue as an editable line between its prose and applies the text unchanged', async () => {
    const user = userEvent.setup();
    const text = "[Shot 1] He (S1) says: <d>[English] I think you're sorry.</d> <cutoff>";
    const onApply = renderWith(text);

    expect(screen.getByText(/keeps its original wording/)).toBeInTheDocument();
    const lines = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    expect(lines).toHaveLength(1);
    expect(lines[0].value).toBe("I think you're sorry.");
    const prose = screen.getAllByPlaceholderText("What's happening in this shot") as HTMLTextAreaElement[];
    expect(prose.map((p) => p.value)).toEqual(['He (S1) says: ', ' <cutoff>']);

    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe(text);
  });

  it('editing a hand-written line keeps the tag after it where it was', async () => {
    const user = userEvent.setup();
    const onApply = renderWith("[Shot 1] He (S1) says: <d>[English] Sorry.</d> <cutoff>");
    const [line] = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    await user.type(line, ' Really.');
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe("[Shot 1] He (S1) says: <d>[English] Sorry. Really.</d> <cutoff>");
  });

  it('reorders dialogue lines with the move buttons', async () => {
    const user = userEvent.setup();
    const onApply = renderWith('[Shot 1] Wide shot. (S1) <d>[English] One.</d> (S2) <d>[English] Two.</d>');
    await user.click(screen.getByRole('button', { name: 'Move line 2 up' }));
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe(
      '[Shot 1] Wide shot. (S2) <d>[English] Two.</d> (S1) <d>[English] One.</d>',
    );
  });

  it('adds a text block after the existing content', async () => {
    const user = userEvent.setup();
    const onApply = renderWith('[Shot 1] Wide shot. A dock.');
    await user.click(screen.getByRole('button', { name: '+ Add text' }));
    const prose = screen.getAllByPlaceholderText("What's happening in this shot") as HTMLTextAreaElement[];
    expect(prose).toHaveLength(2);
    await user.type(prose[1], 'Rain falls.');
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe('[Shot 1] Wide shot. A dock. Rain falls.');
  });

  it('removes a text block', async () => {
    const user = userEvent.setup();
    const onApply = renderWith('[Shot 1] Wide shot. A dock. (S1) <d>[English] Hi.</d>');
    await user.click(screen.getByRole('button', { name: 'Remove text 1' }));
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe('[Shot 1] Wide shot. (S1) <d>[English] Hi.</d>');
  });

  it('adds a new line at the end of the shot, after any closing prose', async () => {
    const user = userEvent.setup();
    const onApply = renderWith('[Shot 1] Wide shot. He speaks. (S1) <d>[English] First.</d> He pauses.');
    await user.click(screen.getByRole('button', { name: '+ Add line' }));
    const lines = screen.getAllByPlaceholderText('What the speaker says') as HTMLTextAreaElement[];
    await user.type(lines[1], 'Later.');
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(onApply.mock.calls[0][0]).toBe(
      '[Shot 1] Wide shot. He speaks. (S1) <d>[English] First.</d> He pauses. (S1) <d>[English] Later.</d>',
    );
  });
});
