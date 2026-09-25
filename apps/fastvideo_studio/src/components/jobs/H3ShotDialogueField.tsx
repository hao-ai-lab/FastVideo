'use client';

import * as React from 'react';

import { Textarea } from '@/components/ui/textarea';
import { H3DialoguePreview } from '@/components/jobs/H3DialoguePreview';
import { applyH3Preset, applyH3Tag, type H3Preset, type H3Tag } from '@/lib/h3Tags';

/** What the shared tag palette drives: the last-focused shot dialogue field. */
export interface H3ShotDialogueHandle {
  insertTag: (tag: H3Tag) => void;
  applyPreset: (preset: H3Preset) => void;
}

export interface H3ShotDialogueFieldProps {
  /** The line's text. The parent owns it (and the undo history), so this is fully controlled. */
  value: string;
  /** Typing. */
  onChange: (value: string) => void;
  /** A tag or preset inserted from the palette -- a discrete edit, unlike typing. */
  onInsert: (value: string) => void;
  onFocus: () => void;
  active?: boolean;
  label?: string;
}

/**
 * One spoken line: a textarea plus, once it contains tags, a styled preview.
 * The shared palette inserts at the cursor through the ref. Undo/redo isn't
 * here -- the shot list editor keeps one history for everything in it.
 */
export const H3ShotDialogueField = React.forwardRef<H3ShotDialogueHandle, H3ShotDialogueFieldProps>(
  function H3ShotDialogueField({ value, onChange, onInsert, onFocus, active, label = 'Dialogue' }, ref) {
    const taRef = React.useRef<HTMLTextAreaElement>(null);
    // Selection to restore once the inserted text has rendered.
    const pending = React.useRef<[number, number] | null>(null);

    React.useEffect(() => {
      if (pending.current && taRef.current) {
        const [s, e] = pending.current;
        taRef.current.focus();
        taRef.current.setSelectionRange(s, e);
        pending.current = null;
      }
    });

    function selection(): [number, number] {
      const ta = taRef.current;
      return [ta?.selectionStart ?? value.length, ta?.selectionEnd ?? value.length];
    }

    React.useImperativeHandle(ref, () => ({
      insertTag(tag) {
        const [s, e] = selection();
        const result = applyH3Tag(value, s, e, tag);
        pending.current = [result.start, result.end];
        onInsert(result.value);
      },
      applyPreset(preset) {
        const [s, e] = selection();
        const result = applyH3Preset(value, s, e, preset);
        pending.current = [result.start, result.end];
        onInsert(result.value);
      },
    }));

    return (
      <div className="flex flex-col gap-1">
        <div className="text-xs text-muted-foreground">
          {label}
          {active && <span className="ml-2 text-accent-blue">tags insert here</span>}
        </div>
        <Textarea
          ref={taRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onFocus={onFocus}
          rows={2}
          spellCheck={false}
          placeholder="What the speaker says"
          className="font-mono text-sm"
        />
        {value.includes('<') && (
          <H3DialoguePreview
            text={value}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm leading-7 text-foreground"
          />
        )}
      </div>
    );
  },
);
