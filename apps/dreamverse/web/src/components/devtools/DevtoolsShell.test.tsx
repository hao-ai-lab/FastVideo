import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import DevtoolsShell from './DevtoolsShell';

describe('DevtoolsShell', () => {
  it('reuses the release shell structure and keeps advanced controls in a drawer', () => {
    render(
      <DevtoolsShell
        storyPresets={[
          {
            id: 'preset_alpha',
            label: 'Preset Alpha',
            segment_prompts: ['intro', 'middle'],
          },
        ]}
        selectedPresetId="preset_alpha"
        editableMode={true}
        editableCanJoin={true}
        editableSegments={['intro', 'middle']}
        promptEvents={[
          {
            status: 'sent',
            source: 'manual',
            text: 'A dramatic reveal',
            latencyMs: 120,
          },
        ]}
      />,
    );

    expect(
      screen.getByRole('heading', { name: 'Preset-driven video continuation' }),
    ).toBeInTheDocument();
    expect(screen.getByText('Devtools Mode')).toBeInTheDocument();
    expect(screen.getByText('Your video will appear here')).toBeInTheDocument();
    expect(screen.getByLabelText('Story preset')).toBeInTheDocument();
    expect(screen.getByLabelText('Continuation prompt')).toBeDisabled();

    expect(screen.getByText('Advanced controls')).toBeInTheDocument();
    expect(screen.getByText('Rewrite inspector')).toBeInTheDocument();
    expect(screen.getByText('Editable preset builder')).toBeInTheDocument();
    expect(screen.getByText('System prompt editor')).toBeInTheDocument();
    expect(screen.getByText('Prompt activity')).toBeInTheDocument();

    expect(
      screen.queryByText('Real-Time 1080p Video Generation with FastLTX2 on a single GPU'),
    ).not.toBeInTheDocument();
    expect(screen.queryByText('Live Prompt Input')).not.toBeInTheDocument();
  });
});
