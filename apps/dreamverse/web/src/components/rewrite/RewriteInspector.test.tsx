import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import RewriteInspector from './RewriteInspector';

describe('RewriteInspector', () => {
  it('shows the current prompt window and latest rewrite details', () => {
    render(
      <RewriteInspector
        currentPromptWindowPrompts={['intro beat', 'dramatic reveal']}
        rewritingSeedPrompts={false}
        rewriteWindowMode={true}
        promptEvents={[
          {
            status: 'rewrite_raw_output',
            source: 'llm_rewrite',
            text: '{"segment_prompts":["intro beat","dramatic reveal"]}',
          },
          {
            status: 'rewrite_ready',
            source: 'llm_rewrite',
            model: 'gpt-oss-120b',
            latencyMs: 512,
            text: '[1] intro beat\n[2] dramatic reveal',
          },
          {
            status: 'rewrite_requested',
            source: 'user_rewrite',
            text: 'tighten the pacing',
          },
        ]}
      />,
    );

    expect(screen.getByRole('complementary', { name: 'Rewrite inspector' })).toBeInTheDocument();
    expect(screen.getByText('Current window snapshot')).toBeInTheDocument();
    expect(screen.getByText('Prompt 1')).toBeInTheDocument();
    expect(screen.getByText('intro beat')).toBeInTheDocument();
    expect(screen.getByText('tighten the pacing')).toBeInTheDocument();
    expect(screen.getByText('rewrite_ready')).toBeInTheDocument();
    expect(screen.getByText(/\[1\] intro beat/)).toBeInTheDocument();
    expect(screen.getByText('Raw model output')).toBeInTheDocument();
  });
});
