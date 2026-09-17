import * as React from 'react';
import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import CurlCode from './CurlCode';
import { jsonApiRequest, requestToCurl } from '@/lib/apiRequest';
import { highlightCurl } from '@/lib/highlightCurl';

describe('CurlCode', () => {
  it.each([false, true])('preserves exact bytes and escapes HTML while highlighting cURL (paths=%s)', (withPaths) => {
    const command = requestToCurl('https://example.test/api', jsonApiRequest('POST', '/jobs', {
      model_id: 'wan/test',
      prompt: `A person's "cat"\n</code><img src=x onerror="alert(1)"> $HOME`,
      num_frames: 60,
      guidance_scale: 5.5,
      enabled: true,
      ...(withPaths ? { image_path: '/files/a\'b"$HOME.png', validation_dataset_file: '' } : {}),
    }));
    const { container } = render(<pre><CurlCode command={command} /></pre>);
    expect(container.textContent).toBe(command);
    expect(highlightCurl(command).map((token) => token.text).join('')).toBe(command);
    expect(container.querySelector('img')).toBeNull();
    for (const kind of ['keyword', 'flag', 'variable', 'key', 'string', 'number', 'literal']) {
      expect(container.querySelector(`[data-syntax="${kind}"]`)).not.toBeNull();
    }
    expect(container.querySelector('[data-syntax="key"]')?.textContent).toBe('"model_id"');
  });
});
