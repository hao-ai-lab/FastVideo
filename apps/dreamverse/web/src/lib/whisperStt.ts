const OPENAI_TRANSCRIPTION_URL =
  'https://api.openai.com/v1/audio/transcriptions';

export interface WhisperSttSession {
  readonly active: boolean;
  stop: () => Promise<string | undefined>;
}

interface StartWhisperSttParams {
  apiKey: string;
  onError?: (error: Error) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export async function startWhisperStt({
  apiKey,
  onError = () => {},
  onOpen = () => {},
  onClose = () => {},
}: StartWhisperSttParams): Promise<WhisperSttSession> {
  if (!apiKey) {
    throw new Error('OpenAI API key is required');
  }

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  } catch (err: unknown) {
    const msg =
      err instanceof Error ? err.message : 'unknown error';
    throw new Error(`Microphone access denied: ${msg}`);
  }

  const chunks: Blob[] = [];
  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: getSupportedMimeType(),
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.start(250);
  onOpen();

  let active = true;

  const session: WhisperSttSession = {
    get active() {
      return active;
    },
    async stop() {
      if (!active) return undefined;
      active = false;

      mediaRecorder.stop();
      stream.getTracks().forEach((track) => track.stop());

      await new Promise<void>((resolve) => {
        mediaRecorder.onstop = () => resolve();
      });

      if (chunks.length === 0) {
        onClose();
        return undefined;
      }

      const blob = new Blob(chunks, {
        type: mediaRecorder.mimeType,
      });
      const ext = mediaRecorder.mimeType.includes('webm')
        ? 'webm'
        : 'ogg';

      const formData = new FormData();
      formData.append('file', blob, `recording.${ext}`);
      formData.append('model', 'whisper-1');

      try {
        const response = await fetch(OPENAI_TRANSCRIPTION_URL, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
          body: formData,
        });

        if (!response.ok) {
          const body = await response.text();
          throw new Error(
            `Whisper STT error (${response.status}): ${body}`,
          );
        }

        const data = await response.json();
        onClose();
        return (data.text as string) || undefined;
      } catch (err: unknown) {
        const error =
          err instanceof Error
            ? err
            : new Error('Whisper STT request failed');
        onError(error);
        onClose();
        return undefined;
      }
    },
  };

  return session;
}

function getSupportedMimeType(): string {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
  ];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return 'audio/webm';
}

export function getWhisperApiKey(): string {
  return process.env.NEXT_PUBLIC_OPENAI_API_KEY || '';
}
