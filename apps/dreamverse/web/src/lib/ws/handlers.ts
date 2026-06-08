export interface DecodedJsonEvent {
  kind: 'json';
  data: any;
}

export interface DecodedBinaryEvent {
  kind: 'binary';
  data: ArrayBuffer;
}

export interface DecodedIgnoreEvent {
  kind: 'ignore';
  data: null;
}

export type DecodedWebSocketEvent =
  | DecodedJsonEvent
  | DecodedBinaryEvent
  | DecodedIgnoreEvent;

export async function decodeWebSocketEvent(
  event: MessageEvent,
): Promise<DecodedWebSocketEvent> {
  if (typeof event.data === 'string') {
    return {
      kind: 'json',
      data: JSON.parse(event.data),
    };
  }

  const chunk =
    event.data instanceof ArrayBuffer
      ? event.data
      : event.data instanceof Blob
        ? await event.data.arrayBuffer()
        : null;

  if (!chunk) {
    return { kind: 'ignore', data: null };
  }

  return {
    kind: 'binary',
    data: chunk,
  };
}

export async function routeSocketMessage(
  data: any,
  handlers: Record<string, (data: any) => Promise<void> | void>,
  onUnhandled:
    | ((data: any) => Promise<void> | void)
    | null = null,
): Promise<boolean> {
  const handler = handlers[data?.type];
  if (handler) {
    await handler(data);
    return true;
  }

  if (onUnhandled) {
    await onUnhandled(data);
  }
  return false;
}
