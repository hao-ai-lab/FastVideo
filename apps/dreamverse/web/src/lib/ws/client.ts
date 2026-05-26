interface CreateWebSocketConnectionParams {
  url: string;
  binaryType?: BinaryType;
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
}

export function createWebSocketConnection({
  url,
  binaryType = 'arraybuffer',
  onOpen = () => {},
  onMessage = () => {},
  onError = () => {},
  onClose = () => {},
}: CreateWebSocketConnectionParams): WebSocket {
  const ws = new WebSocket(url);
  ws.binaryType = binaryType;
  ws.onopen = onOpen;
  ws.onmessage = onMessage;
  ws.onerror = onError;
  ws.onclose = onClose;
  return ws;
}

export function detachAndCloseWebSocket(
  ws: WebSocket | null,
): void {
  if (!ws) return;
  ws.onopen = null;
  ws.onmessage = null;
  ws.onerror = null;
  ws.onclose = null;
  ws.close();
}
