"""Per-WebSocket-connection code: state, controller, and helpers.

Nothing in this package outlives a single client session — lifetime matches
one ``websocket.accept()`` to disconnect cycle.
"""
