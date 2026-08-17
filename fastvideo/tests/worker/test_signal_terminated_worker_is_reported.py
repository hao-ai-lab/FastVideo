# SPDX-License-Identifier: Apache-2.0
"""A worker killed by a signal has to say so.

`worker_main` installs a SIGTERM/SIGINT handler that raises `SystemExit`. That is
a `BaseException`, not an `Exception`, so it walks straight past a bare
`except Exception` and the worker exits having reported nothing. The parent then
sees a closed pipe and raises "See stack trace for root cause" with no stack
trace, which is indistinguishable from a hang, a crash, or an out-of-memory kill.

That matters on machines that terminate processes under memory pressure. A DGX
Spark ships earlyoom configured to send SIGTERM to python first once available
memory drops below six percent, so this is the common failure there and it is
completely silent.

Driving the real function needs a process, a pipe, a distributed init and a
model, so these tests pin the structure instead: the handler exists, it is
ordered before the broad one, and it re-raises rather than swallowing the exit.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

from fastvideo.worker.multiproc_executor import WorkerMultiprocProc


def _worker_main_handlers() -> list[ast.ExceptHandler]:
    """The except clauses guarding the busy loop, in source order."""
    source = textwrap.dedent(inspect.getsource(WorkerMultiprocProc.worker_main))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                isinstance(handler.type, ast.Name) and handler.type.id == "Exception" for handler in node.handlers):
            return node.handlers
    raise AssertionError("worker_main no longer has a try block guarding the busy loop")


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    node = handler.type
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Tuple):
        return {element.id for element in node.elts if isinstance(element, ast.Name)}
    return set()


def test_system_exit_is_handled() -> None:
    handled = set().union(*(_handler_names(handler) for handler in _worker_main_handlers()))

    assert "SystemExit" in handled or "BaseException" in handled, (
        "a signal-terminated worker reports nothing: the SIGTERM handler raises SystemExit, which "
        "`except Exception` does not catch")


def test_system_exit_comes_before_the_broad_handler() -> None:
    """Order is the whole thing. Python takes the first matching clause, so a
    `SystemExit` clause after a `BaseException` one would never run."""
    handlers = _worker_main_handlers()
    positions = {name: index for index, handler in enumerate(handlers) for name in _handler_names(handler)}

    if "SystemExit" in positions and "BaseException" in positions:
        assert positions["SystemExit"] < positions["BaseException"]
    if "SystemExit" in positions:
        assert positions["SystemExit"] < positions["Exception"]


def test_the_handler_re_raises() -> None:
    """Reporting must not turn a termination into a normal return. The process
    still has to exit, and the parent still has to see the pipe close."""
    for handler in _worker_main_handlers():
        if "SystemExit" not in _handler_names(handler):
            continue
        assert any(isinstance(node, ast.Raise) for node in ast.walk(handler)), (
            "the SystemExit handler swallows the exit instead of re-raising it")
        return
    # A BaseException handler covers the case without a dedicated clause.
    assert any("BaseException" in _handler_names(handler) for handler in _worker_main_handlers())
