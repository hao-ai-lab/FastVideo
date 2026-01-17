from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from torch import nn
from typing import Any
import torch


class EventHook(ForwardHook):
    def __init__(self, content: str, event_list: list[str]):
        self.content = content
        self.event_list = event_list

    def name(self) -> str:
        return f"EventHook_{self.content}"

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        print(
            f"[{self.content}] Pre-forward called with args[0].shape: {args[0].shape}"
        )
        self.event_list.append(f"[pre]{self.content}")
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        print(
            f"[{self.content}] Post-forward called with outputs.shape: {output.shape}"
        )
        self.event_list.append(f"[post]{self.content}")
        return output


def test_hook_execution_order():
    """Test that hooks are executed in the correct order: LIFO for pre-hooks, FIFO for post-hooks."""
    # Create a simple model
    model = nn.Linear(10, 20)

    # Create event list to track hook execution order
    events = []

    # Create and push hooks in order: A then B

    manager = ModuleHookManager.get_from_or_default(model)

    hook_a = EventHook("A", events)
    hook_b = EventHook("B", events)

    manager.append_forward_hook(hook_a)
    manager.append_forward_hook(hook_b)

    # Perform a forward pass
    input_tensor = torch.randn(2, 10)
    model(input_tensor)

    # Verify the execution order is [pre_a, pre_b, post_b, post_a]
    # Pre-hooks should be FILO (First In Last Out): A then B
    # Post-hooks should be LIFO (Last In First Out): B then A
    expected_events = ["[pre]A", "[pre]B", "[post]B", "[post]A"]

    assert events == expected_events, (
        f"Expected {expected_events}, but got {events}"
    )
    print(f"✓ Hook execution order test passed: {events}")
