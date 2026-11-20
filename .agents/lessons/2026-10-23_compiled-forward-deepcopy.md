---
date: 2026-10-23
category: evaluation
severity: important
---

# Copying an instance-compiled module can benchmark the original module

## What happened

A decoder quantization experiment replaced every intended linear in a copied
module, yet measured exactly the original output and almost the original time.
The replacement layers were present in `named_modules()`, which was insufficient
evidence that the new computation actually ran.

## Root cause

`ComposedPipelineBase._compile_with_conditions` assigns
`submodule.forward = torch.compile(submodule.forward)`. The returned function
closes over a bound method of the original module. Python's `copy.deepcopy`
copies module state but retains function objects and their closures. The copied
instance therefore still has a compiled `forward` that calls the original
instance. Unwrapping `_orig_mod` alone does not handle this form of compilation.

## Fix

Prefer constructing a fresh module and loading the intended state. If copying
this known instance-compiled form for an experiment, remove the copied instance's
`forward` override before modifying children, then compile its newly bound class
method. Do not remove arbitrary user-defined forward overrides blindly.

## Prevention

Inspect the callable as well as the module tree. Verify a replacement executes
with a temporary hook or profiler trace, compare its output against the intended
numerical oracle, and run final timing in a fresh process. An unchanged output
from a supposedly lossy quantization is a reason to investigate dispatch before
accepting the measurement.
