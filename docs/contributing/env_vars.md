# Environment Variables

FastVideo reads environment variables for expert switches, debugging, profiling, and the settings that launchers such
as `torchrun` provide. This page is the policy for those variables. The contract test
`fastvideo/tests/contract/test_env_policy.py` enforces the policy in the unit CI lane, and the coding-agent skill
`.agents/skills/env-var-conventions/SKILL.md` points here. When the policy changes, update this page and the contract
test in the same pull request.

## Rules

1. **Register every FastVideo variable in `fastvideo/envs.py`.** Each entry declares a type, a default, a category,
   and a description. Variables that other tools own (CUDA, NCCL, PyTorch, launchers) are not registered; code reads
   them directly with `os.environ.get("NAME")`, and the name must be in the external-variable allowlist
   (`EXTERNAL_ALLOWLIST` in the contract test).
2. **Read with `envs.NAME.get()`, write with `envs.NAME.set()`, and change a value in tests with
   `envs.NAME.override()`.** Each type has one parsing rule. A value that the rule rejects raises
   `fastvideo.envs.EnvVarError` instead of falling back to the default.
3. **Name FastVideo variables with the `FASTVIDEO_` prefix.** The second word states the purpose where one applies:
   `ENABLE_`, `DISABLE_`, `USE_`, `FORCE_`, `DEBUG_`, `TEST_`.
4. **Keep a renamed variable as a deprecated alias until the next minor release.** Setting the old name logs a
   warning. Delete a variable that no code reads, and list it in a deprecation table so that setting it logs a
   warning.
5. **Give each setting one source: an argument or an environment variable.** Settings that users change per
   deployment are arguments (CLI or YAML). Expert switches, emergency off switches, and debugging and test switches
   are environment variables.
6. **Read variables inside functions.** `envs.NAME.get()` runs when the function runs, so a changed value takes
   effect without re-importing a module. Module level, class bodies, decorators, and default argument values run at
   import time.
7. **Do not write the environment to pass values between parts of FastVideo.** Pass an argument instead. Tests use
   `envs.NAME.override()`.

## Field types

| Class        | Value type       | Parsing rule                                                                        |
| ------------ | ---------------- | ----------------------------------------------------------------------------------- |
| `EnvBool`    | `bool`           | `1`, `true`, `yes`, `on` are true; `0`, `false`, `no`, `off`, and `""` are false.   |
|              |                  | Case-insensitive; surrounding whitespace is ignored.                                |
| `EnvInt`     | `int`            | `int(value)`                                                                        |
| `EnvFloat`   | `float`          | `float(value)`                                                                      |
| `EnvStr`     | `str` or `None`  | The raw string. A `None` default means that the variable has no default.            |
| `EnvPath`    | `str` or `None`  | The raw string with a leading `~` expanded.                                         |
| `EnvChoice`  | `str`            | Stripped and lower-cased, then checked against the declared `choices`.              |

A default can be a zero-argument function; `get()` calls it on each read while the variable is unset. The path roots
use this to follow `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`.

Using a field without a method, as in `if envs.FASTVIDEO_FA4:`, raises `TypeError`.

## Add a variable

1. Declare the variable in the matching section of `fastvideo/envs.py`:

    ```python
    FASTVIDEO_DEBUG_MY_STAGE = EnvBool(False, category="debug", doc="Log the inputs of MyStage.")
    ```

    The category is one of the values in `envs.CATEGORIES`.

2. Read the variable inside a function:

    ```python
    import fastvideo.envs as envs

    def forward(self, batch):
        if envs.FASTVIDEO_DEBUG_MY_STAGE.get():
            logger.info("MyStage inputs: %s", batch.keys())
    ```

3. Regenerate the table at the end of this page:

    ```bash
    python fastvideo/tests/contract/test_env_policy.py
    ```

4. Run the contract test:

    ```bash
    pytest fastvideo/tests/contract/test_env_policy.py
    ```

In a test, change the value with `override`, which restores the previous value on exit:

```python
with envs.FASTVIDEO_DEBUG_MY_STAGE.override(True):
    run_stage()
```

## What the contract test checks

The test parses every Python file under `fastvideo/`, including `fastvideo/tests/`, with Python's `ast` module. It
skips `fastvideo/third_party/`, which is copied from upstream projects, and the registry `fastvideo/envs.py`. It does
not check `apps/`, `examples/`, `scripts/`, `fastvideo-kernel/`, or `docs/`.

It reports each violation as `<path>: <kind> <name>`:

| Kind               | Code that triggers it                                        | Fix                                          |
| ------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| `read`             | `os.getenv`, `os.environ.get`, `os.environ[...]`, or         | Register the variable and call               |
|                    | `"NAME" in os.environ` with a name outside the allowlist, or | `envs.NAME.get()`. For a variable that       |
|                    | with a name built at runtime (`<dynamic>`)                   | another tool owns, add it to                 |
|                    |                                                              | `EXTERNAL_ALLOWLIST` with a reason.          |
| `write`            | `os.environ[...] = ...`, `setdefault`, `pop`, `del`,         | Pass an argument instead. In tests, use      |
|                    | `os.putenv`, `os.unsetenv`, `monkeypatch.setenv`/`delenv`    | `envs.NAME.override()`.                      |
| `whole-environ`    | `os.environ.copy()`, `dict(os.environ)`, iteration,          | Read the specific variables that the code    |
|                    | `mock.patch.dict(os.environ, ...)`, `os.environ.update`      | needs.                                       |
| `bare-field`       | A registry field used without calling one of its methods,    | Call `envs.NAME.get()`.                      |
|                    | as in `envs.NAME == "auto"` or `getter = envs.NAME.get`      |                                              |
| `import-time-read` | `envs.NAME.get()` outside a function                         | Move the read into the function that uses    |
|                    |                                                              | the value.                                   |
| `prefix`           | A registry entry without the `FASTVIDEO_` prefix             | Rename the variable and keep the old name as |
|                    |                                                              | a deprecated alias.                          |
| `unread`           | A registry entry that no code reads with `get()` or          | Delete the variable and add it to the        |
|                    | `is_set()`                                                   | deprecation table.                           |

The test recognizes `os` imported under another name and `from os import environ, getenv`. Code that reaches the
environment through `importlib` or `getattr(os, "environ")` is left to code review.

The test also checks that every registry entry has a category from `envs.CATEGORIES` and a description, and that the
table at the end of this page matches the registry.

**Known violations.** `KNOWN_VIOLATIONS` in the contract test lists the violations that existed when the policy was
introduced. The list only shrinks. A violation that is not in the list fails the test. A listed violation that no
longer exists also fails the test, so the fixing pull request deletes its entry.

## Registered variables

<!-- BEGIN GENERATED ENV TABLE: python fastvideo/tests/contract/test_env_policy.py -->
| Variable                                       | Type                           | Default                     | Category    | Description                                                                                                                                                                                                                                              |
| ---------------------------------------------- | ------------------------------ | --------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FASTVIDEO_TARGET_DEVICE`                      | str                            | `cuda`                      | build       | Target device of FastVideo: cuda, rocm, neuron, cpu, or openvino.                                                                                                                                                                                        |
| `MAX_JOBS`                                     | str                            | unset                       | build       | Maximum number of parallel compilation jobs. Defaults to the number of CPUs.                                                                                                                                                                             |
| `NVCC_THREADS`                                 | str                            | unset                       | build       | Number of nvcc threads. When set, MAX_JOBS is reduced to avoid oversubscribing the CPU.                                                                                                                                                                  |
| `FASTVIDEO_USE_PRECOMPILED`                    | bool                           | `0`                         | build       | Use precompiled binaries (\*.so).                                                                                                                                                                                                                        |
| `CMAKE_BUILD_TYPE`                             | str                            | unset                       | build       | CMake build type: Debug, Release, or RelWithDebInfo.                                                                                                                                                                                                     |
| `VERBOSE`                                      | bool                           | `0`                         | build       | Print verbose logs during installation.                                                                                                                                                                                                                  |
| `FASTVIDEO_CONFIG_ROOT`                        | path                           | computed                    | path        | Root directory for FastVideo configuration files, at runtime and at installation. Defaults to ~/.config/fastvideo, or $XDG_CONFIG_HOME/fastvideo when XDG_CONFIG_HOME is set.                                                                            |
| `FASTVIDEO_CACHE_ROOT`                         | path                           | computed                    | path        | Root directory for FastVideo cache files. Defaults to ~/.cache/fastvideo, or $XDG_CACHE_HOME/fastvideo when XDG_CACHE_HOME is set.                                                                                                                       |
| `FASTVIDEO_HOST_IP`                            | str                            | `""`                        | distributed | IP address of this node when the node has several network interfaces. Set it on each node for multi-node inference.                                                                                                                                      |
| `FASTVIDEO_LOOPBACK_IP`                        | str                            | `""`                        | distributed | Loopback IP address to use instead of the detected one.                                                                                                                                                                                                  |
| `FASTVIDEO_RAY_PER_WORKER_GPUS`                | float                          | `1.0`                       | distributed | GPUs per Ray worker. A fraction lets Ray schedule several actors on one GPU, so other actors can share the GPUs with FastVideo.                                                                                                                          |
| `FASTVIDEO_RINGBUFFER_WARNING_INTERVAL`        | int                            | `60`                        | distributed | Seconds between warnings while the ring buffer is full.                                                                                                                                                                                                  |
| `FASTVIDEO_NCCL_SO_PATH`                       | str                            | unset                       | distributed | Path to the NCCL library file. Needed because the nccl&gt;=2.19 that PyTorch ships has a bug (https://github.com/NVIDIA/nccl/issues/1234).                                                                                                               |
| `HCCL_SO_PATH`                                 | str                            | unset                       | distributed | Path to the HCCL library file on Ascend NPUs.                                                                                                                                                                                                            |
| `FASTVIDEO_ENGINE_ITERATION_TIMEOUT_S`         | int                            | `60`                        | distributed | Timeout in seconds for each engine iteration.                                                                                                                                                                                                            |
| `FASTVIDEO_WORKER_MULTIPROC_METHOD`            | one of spawn, fork, forkserver | `spawn`                     | distributed | Multiprocessing start method for worker processes.                                                                                                                                                                                                       |
| `FASTVIDEO_ULYSSES_A2A`                        | one of off, auto               | `off`                       | distributed | Sequence-parallel all-to-all backend. off uses the NCCL path in DistributedAutograd.AllToAll4D. auto uses the fused NVLink kernel when the group is a load-store accessible mesh of 2, 4, 6, or 8 ranks in eager execution, and the NCCL path otherwise. |
| `LD_LIBRARY_PATH`                              | str                            | unset                       | external    | Searched for the NCCL library when FASTVIDEO_NCCL_SO_PATH is unset.                                                                                                                                                                                      |
| `LOCAL_RANK`                                   | int                            | `0`                         | external    | Local rank of the process in a distributed run; selects the GPU device id.                                                                                                                                                                               |
| `CUDA_VISIBLE_DEVICES`                         | str                            | unset                       | external    | Visible devices in a distributed run.                                                                                                                                                                                                                    |
| `FASTVIDEO_CONFIGURE_LOGGING`                  | bool                           | `1`                         | logging     | Configure logging at import. When true, FastVideo uses its default logging configuration or the file in FASTVIDEO_LOGGING_CONFIG_PATH.                                                                                                                   |
| `FASTVIDEO_LOGGING_CONFIG_PATH`                | str                            | unset                       | logging     | Path to a JSON logging configuration file.                                                                                                                                                                                                               |
| `FASTVIDEO_LOGGING_LEVEL`                      | str                            | `INFO`                      | logging     | Default logging level.                                                                                                                                                                                                                                   |
| `FASTVIDEO_LOGGING_PREFIX`                     | str                            | `""`                        | logging     | Prefix prepended to every log message.                                                                                                                                                                                                                   |
| `FASTVIDEO_STAGE_LOGGING`                      | bool                           | `0`                         | logging     | Log the time that each pipeline stage takes.                                                                                                                                                                                                             |
| `FASTVIDEO_ATTENTION_BACKEND`                  | str                            | unset                       | attention   | Attention backend, as an AttentionBackendEnum name such as TORCH_SDPA, FLASH_ATTN, VIDEO_SPARSE_ATTN, SAGE_ATTN, or SAGE_ATTN_THREE. FastVideoArgs uses it when FastVideoArgs.attention_backend is unset.                                                |
| `FASTVIDEO_FA4`                                | bool                           | `0`                         | attention   | The FLASH_ATTN backend uses FlashAttention-4 (flash_attn.cute) instead of FA3 or FA2.                                                                                                                                                                    |
| `FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN`       | bool                           | `0`                         | attention   | MiniMax-H3 dense DiT self-attention uses the FlashAttention-4 packed-varlen entry point. This changes the floating-point reduction order, so it is an inference-only opt-in.                                                                             |
| `FASTVIDEO_VSA_SM100A`                         | bool                           | `0`                         | attention   | VIDEO_SPARSE_ATTN_H3 sends no-grad tile-64 forwards to the data-center Blackwell (sm_100a) kernel. fastvideo-kernel reads the same variable with the same rule.                                                                                          |
| `FASTVIDEO_INFERENCE_TORCH_COMPILE`            | bool                           | `0`                         | performance | Compile each DiT transformer block with fullgraph torch.compile at inference. Same as FastVideoArgs.inference_torch_compile=True.                                                                                                                        |
| `FASTVIDEO_VAE_PARALLEL_DECODE`                | bool                           | `0`                         | performance | MiniMax-H3 VAE decode splits its temporal chunks across the sequence-parallel ranks instead of running serially on the output rank. Same as FastVideoArgs.vae_parallel_decode=True.                                                                      |
| `FASTVIDEO_VAE_PARALLEL_ENCODE`                | bool                           | `0`                         | performance | MiniMax-H3 reference-video VAE encode splits its temporal chunks across the sequence-parallel ranks. Same as FastVideoArgs.vae_parallel_encode=True.                                                                                                     |
| `FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY`       | str                            | unset                       | performance | Collective that moves chunks in parallel VAE decode: gather (used when unset) or all_gather.                                                                                                                                                             |
| `FASTVIDEO_MINIMAX_H3_FUSIONS`                 | str                            | `""`                        | performance | MiniMax-H3 inference-only Triton fusions: all, 1, or a comma-separated subset of modulate,qknorm_rope,swiglu. Empty, 0, or none keeps the eager implementation.                                                                                          |
| `FASTVIDEO_TEST_DYNAMO_FULLGRAPH_CAPTURE`      | bool                           | `1`                         | debug       | Enable Dynamo fullgraph capture.                                                                                                                                                                                                                         |
| `FASTVIDEO_NVTX_PROFILE`                       | bool                           | `0`                         | profiling   | Emit NVTX ranges for external profilers such as Nsight Systems.                                                                                                                                                                                          |
| `FASTVIDEO_TORCH_PROFILER_DIR`                 | path                           | unset                       | profiling   | Enables the torch profiler and sets the directory for its traces. Must be an absolute path.                                                                                                                                                              |
| `FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES`       | bool                           | `0`                         | profiling   | Torch profiler records shapes.                                                                                                                                                                                                                           |
| `FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY` | bool                           | `0`                         | profiling   | Torch profiler profiles memory.                                                                                                                                                                                                                          |
| `FASTVIDEO_TORCH_PROFILER_WITH_STACK`          | bool                           | `0`                         | profiling   | Torch profiler captures stacks. Costs about 1.5x runtime and 1.4x trace size.                                                                                                                                                                            |
| `FASTVIDEO_TORCH_PROFILER_WITH_FLOPS`          | bool                           | `0`                         | profiling   | Torch profiler profiles FLOPs.                                                                                                                                                                                                                           |
| `FASTVIDEO_TORCH_PROFILE_REGIONS`              | str                            | `""`                        | profiling   | Comma-separated profiler regions to record. The torch profiler requires at least one region.                                                                                                                                                             |
| `FASTVIDEO_SERVER_DEV_MODE`                    | bool                           | `0`                         | debug       | Run the server in development mode with extra debugging endpoints.                                                                                                                                                                                       |
| `FASTVIDEO_TRACE_FUNCTION`                     | bool                           | `0`                         | debug       | Trace function calls.                                                                                                                                                                                                                                    |
| `FASTVIDEO_TRACE_ACTIVATIONS`                  | bool                           | `0`                         | debug       | Enable activation trace hooks.                                                                                                                                                                                                                           |
| `FASTVIDEO_TRACE_LAYERS`                       | str                            | `""`                        | debug       | Regex filter for traced module names. Empty means all.                                                                                                                                                                                                   |
| `FASTVIDEO_TRACE_STATS`                        | str                            | `abs_mean,sum`              | debug       | Comma-separated activation statistics dumped for each output tensor.                                                                                                                                                                                     |
| `FASTVIDEO_TRACE_OUTPUT`                       | str                            | `/tmp/fv_trace_<pid>.jsonl` | debug       | JSONL path for activation traces. The literal &lt;pid&gt; is replaced at runtime.                                                                                                                                                                        |
| `FASTVIDEO_TRACE_STEPS`                        | str                            | `""`                        | debug       | Comma-separated denoising step indices. Empty means all.                                                                                                                                                                                                 |
| `FASTVIDEO_CFG_GATE_STEP`                      | float                          | `1.0`                       | sampling    | CFG gating fraction in [0, 1]. Steps before len(timesteps) \* X run the conditional and unconditional forwards; later steps reuse the cached difference. 1.0 disables gating.                                                                            |
<!-- END GENERATED ENV TABLE -->
