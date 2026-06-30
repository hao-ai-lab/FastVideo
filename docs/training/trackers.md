# Training Trackers

FastVideo can send training metrics and validation media to Weights & Biases
or SwanLab. Tracking runs only on global rank 0, and local tracker files are
stored under `<output_dir>/tracker`.

## Supported Trackers

| Value | Backend | Installation |
|-------|---------|--------------|
| `wandb` | Weights & Biases | Included with FastVideo |
| `swanlab` | SwanLab | Install the optional `swanlab` dependency |
| `none` | Disable external tracking | No additional package |

You can enable more than one backend, for example `trackers: [wandb, swanlab]`.
Metrics and validation media are converted to the artifact type required by
each backend.

## Install SwanLab

For a published FastVideo installation, install the SwanLab extra:

```bash
uv pip install "fastvideo[swanlab]"
```

For an editable source checkout, include the same extra during installation:

```bash
uv pip install -e ".[swanlab]"
```

If FastVideo is already installed, you can install the compatible SDK directly:

```bash
uv pip install "swanlab>=0.6.7"
```

Authenticate once before starting a training run:

```bash
swanlab login
```

See the [SwanLab login documentation](https://docs.swanlab.cn/en/api/cli-swanlab-login.html)
for non-interactive and self-hosted setups.

## Configure Tracking

Select SwanLab in the YAML config used by the modular training framework:

```yaml
training:
  checkpoint:
    output_dir: outputs/my_run
  tracker:
    trackers: [swanlab]
    project_name: my_project
    run_name: my_run
```

To log to both supported services:

```yaml
training:
  tracker:
    trackers: [wandb, swanlab]
    project_name: my_project
    run_name: my_run
```

An empty or omitted `trackers` list selects W&B when `project_name` is set.
Use an explicit `none` entry to disable external tracking:

```yaml
training:
  tracker:
    trackers: [none]
```

## Validation Videos

SwanLab currently accepts GIF video artifacts. FastVideo converts validation
MP4 files and in-memory video arrays to GIF automatically before logging them.
For video files, FastVideo uses the sampling frame rate supplied by the caller,
or the source file's frame rate when no value is supplied. In-memory arrays use
the frame rate supplied by the caller. Both forms fall back to 16 FPS when no
frame rate is available.

For details about configuring validation callbacks, see
[Training Infrastructure](train_infra.md#callbacks-pluggable-hooks).
