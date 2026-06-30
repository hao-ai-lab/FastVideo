from types import SimpleNamespace

import torch

from fastvideo.train.methods.rl.diffusion_nft import DiffusionNFTMethod


class _FakeEMA:

    def __init__(self):
        self.updates = 0

    def update(self, module):
        del module
        self.updates += 1


class _FakeTracker:

    def __init__(self):
        self.videos = []
        self.artifacts = []

    def video(self, data, *, caption=None, fps=None):
        self.videos.append({
            "data": data,
            "caption": caption,
            "fps": fps,
        })
        return f"video-{len(self.videos)}"

    def log_artifacts(self, artifacts, step):
        self.artifacts.append((artifacts, step))


def test_reward_diagnostic_metrics_match_per_prompt_groups():
    method = object.__new__(DiffusionNFTMethod)
    method._trained_prompt_hashes = set()

    sample_items = [{
        "prompts": ["a", "a"],
    }, {
        "prompts": ["b", "b"],
    }]
    rewards = {"avg": torch.tensor([1.0, 3.0, 2.0, 6.0])}

    metrics = method._reward_diagnostic_metrics(sample_items, rewards)

    assert metrics["group_size"] == 2.0
    assert metrics["trained_prompt_num"] == 2.0
    assert torch.isclose(metrics["zero_std_ratio"], torch.tensor(0.0))
    assert torch.isclose(metrics["reward_std_mean"], torch.tensor(1.5))
    assert torch.isclose(metrics["mean_reward_100"], torch.tensor(3.0))
    assert torch.isclose(metrics["mean_reward_50"], torch.tensor(4.5))

    method._reward_diagnostic_metrics(sample_items, rewards)
    assert len(method._trained_prompt_hashes) == 2


def test_update_ema_honors_update_after_step():
    method = object.__new__(DiffusionNFTMethod)
    method._ema_enabled = True
    method._student_ema = _FakeEMA()
    method._ema_update_count = 0
    method._ema_update_after_step = 1
    method.student = SimpleNamespace(transformer=object())

    method._update_ema()
    assert method._student_ema.updates == 0
    assert method._ema_update_count == 1

    method._update_ema()
    assert method._student_ema.updates == 1
    assert method._ema_update_count == 2


def test_log_validation_samples_caps_video_count_and_uses_configured_fps():
    method = object.__new__(DiffusionNFTMethod)
    method._validation_config = SimpleNamespace(fps=16, max_samples=1)
    tracker = _FakeTracker()
    method.tracker = tracker

    method._log_validation_samples(
        [
            {
                "index": 1,
                "prompt": "second",
                "media": torch.ones(3, 2, 4, 5),
                "rewards": {
                    "avg": 0.2,
                },
            },
            {
                "index": 0,
                "prompt": "first",
                "media": torch.ones(3, 2, 4, 5),
                "rewards": {
                    "avg": 0.1,
                },
            },
        ],
        iteration=10,
    )

    assert len(tracker.videos) == 1
    assert tracker.videos[0]["fps"] == 16
    assert "first" in tracker.videos[0]["caption"]
    assert tracker.artifacts == [({"validation/videos": ["video-1"]}, 10)]


def test_num_train_timesteps_uses_explicit_schedule_length():
    method = object.__new__(DiffusionNFTMethod)
    method._sample_steps = 25
    method._timestep_fraction = 0.5
    method._sampling_config = SimpleNamespace(
        timesteps=[900, 800, 700, 600, 500, 400, 300, 200, 100, 10],
        sigmas=None,
    )

    assert method._num_train_timesteps() == 5


def test_checkpoint_state_saves_frozen_old_policy_weights():
    student = torch.nn.Linear(2, 2)
    old = torch.nn.Linear(2, 2)
    for param in old.parameters():
        param.requires_grad_(False)

    method = object.__new__(DiffusionNFTMethod)
    method._role_models = {
        "student": SimpleNamespace(
            transformer=student,
            _trainable=True,
        ),
        "old": SimpleNamespace(
            transformer=old,
            _trainable=False,
        ),
    }
    method.student = method._role_models["student"]
    method.old = method._role_models["old"]
    method._student_optimizer = None
    method._student_lr_scheduler = None
    method._ema_enabled = False

    states = method.checkpoint_state()
    old_state = states["roles.old.transformer"].state_dict()

    assert "weight" in old_state
    assert "bias" in old_state
    assert torch.equal(old_state["weight"], old.weight)
