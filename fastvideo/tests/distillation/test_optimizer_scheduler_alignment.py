import torch

from fastvideo.train.methods.base import TrainingMethod


class _FakeScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class _FakeOptimizer(torch.optim.Optimizer):
    def __init__(self) -> None:
        super().__init__([torch.zeros((), requires_grad=True)], {})
        self.step_calls = 0
        self.zero_grad_calls = 0

    def step(self, closure=None):  # noqa: ANN001, ANN201
        self.step_calls += 1
        if closure is not None:
            closure()

    def zero_grad(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.zero_grad_calls += 1


class _FakeModel:
    transformer = None

    def on_train_start(self) -> None:
        pass

    def get_rng_generators(self) -> dict:
        return {}


class _FakeCfg:
    class training:
        pass

    method: dict = {}
    validation: dict = {}


class _ScheduleMethod(TrainingMethod):
    def __init__(self, interval: int) -> None:
        self.student_opt = _FakeOptimizer()
        self.critic_opt = _FakeOptimizer()
        self.student_sched = _FakeScheduler()
        self.critic_sched = _FakeScheduler()
        cfg = _FakeCfg()
        cfg.method = {}
        cfg.validation = {}
        role_models = {"student": _FakeModel()}  # type: ignore[dict-item]
        super().__init__(cfg=cfg, role_models=role_models)
        self.interval = interval

    @property
    def _optimizer_dict(self):  # noqa: ANN201
        return {"student": self.student_opt, "critic": self.critic_opt}

    @property
    def _lr_scheduler_dict(self):  # noqa: ANN201
        return {"student": self.student_sched, "critic": self.critic_sched}

    def _update_student(self, iteration: int) -> bool:
        return iteration % self.interval == 0

    def single_train_step(self, batch, iteration, *, current_vsa_sparsity=0.0):  # noqa: ANN001, ANN201
        loss = torch.zeros((), requires_grad=True)
        return {"total_loss": loss}, {}, {}

    def get_optimizers(self, iteration):  # noqa: ANN001, ANN201
        optimizers = [self.critic_opt]
        if self._update_student(iteration):
            optimizers.append(self.student_opt)
        return optimizers

    def get_lr_schedulers(self, iteration):  # noqa: ANN001, ANN201
        schedulers = [self.critic_sched]
        if self._update_student(iteration):
            schedulers.append(self.student_sched)
        return schedulers


def test_optimizer_scheduler_alignment() -> None:
    method = _ScheduleMethod(interval=5)

    for step in range(1, 11):
        method.optimizers_schedulers_step(step)

    assert method.critic_opt.step_calls == 10
    assert method.critic_sched.step_calls == 10
    assert method.student_opt.step_calls == 2
    assert method.student_sched.step_calls == 2
