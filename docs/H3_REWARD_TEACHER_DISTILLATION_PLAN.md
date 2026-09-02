# H3 reward-scored trajectory distillation for FastH3

## Decision

Add a **new, selectable offline H3-to-FastH3 distillation path** based on the
student half of REST/AMD (*RL-Native Distillation: Exploiting Scored
Trajectories for Few-Step Image Generation*, arXiv:2608.09226).

The existing paper-faithful FastH3 RVM implementation remains unchanged.

The new path is an offline specialization:

1. freeze Base H3;
2. sample `K` dense H3 trajectories for each training prompt;
3. score only each trajectory's terminal video with the existing published or
   Physion/MJ reward profile;
4. compute group-relative advantages independently per reward source;
5. cache four student-aligned trajectory segments plus immutable reward and
   provenance metadata;
6. train the released four-forward FastH3 LoRA by segment-velocity regression,
   modulated by REST's signed AMD coefficient.

This deliberately adopts an existing method instead of adding another RL
algorithm. The teacher is not updated, the FastH3 student does not generate the
training rollouts, and MJ-VIDEO is absent from the optimizer process after the
cache is built.

## Why REST/AMD is the closest published fit

REST already treats every teacher trajectory segment as a state-action
example. For teacher states at student boundaries

\[
  t_0 > t_1 > \cdots > t_K,
\]

its base target is the segment slope

\[
  v_k^{\mathrm{gt}}=
  \frac{x_{t_{k+1}}-x_{t_k}}{t_{k+1}-t_k}.
\]

For each reward source `j`, REST standardizes rewards within the `K` rollouts
for one prompt, clips the advantage to `[-1, 1]`, mixes advantages with
non-negative coefficients, and applies

\[
  c_i = \lambda(A_i^{\mathrm{mix}}+b),
  \qquad \lambda=1,\quad b=0.5.
\]

The same coefficient is shared by every segment from rollout `i`. Positive
coefficients imitate good teacher paths more strongly; sufficiently low-reward
paths receive a mild repulsive gradient.

## H3-specific correction

H3 uses one base denoising stage but different rational sigma shifts for video
(`12.0`) and audio (`3.0`). Therefore the target must **not** divide by the raw
base-timestep delta. For modality `m`, use

\[
  v_{k,m}^{\mathrm{gt}}=
  \frac{x_{k+1,m}-x_{k,m}}
       {\sigma_m(t_{k+1})-\sigma_m(t_k)}.
\]

This is exactly the `noise - clean` velocity convention returned by the
existing MiniMax H3 training adapter.

## Video reward, audio preservation

The reward profile evaluates video. It must not create a signed reward gradient
on audio. The initial production objective is

\[
  L = L_{\mathrm{AMD,video}} +
      \gamma_{a} L_{\mathrm{imitate,audio}},
  \qquad \gamma_a=1.
\]

Thus video receives reward-modulated teacher imitation, while audio receives
ordinary positive teacher-trajectory imitation. The shared transformer can
still transfer useful cross-modal structure, but low visual reward never tells
the model to repel the corresponding audio.

## Exact deployed schedule

The FastH3 student boundaries are fixed to

```
[1000, 750, 500, 250, 0]
```

The cache builder uses a configurable piecewise-dense H3 Euler grid containing
all five boundaries exactly. The default is 12 H3 substeps per student segment,
48 teacher forwards total. This is close to the released full-step H3 solver
budget while avoiding interpolation between stored states.

## Cache and execution contract

Cache generation is intentionally a **single data-parallel replica** job,
typically `SP4 x DP1` on four H100s. All sequence-parallel ranks participate in
every H3 forward. The SP leader alone decodes, scores, and writes samples.

A cache contains:

```
metadata.json
samples/00000000.pt
samples/00000001.pt
...
COMPLETE
```

Each tensor-only sample stores the five packed video/audio anchor states, the
five base timesteps, reward component values, per-component advantages, mixed
advantage, AMD coefficient, deterministic seed, prompt/candidate indices, and
text conditioning tensors. Metadata fingerprints the teacher, student
schedule, reward configuration, prompt source, dimensions, and FastVideo code
version. A mismatch is fatal; stale caches are never silently reused.

After `COMPLETE` exists, the production `SP4 x DP2` or `SP4 x DP4` training job
omits the H3 teacher and all reward models. It reads the immutable cache and
optimizes only the FastH3 LoRA.

## Validation ladder

1. CPU unit tests for advantage normalization, signed gradients, shifted-sigma
   targets, schedule anchors, and cache fingerprints.
2. Four-GPU cache smoke: two prompts, `K=2`, strict H3 load, finite distinct
   rewards, exact anchor tensors, successful resume.
3. One optimizer step from the completed cache; verify finite video/audio loss,
   LoRA-only gradients, and four timestep buckets.
4. Eight-GPU matched pilot versus the existing FastH3-on-policy RVM profile,
   holding prompts, reward profile, LoRA, learning rate, optimizer budget, and
   held-out evaluation fixed.
5. Scale only after held-out reward improves without visible motion collapse or
   audio degradation.

## Non-goals

- no change to RVM;
- no GRPO/PPO update;
- no differentiating through MJ-VIDEO;
- no student on-policy rollout collection;
- no endpoint-only top-K rejection that throws away teacher dynamics;
- no reward sign applied to the audio imitation loss;
- no claim of empirical success before the real H3 cache and GPU pilot pass.

## Primary sources

- REST/AMD: https://arxiv.org/abs/2608.09226
- RVM: https://arxiv.org/abs/2608.23664
- DMD2: https://arxiv.org/abs/2405.14867
- FastH3 model collection: https://huggingface.co/collections/FastVideo/fastvideo-fasth3
