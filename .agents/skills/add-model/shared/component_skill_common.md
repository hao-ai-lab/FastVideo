# Component Skill Common Instructions

These instructions apply to `add-model-03-port-dit`, `add-model-04-port-vae`,
`add-model-05-port-encoder`, and `add-model-06-port-generic`. Bucket-specific skills
add target paths, implementation patterns, drift checks, and scope questions.

## Required Context

Require the complete packet from `../contracts/component_context.md`.

Do not start if the official definition files, official instantiation files, or
parity test path are missing. Ask the `/add-model` orchestrator for the complete
component context packet instead of rediscovering broad scope silently.

If the parity scaffold is missing, create it first with
`../../add-model-02-parity/templates/component_parity_test.py`.

## Prototype Mode

Prototype mode runs before conversion:

- implement or prove reuse for the minimal native component, config, export, and
  `EntryClass` surface needed by the relevant loader;
- instantiate with random weights using the exact official architecture args, or
  instantiate/document stateless components with no weights;
- dump official and FastVideo `state_dict()` names/shapes for every stateful
  component so conversion can derive mappings from real surfaces;
- return concerns discovered during prototype work, such as ambiguous official
  flags, shape mismatches, private ops, missing loader buckets, passthrough
  weights, or output heads;
- update `local_tests_readme` with prototype status and key-dump paths;
- update `port_state_file` with prototype status, open questions, issues, and
  handoff notes;
- do not chase numerical parity and do not block on converted weights.

Prototype mode succeeds when the component imports, instantiates with official
args, and required key/shape dumps exist. Converted weights and parity PASS are
not required yet.

## Parity-Debug Mode

Parity-debug mode runs after conversion:

- strict-load converted weights through the same path the pipeline will use, or
  document that the component is stateless or an approved passthrough;
- use `conversion_context` and `concerns_or_unknowns` to decide whether a failure
  belongs to mapping, loading, implementation, tokenization, normalization,
  scheduler semantics, or the parity test;
- run only the component parity test first with `pytest <parity_test> -v -s`;
- if it skips, fix the missing official import, FastVideo class, tokenizer,
  converted weights, or path;
- if it fails numerically, add targeted intermediate comparisons to identify the
  first divergent operation or tensor;
- update component implementation only when the failure is a component
  layer/config/forward/contract bug;
- update `local_tests_readme` with the command, result, and blocker or PASS;
- update `port_state_file` with parity status, resolved/new issues, open
  questions, and handoff notes;
- keep iterating until the test is a non-skip PASS or return a precise blocker.

## Conversion Boundary

Component skills must not patch conversion scripts or converted weights ad hoc.
If the first drift or strict-load failure points to wrong keys, missing tensors,
shape mismatches, component prefixes, split/fuse logic, skipped-key policy, or
config emission, return a conversion retry request for
`../../add-model-07-conversion/SKILL.md` matching
`../contracts/conversion_request.md`.
Resume parity-debug only after conversion returns an updated handoff.

## Reuse Proof

When `fastvideo_target_files` point to existing FastVideo code instead of a new
port:

- compare the official definition against the FastVideo target: graph/operation
  structure, parameter or state shapes, normalization, activation, positional or
  temporal behavior, scaling constants, dtype behavior, state-dict names, output
  containers, and output tensors;
- compare the official instantiation against the FastVideo config and loader
  args: constructor args, config values, defaults, variant flags, optional
  submodules, checkpoint metadata, tokenizer/media paths, and loader path;
- treat a matching class instantiated with different args as not reusable;
- record reuse evidence in `local_tests_readme` and keep the reused component in
  `prototype_key_dumps` when it owns state so conversion and parity-debug use the
  same surface;
- still run parity-debug to a non-skip PASS. If mismatch is found, return the
  concern so `/add-model` can switch the component to a native port.

## Handoff

Return `../contracts/component_skill_handoff.md`.

Mode-specific expectations:

- In `mode=prototype`, `parity_status` may be `scaffold_skip` or `blocked`; key
  dumps are the required artifact and successful prototype handoff should use
  `next_step=phase_5_conversion`.
- In `mode=parity-debug`, final success requires
  `parity_status=non_skip_pass`.
- If conversion is implicated, return `conversion_retry_request` and leave
  conversion edits to `add-model-07-conversion`.
- If production loading is non-strict, list allowed missing/unexpected keys in
  the parity test or handoff and mark
  `strict_load=pass_with_documented_exclusions`.

## Escape Hatches

Follow `common_rules.md`. Do not ask for normal prototype or parity-debug
failures such as missing imports, tokenizer/path issues, red parity,
strict-load failures, key mismatches, shape mismatches, or implementation bugs.
Return conversion retry requests or precise blockers as directed by the workflow.

Ask only when component work requires a scope or safety decision, such as
dropping a required stream/output/path, changing core dependencies, accepting
private model code or unsupported private ops, choosing between incompatible
official definitions, creating a new loader bucket, or loosening required parity.
