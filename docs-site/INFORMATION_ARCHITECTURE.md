# Documentation and cookbook plan

Readers should be able to choose a model, find their hardware path, generate a
result, and then customize or train it without searching several sections for
the same instructions.

## One home for each kind of information

| Area                  | Question it answers                      | Owns                                                                                                                                    |
| --------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Fast Models           | Which FastVideo release should I choose? | Our released families, checkpoint exports, collections, short capability descriptions, links to recipes                                 |
| Support matrix        | What else can FastVideo run?             | The wider supported model set and optimization compatibility                                                                            |
| Hardware and runtimes | Where can I run it?                      | NVIDIA RTX/workstation and data center CUDA paths, DGX Spark, Apple MLX, AMD ROCm, platform requirements and limits                     |
| Documentation         | How does FastVideo work?                 | Installation, the first generation, configuration, CLI, parallelism, offloading, integrations, training, architecture and API reference |
| Cookbook              | How do I run this model for this task?   | One family home, runtime choices, runnable recipes, model-specific configuration, troubleshooting and evidence                          |

Fast Models highlights our models without implying that FastVideo only supports
those releases. CUDA and ROCm are runtimes; RTX, data center GPUs and DGX Spark
are hardware paths. MLX is the Apple Silicon runtime. These distinctions should
be visible before readers choose an installation command.

## Included now

- One Hardware and runtimes overview links the five hardware paths. AMD is
  explicitly source-backed support with incomplete installation/device coverage,
  rather than a promise that CUDA recipes or kernels work on every AMD GPU.
- The getting-started sidebar contains Quick start, Installation, Hardware and
  runtimes, and Python API basics.
- Run inference contains shared tasks. Hardware and Integrations are expandable
  groups; individual hardware recipes do not interrupt onboarding.
- Model-specific references sit beside their family recipes in the Cookbook
  sidebar. There is no separate Model guides list competing with the cookbook.
  Existing LTX, Kandinsky and FastH3 reference URLs remain intact.
- Fast Models points FastH3 and FastWan readers to their cookbook families.
  FastMetal also has a direct MLX setup entry.
- Generated examples have one visible page title, with the source link below it.
  Their previous heading fragments remain available.

## The intended reader flow

1. Start at **Fast Models** to choose a released model, or the support matrix for
   the broader catalog.
2. Check **Hardware and runtimes**, then follow the matching installation guide.
3. Open the model's **Cookbook family**. Choose the task and runtime, configure
   a recipe, and run it.
4. Follow links to shared configuration, offloading or optimization docs when
   more explanation is needed.
5. Move to training, distillation or contributor guides for development work.

The quick start remains a short route for readers who want a first result before
browsing the catalog.

## Next editorial pass

1. **Consolidate onboarding.** Keep one first-video tutorial. Remove repeated
   installation blocks from the inference quick start after preserving its unique
   image-conditioning examples. Make Installation a platform chooser and keep
   installation commands in platform guides.
2. **Give every model one family home.** Expand the cookbook family page with a
   short capability overview, runtime/task recipes, checkpoint choices, limits,
   troubleshooting and next steps. Fold overlapping model-specific guides into
   that page or link a detailed reference from it. Preserve old routes with
   redirects before deleting any page.
3. **Separate explanation from recipes.** Hardware owns environment setup;
   cookbook owns model commands; shared docs explain the options. The RTX PRO
   6000 page's kernel internals and rejected experiments belong in research or
   contributor material, separate from its runnable recipe.
4. **Reduce pointer pages and duplicate references.** Replace the two-sentence
   ComfyUI pointer with a useful integration guide, or link directly to its
   maintained README. Reconcile the two attention-backend development guides.
   Let Architecture own the package map instead of repeating it in the agent guide.
5. **Split long optimization material by task.** Keep attention selection,
   quantization, distributed execution and offloading cross-linked, with one
   owner for each setting. Keep STA and mask search clearly archived.
6. **Fill the ROCm evidence gap.** Document a complete AMD installation and model
   run with GPU architecture, ROCm/PyTorch versions, model revision, attention
   backend, expected output and measured memory. Source implementation alone
   should not turn into a broad device-support claim.

## Per-page structure

Task guides follow: outcome → requirements → runnable example → expected output
→ configuration → troubleshooting → next step. A model recipe adds its checkpoint,
workload, runtime, hardware evidence and measured limitations. Detailed reference
and implementation history should not interrupt setup commands.

See the [authoring guide](../docs/contributing/documentation.md). Examples and
Python reference stay source-generated and searchable; they do not need hundreds
of sidebar entries.

## Publishing

Blume replaces MkDocs through the existing GitHub Pages workflow at
`https://haoailab.com/FastVideo/`. Only `main` deploys. The migration manifest
checks previous routes and heading fragments. This PR changes navigation and
presentation while keeping the technical content available; the consolidation
steps above remain editorial follow-ups.

## Cookbook model pages

The cookbook catalog includes the maintained video, image, audio, and world
models, with category and family filters. FastH3 appears first. Each checkpoint
has its own page; related task or hardware recipes become variants on that page.
Family groups hold the pages alongside their broader runbooks.

FastH3 8-Step V2, FastWan2.1 1.3B, Wan2.1 I2V 14B, and LTX2 Distilled use the five
existing editable serving deployments. Their pages have a model header, YAML
actions, generated commands, and a compact hardware/options matrix. Other models
publish existing source commands and YAML without inventing editable settings.
Hardware-fit validation remains unverified. New quantization, encoder, or runtime
choices require a backend recipe before becoming selectable.
