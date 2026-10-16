# Documentation structure and editorial plan

The reader's path should be **install → generate → choose a model → configure
and optimize → train or extend**. The sidebar previously put general tasks,
hardware recipes, model releases, and development instructions at the same level.

## Included in this PR

The sidebar now follows these groups, without moving existing URLs:

| Group                                     | Reader's task                                     | Contents                                                               |
| ----------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| Get started                               | Generate the first result                         | Quick start, installation, Python API basics                           |
| Fast Models                               | Pick a released checkpoint                        | Family cards, setup links, searchable exports and collections          |
| Run inference                             | Configure a working pipeline                      | Configuration, CLI, support matrix, memory, performance, LoRA          |
| Hardware                                  | Find platform-specific setup                      | NVIDIA GPUs, MLX, DGX Spark, paired Sparks, Spark tuning, RTX PRO 6000 |
| Integrations                              | Use FastVideo in another application              | ComfyUI, server and API clients                                        |
| Model guides                              | Follow model-specific instructions                | FastH3 schedules, LTX, Kandinsky generation and upscaling              |
| Cookbook                                  | Configure a deployment or follow a family runbook | React editor and existing family URLs                                  |
| Train and fine-tune                       | Adapt a model                                     | Data, fine-tuning, QAT, infrastructure, examples                       |
| Distill models                            | Train a faster student                            | Preprocessing, DMD, AnyFlow                                            |
| Attention research                        | Understand or use an attention method             | VSA, archived STA, backend development                                 |
| Contribute / Architecture / API reference | Extend and understand FastVideo                   | Development, testing, contracts, source reference                      |

Fast Models takes its selection-first structure, family cards, and consistent
catalog rows from the OpenAI models page. It keeps FastVideo's neutral theme and
links only to the maintained checkpoint catalog. It makes no new performance or
memory-fit claims.

## Consolidate next

These are editorial proposals. This PR does not delete useful technical content
or change runnable commands to make the sidebar shorter.

| Current pages                                             | Recommendation                                                                                                                                           | Reason and URL handling                                                                                                                                                                                       |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Getting started quick start + inference quick start       | Keep one first-video tutorial. Turn the inference guide into a task-oriented follow-on for image conditioning and generation options.                    | Both currently repeat installation and first-video code. Keep the existing route until its unique examples are moved; then redirect it to the matching section.                                               |
| Installation + NVIDIA GPU install                         | Make Installation a short platform chooser. Keep CUDA commands only in the GPU guide.                                                                    | Both contain environment and package setup. One owner per platform avoids drift.                                                                                                                              |
| Spark install + paired Sparks + Spark tuning              | Keep one DGX Spark entry in Hardware with install, multi-node, and tuning subpages.                                                                      | Pairing is an advanced deployment task. It should not interrupt the universal onboarding sequence. Keep its detailed networking and Ray instructions.                                                         |
| FastH3 on RTX PRO 6000                                    | Split the runnable hardware recipe from kernel internals and experiment history.                                                                         | The page mixes usage, per-block measurements, implementation details, and rejected experiments. Hardware owns the recipe; research/development owns the measurements. Preserve fragment links when splitting. |
| ComfyUI                                                   | Replace the current two-sentence pointer with a maintained integration guide, or link directly to the canonical ComfyUI README from an integrations hub. | The current page adds no instructions beyond the README link. Retain its URL as a redirect if removed.                                                                                                        |
| Optimizations + configuration                             | Let Configuration own supported settings. Split attention selection, quantization, distributed execution, and offloading into linked optimization tasks. | The optimization page is long and mixes mechanisms, hardware constraints, and release-specific flags. Avoid duplicating the offloading guide.                                                                 |
| Model-specific inference pages + cookbook family runbooks | Keep model capabilities and limitations in model guides; keep runnable deployment configuration in the cookbook.                                         | LTX and Kandinsky are model tasks, not peers of general configuration. Share setup links instead of copying commands.                                                                                         |
| `attention/developer` + `contributing/attention_backend`  | Reconcile into one backend-development guide under Contribute.                                                                                           | Both describe backend registration and implementation. Preserve the older URL as a redirect after checking for unique kernel instructions.                                                                    |
| Design overview + coding agents guide                     | Let Architecture own the package/configuration explanation. Keep the agent guide focused on the porting workflow and pitfalls.                           | Both currently reproduce the project structure. Link to the architecture instead of maintaining two maps.                                                                                                     |
| STA and mask-search examples                              | Keep in an explicitly archived research subsection.                                                                                                      | They describe an archive branch and should not look like a supported first-run path.                                                                                                                          |
| Generated example indexes and 880 API pages               | Keep searchable and cross-linked, but out of the main sidebar.                                                                                           | They are reference material. Do not hand-maintain generated pages or expose hundreds of navigation entries.                                                                                                   |

## Per-page contract

Use the [documentation authoring guide](../docs/contributing/documentation.md).
Every task page should answer: what will I achieve, what do I need, what do I run,
what output should I expect, what can go wrong, and where do I go next?

Keep setup guides short enough to follow sequentially. Put platform exceptions
next to the relevant step, and put performance evidence in a separate labeled
section with its device, checkpoint, configuration, and measurement date.

## Cutover and compatibility

This PR replaces the MkDocs site and build dependencies. Griffe reads Python
source without importing the model runtime. Canonical Markdown still goes through
Blume's pinned syntax normalizer so existing admonitions and tabs keep rendering.
The frozen migration manifest checks old routes and heading fragments.

The existing GitHub Pages workflow uploads Blume's `dist/` and deploys only from
`main` at `https://haoailab.com/FastVideo/`. There is one site and one publishing
workflow. Existing cookbook family selectors remain available inside the Blume
shell while their design is finished. All cookbook guide links stay on this site.

Before merging, review the new catalog and sidebar, run the full documentation
checks, and inspect CI. Any later page deletion needs a redirect and a link check.
