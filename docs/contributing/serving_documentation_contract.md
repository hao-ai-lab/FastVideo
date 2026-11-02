# Serving documentation contract

This page defines the contributor-facing boundary between the HTTP schema, runtime
capabilities, and evidence. It is a handoff contract for the server and docs-site
owners; it is not an HTTP reference and does not restate endpoint behavior.

## Ownership

- **Server owner:** FastAPI routes, Pydantic request/response models, error
  envelopes, status codes, authentication, streaming, and the generated OpenAPI
  document.
- **Docs-site owner:** the HTTP reference route, copyable client examples, and
  reference navigation. Keep it separate from the Python SDK/CLI reference.
- **Capacity owner:** hardware identity, memory estimates, and evidence
  applicability. These fields never prove HTTP or model capability.

OpenAPI is a schema description. It does not establish OpenAI API or SDK
behavioral compatibility. Claim compatibility only for an exact endpoint,
client and version, payload, response, and error behavior that has a test or a
recorded run.

## Generated schema

Generate the HTTP schema from the application's actual routes and Pydantic
models and commit a deterministic export for review. Do not maintain a parallel
hand-written schema.

The export/check path must not start the application lifespan, load model
weights, initialize CUDA/MLX, require a GPU, bind a port, or fetch a remote
source. Inspect imports before claiming that property; moving an import can
reintroduce a lifespan or hardware dependency. Do not publish local paths,
credentials, or private deployment URLs in the export or examples.

The current server returns request-validation failures as HTTP 400 with its
OpenAI-style error envelope. Document those actual responses. Do not copy
FastAPI's default 422 schema when the mounted application does not return it.

## Workload IDs

Use these shared IDs in runtime `workloads`, recipe/deployment `workload`, and
capability rows. The shared media map lives in
`docs/js/cookbook-workloads.mjs`; the catalog builder and sample client use it as
the single workload allow-list. The server owner implements endpoints and the
docs-site owner renders from the declared input/output media rather than
assuming video.

| ID | Input | Output | Minimal semantics |
| --- | --- | --- | --- |
| `t2v` | text | video | Text prompt to a video clip |
| `i2v` | image + text | video | Image reference drives the first frame or motion |
| `t2i` | text | image | Text prompt to one still image |
| `t2a` | text | audio | Text prompt to an audio clip |
| `v2a` | video + text | audio | Video soundtrack, speech, or audio response |

The map records the canonical sample endpoint, extension, and media type used
by cookbook examples; it does not prove that a model or deployment supports the
workload. Do not infer support from an ID. The capability row must name the
actual operation and media type, and unsupported combinations stay
`unsupported` or `unknown` rather than falling back to a video route.

## Capability matrix

Schema presence does not mean every model supports an operation. Serving docs
must publish a matrix with stable fields:

| Field | Meaning |
| --- | --- |
| `model_id` | Checkpoint identity used by the deployment |
| `deployment_id` | Canonical cookbook deployment (`<model>/<deployment>`) |
| `workload` | Shared workload ID and its input/output media |
| `runtime` | FastVideo runtime/adapter identifier |
| `endpoint` | Method and path |
| `operation` | Create, poll, download, cancel, list, or delete |
| `status` | `supported`, `unsupported`, or `unknown` |
| `constraint` | Required option, model limitation, or admission rule |
| `evidence` | Link and kind, with revisions; see below |

Mark FastVideo extensions explicitly. State unsupported and unknown operations
instead of implying parity. Cover success and error envelopes, multipart and
binary responses, and the implemented job lifecycle (create, poll, download,
cancel/delete). Document authentication as configured. Call out streaming and
WebSocket limitations rather than describing a capability the server does not
mount.

## Evidence

Use three distinct labels:

- `schema`: the route and shapes exist in the generated OpenAPI document.
- `contract_test`: a focused test exercises the request/response or client path.
- `measured_generation`: a recorded run produced the media on named hardware.

Only the last label supports a GPU-validated generation claim. A schema check,
unit test, or successful process start does not establish output quality, model
capability, or memory fit. Link the revision and the exact client version with
any compatibility claim.

Capacity evidence follows the same rule: a completed generation is
compatibility evidence, while a device-wide memory measurement is required for a
measured fit. See [capacity estimation](capacity_estimation.md#optional-successful-run-evidence)
for the memory-evidence schema.

## User documentation

Start from a maintained cookbook recipe and a selected hardware/runtime
combination, then show the first generation with a matching cURL, OpenAI Python,
or OpenAI JavaScript example where that path is tested. Follow with polling,
download, and troubleshooting. Keep in-process FastVideo SDK/CLI examples
separate so readers do not confuse them with HTTP clients.

A static documentation site must remain useful without a running inference
server. A browser "Try it" control is optional and may only target a
user-selected reachable server within that server's actual CORS, authentication,
and HTTPS policy. Do not add a shared public inference endpoint, a hosted
request proxy, a broad CORS exception, or an authentication bypass for a
tutorial.

## Contributor checks

Update the model/deployment/runtime capability row when adding serving support.
Include the generated schema, focused contract/client tests, link checks, and
the evidence links. CI should detect a stale export and check meaningful
request, response, and client behavior independently of the schema diff.
