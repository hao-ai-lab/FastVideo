# v2_examples/workflows — cross-model composition

Runnable examples of **cross-model `Workflow`s** — chaining two *distinct* models, threading typed
artifacts between them (designv4 §9.6). A `Program` composes the loops of ONE resident model
(step-interleaved, parity-gated); a `Workflow` composes *across* model instances, above the engine — so
the per-step hot path is unchanged and each model keeps its own interleave-parity guarantee.

The shipped example is **T2I → I2V** (the realistic FLUX→Wan pipeline): a text-to-image model produces
an image, which conditions an image-to-video model. Run from anywhere:

```bash
python3 v2_examples/workflows/01_t2i_then_i2v.py
```

| Script | Shows |
|---|---|
| `01_t2i_then_i2v.py` | run the cross-model workflow; the I2V stage provably consumes the generated image |
| `02_register_as_servable.py` | a workflow is a first-class servable — declared `requires`, validated, addressable by `workflow_id` like a model; the declarative catalog + `WorkflowRegistry` |
