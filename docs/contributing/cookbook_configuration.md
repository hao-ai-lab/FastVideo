# Extending cookbook configuration metadata

Keep option definitions beside FastVideo's existing Python configuration code.
The cookbook exporter turns those declarations into JSON Schema for the browser;
contributors should not duplicate them in per-model cookbook YAML or edit
generated files under `docs/assets/cookbook-config/` by hand.

`docs/cookbook/config-builder-models.yaml` selects the registered model IDs
published in the builder and their dropdown order. Generation writes a small
`index.json` and one complete catalog under `models/` for each selected ID. The
browser fetches the selected catalog when needed. These assets are generated
before the docs build and are not committed; the YAML contains no field types,
defaults or constraints.

## Choose the existing owner

| Information to add | Python owner | Current exporter support |
|---|---|---|
| A common serving field, type or numeric bound | The relevant dataclass in `fastvideo/api/schema.py` | Exported through `TypeAdapter(ServeConfig)`. Applies to every model using that field. |
| A public pipeline option specific to a model or family | Its registered pipeline config class and existing `add_cli_args()` declarations | Legacy options under `generator.pipeline.experimental` export field metadata, defaults and CLI choices. |
| A model's sampling default, such as frame count | Its existing preset in `fastvideo/pipelines/basic/<family>/presets.py` | Defaults override the common schema defaults. Registered sampling-class defaults are the fallback when no preset is selected. |
| A model-specific bound on a shared field, or a rule about supported combinations | The model or family's Python declarations | Requires a shared exporter extension; see below. |

For Wan, pipeline declarations live in
`fastvideo/models/wan/pipeline_config.py`, while sampling defaults live in
`fastvideo/pipelines/basic/wan/presets.py`. Put a restriction on the narrowest
applicable class: changing a base class can affect all its subclasses.

## Add a numeric bound

For an already public experimental pipeline option, keep its type and default
and add verified bounds with standard dataclass field metadata. This illustrative
field has a supported range of zero through one:

```python
from dataclasses import field

custom_gain: float = field(
    default=0.5,
    metadata={"ge": 0.0, "le": 1.0},
)
```

The exporter produces `type: number`, `default: 0.5`, `minimum: 0.0` and
`maximum: 1.0`. The demo renders a numeric input and Ajv validates its value.
For a new pipeline option, also expose it through the existing public CLI
declaration and ensure the runtime consumes it. Adding an arbitrary dataclass
attribute alone does not make a working public option.

For categorical options, keep the Python annotation and CLI `choices`
consistent: the current experimental-option exporter takes its enum choices
from the CLI declaration. Its help text also supplies the displayed description.
Preserve `None` when it means automatic selection. A recommended or tested value
is not evidence of a hard supported bound.

## Extensions still needed

The current exporter does **not** extract model-specific sampling ranges from
presets or sampling classes; it extracts their defaults only. It also copies
only the model default when a legacy pipeline argument maps to an existing typed
serving path. Extra constraints on that pipeline declaration are not yet merged.

For those cases, extend the existing Python declaration and the shared exporter
once to carry the constraint to the appropriate serving schema path. Then each
model can supply its own metadata through the same mechanism. Do not put range
objects into a preset's `defaults` mapping: the runtime expects values there.
Field applicability and cross-field rules likewise need explicit declarations;
the exporter cannot infer them from inheritance or arbitrary Python validators.

## Runtime impact and verification

Adding `ge` or `le` metadata while preserving the type and default does not, by
itself, change ordinary dataclass construction or FastVideo's native config
parser. It changes exported browser validation. Pydantic validation consumers
can enforce the metadata, but the native parser does not currently enforce those
bounds. If the server must reject the same values, add or reuse runtime validation
deliberately and test it separately.

Changing defaults, types, CLI choices or runtime validators can change existing
behavior. Treat those as API changes, even when motivated by the cookbook.

After changing a declaration:

1. Add a focused case in `tests/local_tests/test_cookbook_config_metadata.py`
   for the model's exported type, default and constraints.
2. Regenerate the catalogs from the repository root with
   `python docs/cookbook_config.py`, then run `mkdocs serve` or `mkdocs build`.
   The [docs setup instructions](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/README.md) include the CPU dependencies for
   generation; a full FastVideo inference installation is unnecessary.
3. Run the metadata tests, `test_cookbook_config_roundtrip.py`, and
   `node --test tests/local_tests/test_cookbook_config.mjs`; check valid and
   invalid selections in the demo. Run runtime tests if behavior changed.

The shared browser code should not need model-specific branches. UI labels and
layout stay in the frontend; hardware capacity estimation remains separate from
configuration constraints.
