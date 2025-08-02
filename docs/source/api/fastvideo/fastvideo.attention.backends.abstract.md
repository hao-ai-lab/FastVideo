# {py:mod}`fastvideo.attention.backends.abstract`

```{py:module} fastvideo.attention.backends.abstract
```

```{autodoc2-docstring} fastvideo.attention.backends.abstract
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AttentionBackend <fastvideo.attention.backends.abstract.AttentionBackend>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`AttentionImpl <fastvideo.attention.backends.abstract.AttentionImpl>`
  -
* - {py:obj}`AttentionLayer <fastvideo.attention.backends.abstract.AttentionLayer>`
  -
* - {py:obj}`AttentionMetadata <fastvideo.attention.backends.abstract.AttentionMetadata>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadata
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`AttentionMetadataBuilder <fastvideo.attention.backends.abstract.AttentionMetadataBuilder>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadataBuilder
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <fastvideo.attention.backends.abstract.T>`
  - ```{autodoc2-docstring} fastvideo.attention.backends.abstract.T
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

`````{py:class} AttentionBackend
:canonical: fastvideo.attention.backends.abstract.AttentionBackend

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} accept_output_buffer
:canonical: fastvideo.attention.backends.abstract.AttentionBackend.accept_output_buffer
:type: bool
:value: >
   False

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend.accept_output_buffer
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_builder_cls() -> type[fastvideo.attention.backends.abstract.AttentionMetadataBuilder]
:canonical: fastvideo.attention.backends.abstract.AttentionBackend.get_builder_cls
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend.get_builder_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_impl_cls() -> type[fastvideo.attention.backends.abstract.AttentionImpl]
:canonical: fastvideo.attention.backends.abstract.AttentionBackend.get_impl_cls
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend.get_impl_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_metadata_cls() -> type[fastvideo.attention.backends.abstract.AttentionMetadata]
:canonical: fastvideo.attention.backends.abstract.AttentionBackend.get_metadata_cls
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend.get_metadata_cls
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_name() -> str
:canonical: fastvideo.attention.backends.abstract.AttentionBackend.get_name
:abstractmethod:
:staticmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionBackend.get_name
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} AttentionImpl(num_heads: int, head_size: int, softmax_scale: float, causal: bool = False, num_kv_heads: int | None = None, prefix: str = '', **extra_impl_args)
:canonical: fastvideo.attention.backends.abstract.AttentionImpl

Bases: {py:obj}`abc.ABC`, {py:obj}`typing.Generic`\[{py:obj}`fastvideo.attention.backends.abstract.T`\]

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.T) -> torch.Tensor
:canonical: fastvideo.attention.backends.abstract.AttentionImpl.forward
:abstractmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionImpl.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} postprocess_output(output: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.T) -> torch.Tensor
:canonical: fastvideo.attention.backends.abstract.AttentionImpl.postprocess_output

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionImpl.postprocess_output
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} preprocess_qkv(qkv: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.T) -> torch.Tensor
:canonical: fastvideo.attention.backends.abstract.AttentionImpl.preprocess_qkv

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionImpl.preprocess_qkv
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} AttentionLayer
:canonical: fastvideo.attention.backends.abstract.AttentionLayer

Bases: {py:obj}`typing.Protocol`

````{py:method} forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, attn_metadata: fastvideo.attention.backends.abstract.AttentionMetadata) -> torch.Tensor
:canonical: fastvideo.attention.backends.abstract.AttentionLayer.forward

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionLayer.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} AttentionMetadata
:canonical: fastvideo.attention.backends.abstract.AttentionMetadata

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadata
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} asdict_zerocopy(skip_fields: set[str] | None = None) -> dict[str, typing.Any]
:canonical: fastvideo.attention.backends.abstract.AttentionMetadata.asdict_zerocopy

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadata.asdict_zerocopy
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} current_timestep
:canonical: fastvideo.attention.backends.abstract.AttentionMetadata.current_timestep
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadata.current_timestep
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} AttentionMetadataBuilder()
:canonical: fastvideo.attention.backends.abstract.AttentionMetadataBuilder

Bases: {py:obj}`abc.ABC`, {py:obj}`typing.Generic`\[{py:obj}`fastvideo.attention.backends.abstract.T`\]

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadataBuilder
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadataBuilder.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} build(**kwargs: dict[str, typing.Any]) -> fastvideo.attention.backends.abstract.AttentionMetadata
:canonical: fastvideo.attention.backends.abstract.AttentionMetadataBuilder.build
:abstractmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadataBuilder.build
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} prepare() -> None
:canonical: fastvideo.attention.backends.abstract.AttentionMetadataBuilder.prepare
:abstractmethod:

```{autodoc2-docstring} fastvideo.attention.backends.abstract.AttentionMetadataBuilder.prepare
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:data} T
:canonical: fastvideo.attention.backends.abstract.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} fastvideo.attention.backends.abstract.T
:parser: docs.source.autodoc2_docstring_parser
```

````
