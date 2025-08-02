# {py:mod}`fastvideo.layers.vocab_parallel_embedding`

```{py:module} fastvideo.layers.vocab_parallel_embedding
```

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding
:parser: docs.source.autodoc2_docstring_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`UnquantizedEmbeddingMethod <fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`VocabParallelEmbedding <fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`VocabParallelEmbeddingShardIndices <fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_masked_input_and_mask <fastvideo.layers.vocab_parallel_embedding.get_masked_input_and_mask>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.get_masked_input_and_mask
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`pad_vocab_size <fastvideo.layers.vocab_parallel_embedding.pad_vocab_size>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.pad_vocab_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`vocab_range_from_global_vocab_size <fastvideo.layers.vocab_parallel_embedding.vocab_range_from_global_vocab_size>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.vocab_range_from_global_vocab_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
* - {py:obj}`vocab_range_from_per_partition_vocab_size <fastvideo.layers.vocab_parallel_embedding.vocab_range_from_per_partition_vocab_size>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.vocab_range_from_per_partition_vocab_size
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DEFAULT_VOCAB_PADDING_SIZE <fastvideo.layers.vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE>`
  - ```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE
    :parser: docs.source.autodoc2_docstring_parser
    :summary:
    ```
````

### API

````{py:data} DEFAULT_VOCAB_PADDING_SIZE
:canonical: fastvideo.layers.vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE
:value: >
   64

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.DEFAULT_VOCAB_PADDING_SIZE
:parser: docs.source.autodoc2_docstring_parser
```

````

`````{py:class} UnquantizedEmbeddingMethod
:canonical: fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod

Bases: {py:obj}`fastvideo.layers.quantization.base_config.QuantizeMethodBase`

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} apply(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor
:canonical: fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.apply

````

````{py:method} create_weights(layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int], input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs)
:canonical: fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.create_weights

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.create_weights
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} embedding(layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor
:canonical: fastvideo.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod.embedding

````

`````

`````{py:class} VocabParallelEmbedding(num_embeddings: int, embedding_dim: int, params_dtype: torch.dtype | None = None, org_num_embeddings: int | None = None, padding_size: int = DEFAULT_VOCAB_PADDING_SIZE, quant_config: fastvideo.layers.quantization.base_config.QuantizationConfig | None = None, prefix: str = '')
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding

Bases: {py:obj}`torch.nn.Module`

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding
:parser: docs.source.autodoc2_docstring_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.__init__
:parser: docs.source.autodoc2_docstring_parser
```

````{py:method} extra_repr() -> str
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.extra_repr

````

````{py:method} forward(input_)
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.forward
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} get_sharded_to_full_mapping() -> list[int] | None
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.get_sharded_to_full_mapping

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.get_sharded_to_full_mapping
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:method} weight_loader(param: torch.nn.parameter.Parameter, loaded_weight: torch.Tensor)
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.weight_loader

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbedding.weight_loader
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

`````{py:class} VocabParallelEmbeddingShardIndices
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices
:parser: docs.source.autodoc2_docstring_parser
```

````{py:attribute} added_vocab_end_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.added_vocab_end_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.added_vocab_end_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} added_vocab_start_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.added_vocab_start_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.added_vocab_start_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_added_elements
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_elements
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_elements
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_added_elements_padded
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_elements_padded
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_elements_padded
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_added_vocab_padding
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_vocab_padding
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_added_vocab_padding
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_elements_padded
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_elements_padded
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_elements_padded
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_org_elements
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_elements
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_elements
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_org_elements_padded
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_elements_padded
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_elements_padded
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:property} num_org_vocab_padding
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_vocab_padding
:type: int

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.num_org_vocab_padding
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} org_vocab_end_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.org_vocab_end_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.org_vocab_end_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} org_vocab_start_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.org_vocab_start_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.org_vocab_start_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} padded_added_vocab_end_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_added_vocab_end_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_added_vocab_end_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} padded_added_vocab_start_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_added_vocab_start_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_added_vocab_start_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} padded_org_vocab_end_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_org_vocab_end_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_org_vocab_end_index
:parser: docs.source.autodoc2_docstring_parser
```

````

````{py:attribute} padded_org_vocab_start_index
:canonical: fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_org_vocab_start_index
:type: int
:value: >
   None

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.VocabParallelEmbeddingShardIndices.padded_org_vocab_start_index
:parser: docs.source.autodoc2_docstring_parser
```

````

`````

````{py:function} get_masked_input_and_mask(input_: torch.Tensor, org_vocab_start_index: int, org_vocab_end_index: int, num_org_vocab_padding: int, added_vocab_start_index: int, added_vocab_end_index: int) -> tuple[torch.Tensor, torch.Tensor]
:canonical: fastvideo.layers.vocab_parallel_embedding.get_masked_input_and_mask

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.get_masked_input_and_mask
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int
:canonical: fastvideo.layers.vocab_parallel_embedding.pad_vocab_size

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.pad_vocab_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int, offset: int = 0) -> collections.abc.Sequence[int]
:canonical: fastvideo.layers.vocab_parallel_embedding.vocab_range_from_global_vocab_size

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.vocab_range_from_global_vocab_size
:parser: docs.source.autodoc2_docstring_parser
```
````

````{py:function} vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int, offset: int = 0) -> collections.abc.Sequence[int]
:canonical: fastvideo.layers.vocab_parallel_embedding.vocab_range_from_per_partition_vocab_size

```{autodoc2-docstring} fastvideo.layers.vocab_parallel_embedding.vocab_range_from_per_partition_vocab_size
:parser: docs.source.autodoc2_docstring_parser
```
````
