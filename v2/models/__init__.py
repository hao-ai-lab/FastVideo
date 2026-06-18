"""v2 model architectures (vendored from fastvideo/models/). Mirrors fastvideo's layout — ``dits/``,
``vaes/``, ``encoders/``, ``audio/``, ``upsamplers/`` — so a per-module cutover is a mechanical
``cp`` + ``sed 'fastvideo.'->'v2.'``. Submodule bodies currently start as re-export STUBS backed by
fastvideo (see memory: v2-vendoring-approach); nothing is imported eagerly here so ``import v2`` stays
torch-free — the stubs (which import fastvideo/torch) load only when the GPU backend references them."""
