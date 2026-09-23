from dreamverse.generation_worker import _create_generation_backend
from dreamverse.ltx2_generation import LTX2GenerationBackend


def test_create_generation_backend_ltx2_module_import():
    backend = _create_generation_backend("ltx2", gpu_id=3)

    assert isinstance(backend, LTX2GenerationBackend)
    assert backend.gpu_id == 3


def test_create_generation_backend_cosmos25_dfd_module_import():
    from dreamverse.cosmos25_dfd_generation import Cosmos25DFDGenerationBackend

    backend = _create_generation_backend("cosmos25_dfd", gpu_id=2)

    assert isinstance(backend, Cosmos25DFDGenerationBackend)
    assert backend.gpu_id == 2
