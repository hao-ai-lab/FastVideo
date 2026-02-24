import os


def pytest_collection_modifyitems(config, items):
    """Optionally keep only tests with a matching model_id parameter."""
    model_id = os.environ.get("FASTVIDEO_SSIM_MODEL_ID")
    if not model_id:
        return

    selected = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            deselected.append(item)
            continue
        if callspec.params.get("model_id") == model_id:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
