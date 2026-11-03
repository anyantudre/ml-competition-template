"""One pipeline per modality. Each fits one fold and predicts any DataFrame of the same shape."""

from __future__ import annotations


def get_pipeline(cfg: dict, task: str, target_enc, metric: str, columns):
    modality = cfg["modality"]
    if modality == "tabular":
        from .tabular import TabularPipeline as cls
    elif modality == "text":
        from .text import TextPipeline as cls
    elif modality == "image":
        from .vision import ImagePipeline as cls
    else:  # segmentation has its own loop: mlt.pipelines.segmentation.run_segmentation
        raise ValueError(f"no fold pipeline for modality '{modality}'")
    return cls(cfg, task, target_enc, metric, columns)
