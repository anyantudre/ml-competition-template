"""Image classification, multilabel and regression with any timm backbone.

data.image_col holds file names (or paths) found in data.image_dir / data.test_image_dir.
model.backbone is a timm name: resnet18, efficientnet_b0, convnext_tiny, ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils import require, seed_everything
from .base import Pipeline


class ImagePipeline(Pipeline):
    def __init__(self, *args):
        super().__init__(*args)
        require("timm", "vision")
        self.backbone = self.cfg["model"]["backbone"] or "resnet18"
        d = self.cfg["data"]
        self.dirs = [d["image_dir"], d["test_image_dir"]]

    def _model(self, pretrained: bool):
        from ..models.vision import ImageClassifier

        return ImageClassifier(self.backbone, self.n_out, pretrained=pretrained, **self.cfg["model"]["params"])

    def _dataset(self, df, y, train: bool):
        from ..models.vision import ImageDataset, resolve_paths, transforms

        t = self.cfg["train"]
        paths = resolve_paths(df[self.cfg["data"]["image_col"]], self.dirs)
        return ImageDataset(paths, y, transforms(t["image_size"], train, t["augment"]), self.task)

    def fit_fold(self, train, y_train, valid, y_valid, test, fold):
        import torch

        from .. import engine

        seed = self.cfg["seed"] + fold
        seed_everything(seed)
        torch.manual_seed(seed)
        model = self._model(self.cfg["train"]["pretrained"])
        engine.fit(model, self._dataset(train, y_train, True), self._dataset(valid, y_valid, False),
                   tcfg=self.cfg["train"], task=self.task, score_fn=self.score_fn(y_valid),
                   greater_is_better=self.greater_is_better, seed=seed, log_prefix=f"  fold {fold} ")
        return {"model": model.cpu()}

    def predict(self, artifact, df) -> np.ndarray:
        from .. import engine

        tta = engine.flip_tta(self.task) if self.cfg["train"]["tta"] else None
        return engine.predict(artifact["model"], self._dataset(df, None, False), tcfg=self.cfg["train"],
                              task=self.task, tta=tta)

    def save(self, artifact, folder: Path):
        import torch

        folder.mkdir(parents=True, exist_ok=True)
        torch.save(artifact["model"].state_dict(), folder / "model.pt")

    def load(self, folder: Path):
        import torch

        model = self._model(pretrained=False)
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        return {"model": model}
