import typing as tp
from pathlib import Path

import albumentations as A
import numpy as np
import timm
import torch

from fishbot.model.abstract import VisionModel


class ClassificationNet(VisionModel):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        size: tp.Tuple[int, int] = (64, 64),
        use_pretrained: bool = False,
    ) -> None:
        super(ClassificationNet, self).__init__()

        self.preprocess = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=size[0],
                    min_width=size[1],
                    border_mode=0,
                    value=(0, 0, 0),
                ),
                A.CenterCrop(*size),
                A.Normalize(),
            ]
        )
        self._encoder_name = encoder_name

        self._size = size
        self._model = timm.create_model(
            encoder_name, pretrained=use_pretrained, num_classes=1
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self._model(input_image)

    def state_dict(self, **kwargs) -> tp.Dict[str, tp.Any]:
        state_dict = super().state_dict(**kwargs)

        return {
            "state_dict": state_dict,
            "size": self._size,
            "encoder_name": self._encoder_name,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        state_dict = torch.load(checkpoint_path)
        detector = cls(state_dict["encoder_name"], state_dict["size"])
        detector.load_state_dict(state_dict["state_dict"])
        return detector

    def _inference(self, input_image: np.array, **kwargs) -> float:
        # Preprocess image
        input_image = self.preprocess(image=input_image)["image"]
        input_image = torch.Tensor(input_image).unsqueeze(0)
        input_image = input_image.permute(0, 3, 1, 2)

        # Get predictions
        with torch.no_grad():
            predicted = (
                torch.sigmoid(self.forward(input_image)[0]).numpy().flatten()[0]
            )

        return predicted
