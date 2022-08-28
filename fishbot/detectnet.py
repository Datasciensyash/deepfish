import pickle
import typing as tp
from pathlib import Path
import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from scipy import ndimage

from fishbot.abstract import VisionModel


class BobberDetector(VisionModel):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        size: tp.Tuple[int, int] = (224, 320),
        mask: tp.Optional[Path] = None
    ) -> None:
        super(BobberDetector, self).__init__()

        self.preprocess = A.Compose([A.Resize(*size), A.Normalize()])
        self._encoder_name = encoder_name
        self._mask = None

        if mask is not None:
            with mask.open('rb') as file:
                self._mask = pickle.load(file)
                self._mask = A.Resize(*size)(image=self._mask)['image']

        self._size = size
        self._model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=5,
            in_channels=3,
            classes=1,
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
    def from_checkpoint(cls, checkpoint_path: Path, mask: tp.Optional[Path] = None):
        state_dict = torch.load(checkpoint_path)
        detector = cls(state_dict["encoder_name"], state_dict["size"], mask=mask)
        detector.load_state_dict(state_dict["state_dict"])
        return detector

    def _inference(
        self, input_image: np.array, threshold: float = 0.3
    ) -> tp.Optional[tp.Tuple[int, int]]:
        # Get image size
        sizex, sizey = input_image.shape[0], input_image.shape[1]

        # Preprocess image
        input_image = self.preprocess(image=input_image)["image"]
        input_image = torch.Tensor(input_image).unsqueeze(0)
        input_image = input_image.permute(0, 3, 1, 2)

        # Get mask
        with torch.no_grad():
            output_mask = torch.sigmoid(self.forward(input_image)).squeeze(0).cpu().numpy()[0]

        # Modify mask if mask in inputs
        if self._mask is not None:
            output_mask = output_mask * self._mask

        # Apply threshold on mask
        _output_mask = output_mask > threshold
        if np.any(_output_mask):
            output_mask = _output_mask.astype(int)
        else:
            return None

        # post-process mask and get coordinates
        output_mask, nr_objects = ndimage.label(output_mask)

        output_mask = output_mask == np.argmax(np.bincount(output_mask.flatten())[1:]) + 1
        xs, ys = np.nonzero(output_mask)
        coordinates = (
            int(np.mean(ys) / self._size[1] * sizey),
            int(np.mean(xs) / self._size[0] * sizex),
        )

        return coordinates
