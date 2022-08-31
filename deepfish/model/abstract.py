import abc
import os
import pickle
import time
import typing as tp
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from deepfish.constants import ENVIRON_SAVE_DATA_VARIABLE_NAME

VisionModelType = tp.TypeVar("VisionModelType")


class VisionModel(torch.nn.Module, abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_checkpoint(cls: tp.Type[VisionModelType], checkpoint_path: Path) -> VisionModelType:
        pass

    def _inference(self, input_image: np.ndarray, **kwargs) -> tp.Any:
        pass

    def inference(self, input_image: np.ndarray, **kwargs) -> tp.Any:
        outputs = self._inference(input_image, **kwargs)

        if not os.environ.get(ENVIRON_SAVE_DATA_VARIABLE_NAME, "False") != "True":
            return outputs

        save_location = Path(__file__).parent.parent / "logs" / self.__class__.__name__
        save_location.mkdir(exist_ok=True, parents=True)

        sample_prefix = int(time.time())
        Image.fromarray(input_image).save(save_location / f"{sample_prefix}.png")
        with (save_location / f"{sample_prefix}.pkl").open("wb") as file:
            pickle.dump(outputs, file)

        return outputs
