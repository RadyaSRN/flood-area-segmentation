from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from PIL import Image

from flood_area_segmentation.models.unet import UNet

TARGET_HEIGHT = 512
TARGET_WIDTH = 512
PIXEL_MAX = 255


@pytest.fixture
def tmp_image_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию с тестовыми изображениями и масками.

    Args:
        tmp_path: Временная директория pytest.

    Returns:
        Путь к директории с данными.
    """
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    for i in range(5):
        img_array = np.random.randint(
            0, 256, (TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8
        )
        mask_array = (
            np.random.randint(0, 2, (TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
            * PIXEL_MAX
        )

        Image.fromarray(img_array).save(images_dir / f"{i}.jpg")
        Image.fromarray(mask_array).save(masks_dir / f"{i}.png")

    return tmp_path


@pytest.fixture
def tmp_model() -> UNet:
    """Создаёт UNet модель с random weights для тестирования.

    Returns:
        Инициализированная модель UNet.
    """
    model = UNet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        learning_rate=0.001,
        pos_weight=1.0,
        threshold=0.5,
    )
    return model


@pytest.fixture
def tmp_checkpoint(tmp_path: Path, tmp_model: UNet) -> Path:
    """Сохраняет временный checkpoint модели в формате PyTorch Lightning.

    Args:
        tmp_path: Временная директория pytest.
        tmp_model: Модель для сохранения.

    Returns:
        Путь к checkpoint файлу.
    """
    ckpt_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": tmp_model.state_dict(),
            "hyper_parameters": dict(tmp_model.hparams),
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 0,
        },
        ckpt_path,
    )
    return ckpt_path


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Создаёт тестовое изображение.

    Args:
        tmp_path: Временная директория pytest.

    Returns:
        Путь к изображению.
    """
    img_path = tmp_path / "test_image.jpg"
    img_array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(img_path)
    return img_path


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Создаёт dummy input tensor для модели.

    Returns:
        Tensor формы [1, 3, TARGET_HEIGHT, TARGET_WIDTH].
    """
    return torch.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH)
