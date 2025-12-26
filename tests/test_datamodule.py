from pathlib import Path

import numpy as np
import torch

from flood_area_segmentation.data.datamodule import (
    FloodDataModule,
    FloodSegmentationDataset,
)

TARGET_HEIGHT = 512
TARGET_WIDTH = 512
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
SEED = 42


class TestFloodSegmentationDataset:
    """Тесты для датасета сегментации."""

    def test_dataset_length(self, tmp_image_dir: Path) -> None:
        """Проверяет, что длина датасета соответствует количеству файлов.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        images_dir = tmp_image_dir / IMAGES_SUBDIR
        masks_dir = tmp_image_dir / MASKS_SUBDIR

        image_paths = sorted(images_dir.glob("*.jpg"))
        mask_paths = [masks_dir / f"{p.stem}.png" for p in image_paths]

        dataset = FloodSegmentationDataset(image_paths, mask_paths)

        assert len(dataset) == 5, f"Ожидалось 5 элементов, получено {len(dataset)}"

    def test_dataset_returns_tensors(self, tmp_image_dir: Path) -> None:
        """Проверяет, что датасет возвращает тензоры.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        images_dir = tmp_image_dir / IMAGES_SUBDIR
        masks_dir = tmp_image_dir / MASKS_SUBDIR

        image_paths = sorted(images_dir.glob("*.jpg"))
        mask_paths = [masks_dir / f"{p.stem}.png" for p in image_paths]

        dataset = FloodSegmentationDataset(image_paths, mask_paths)
        image, mask = dataset[0]

        assert isinstance(image, (torch.Tensor, np.ndarray)), (
            "Изображение должно быть тензором или массивом"
        )
        assert isinstance(mask, torch.Tensor), "Маска должна быть тензором"

    def test_mask_binary(self, tmp_image_dir: Path) -> None:
        """Проверяет, что маска бинаризуется.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        images_dir = tmp_image_dir / IMAGES_SUBDIR
        masks_dir = tmp_image_dir / MASKS_SUBDIR

        image_paths = sorted(images_dir.glob("*.jpg"))
        mask_paths = [masks_dir / f"{p.stem}.png" for p in image_paths]

        dataset = FloodSegmentationDataset(image_paths, mask_paths)
        _, mask = dataset[0]

        unique_values = set(torch.unique(mask).tolist())
        assert unique_values.issubset({0.0, 1.0}), (
            f"Маска должна содержать только {{0.0, 1.0}}, получено {unique_values}"
        )

    def test_mask_shape(self, tmp_image_dir: Path) -> None:
        """Проверяет, что маска имеет правильную форму [1, H, W].

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        images_dir = tmp_image_dir / IMAGES_SUBDIR
        masks_dir = tmp_image_dir / MASKS_SUBDIR

        image_paths = sorted(images_dir.glob("*.jpg"))
        mask_paths = [masks_dir / f"{p.stem}.png" for p in image_paths]

        dataset = FloodSegmentationDataset(image_paths, mask_paths)
        _, mask = dataset[0]

        assert len(mask.shape) == 3, (
            f"Маска должна быть 3D [1, H, W], получено {len(mask.shape)}D"
        )
        assert mask.shape[0] == 1, (
            f"Первая размерность маски должна быть 1, получено {mask.shape[0]}"
        )


class TestFloodDataModule:
    """Тесты для DataModule."""

    def test_split_deterministic(self, tmp_image_dir: Path) -> None:
        """Проверяет, что split детерминирован при одном seed.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        dm1 = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            seed=SEED,
        )
        dm1.setup()

        dm2 = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            seed=SEED,
        )
        dm2.setup()

        train1 = [p.name for p in dm1.train_dataset.image_paths]
        train2 = [p.name for p in dm2.train_dataset.image_paths]

        assert train1 == train2, "При одинаковом seed train выборки должны совпадать"

    def test_split_different_seeds(self, tmp_image_dir: Path) -> None:
        """Проверяет, что разные seed дают разные split.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        dm1 = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            seed=SEED,
        )
        dm1.setup()

        dm2 = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            seed=SEED + 1,
        )
        dm2.setup()

        train1 = [p.name for p in dm1.train_dataset.image_paths]
        train2 = [p.name for p in dm2.train_dataset.image_paths]

        assert train1 != train2, "При разных seed train выборки должны отличаться"

    def test_val_split_ratio(self, tmp_image_dir: Path) -> None:
        """Проверяет, что val_split соответствует заданному.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        dm = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            val_split=0.2,
        )
        dm.setup()

        total = len(dm.train_dataset) + len(dm.val_dataset)
        val_ratio = len(dm.val_dataset) / total

        assert abs(val_ratio - 0.2) < 0.3, (
            f"val_split должен быть примерно 0.2, получено {val_ratio}"
        )

    def test_save_hyperparameters(self, tmp_image_dir: Path) -> None:
        """Проверяет, что гиперпараметры сохраняются.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        dm = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            batch_size=8,
            image_height=TARGET_HEIGHT,
            image_width=TARGET_WIDTH,
        )

        assert dm.hparams.batch_size == 8
        assert dm.hparams.image_height == TARGET_HEIGHT
        assert dm.hparams.image_width == TARGET_WIDTH

    def test_dataloader_batch_size(self, tmp_image_dir: Path) -> None:
        """Проверяет, что DataLoader возвращает правильный batch_size.

        Args:
            tmp_image_dir: Временная директория pytest.
        """
        dm = FloodDataModule(
            data_dir=str(tmp_image_dir),
            images_subdir=IMAGES_SUBDIR,
            masks_subdir=MASKS_SUBDIR,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        images, masks = batch

        assert images.shape[0] <= 2, (
            f"Batch size должен быть <= 2, получено {images.shape[0]}"
        )
