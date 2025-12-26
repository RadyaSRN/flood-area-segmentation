from pathlib import Path

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from dvc.repo import Repo
from PIL import Image
from torch.utils.data import DataLoader, Dataset

STANDARD_BINARIZATION_THRESHOLD = 127


def ensure_data(project_root: Path, data_dir: str) -> None:
    """Проверяет наличие данных и скачивает их через DVC при необходимости.

    Args:
        project_root: Корневая директория проекта.
        data_dir: Относительный путь к директории с данными.
    """
    data_path = project_root / data_dir
    if not data_path.exists() or not any(data_path.iterdir()):
        with Repo(str(project_root)) as repo:
            repo.pull()


class FloodSegmentationDataset(Dataset):
    """Датасет для сегментации затопленных областей.

    Загружает изображения и маски, применяет трансформации.

    Args:
        image_paths: Список путей к изображениям.
        mask_paths: Список путей к маскам.
        transform: Трансформации albumentations для применения.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        transform: A.Compose | None = None,
    ):
        """Инициализирует датасет.

        Args:
            image_paths: Список путей к изображениям.
            mask_paths: Список путей к маскам.
            transform: Трансформации albumentations для применения.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        """Возвращает количество элементов в датасете."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Возвращает пару (изображение, маска) по индексу.

        Args:
            idx: Индекс элемента в датасете.

        Returns:
            Кортеж из тензора изображения и тензора маски.
        """
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > STANDARD_BINARIZATION_THRESHOLD).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask


class FloodDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule для датасета затопленных областей.

    Управляет загрузкой данных, разделением на train/val, трансформациями.

    Args:
        data_dir: Путь к директории с данными относительно корня проекта.
        images_subdir: Поддиректория с изображениями.
        masks_subdir: Поддиректория с масками.
        image_height: Высота изображения после ресайза.
        image_width: Ширина изображения после ресайза.
        batch_size: Размер батча.
        num_workers: Количество воркеров для DataLoader.
        val_split: Доля данных для валидации.
        mean: Средние значения для нормализации (RGB).
        std: Стандартные отклонения для нормализации (RGB).
        seed: Seed для воспроизводимости разделения данных.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        images_subdir: str = "images",
        masks_subdir: str = "masks",
        image_height: int = 512,
        image_width: int = 512,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.2,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        seed: int = 42,
    ):
        """Инициализирует DataModule.

        Args:
            data_dir: Путь к директории с данными относительно корня проекта.
            images_subdir: Поддиректория с изображениями.
            masks_subdir: Поддиректория с масками.
            image_height: Высота изображения после ресайза.
            image_width: Ширина изображения после ресайза.
            batch_size: Размер батча.
            num_workers: Количество воркеров для DataLoader.
            val_split: Доля данных для валидации.
            mean: Средние значения для нормализации (RGB).
            std: Стандартные отклонения для нормализации (RGB).
            seed: Seed для воспроизводимости разделения данных.
        """
        super().__init__()
        self.save_hyperparameters()

        self.project_root = Path(__file__).resolve().parents[2]
        self.data_path = self.project_root / data_dir
        self.images_dir = self.data_path / images_subdir
        self.masks_dir = self.data_path / masks_subdir

        self.train_dataset: FloodSegmentationDataset | None = None
        self.val_dataset: FloodSegmentationDataset | None = None

    def prepare_data(self) -> None:
        """Скачивает данные через DVC если они отсутствуют."""
        ensure_data(self.project_root, self.hparams.data_dir)

    def setup(self, stage: str | None = None) -> None:
        """Разделяет данные на train/val и создаёт датасеты.

        Args:
            stage: Стадия ('fit', 'validate', 'test', 'predict').
        """
        image_files = sorted(self.images_dir.glob("*.jpg"))
        mask_files = []

        for img_path in image_files:
            mask_path = self.masks_dir / f"{img_path.stem}.png"
            mask_files.append(mask_path)

        num_samples = len(image_files)
        indices = list(range(num_samples))

        rng = np.random.default_rng(self.hparams.seed)
        rng.shuffle(indices)

        split_idx = int(num_samples * (1 - self.hparams.val_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_images = [image_files[i] for i in train_indices]
        train_masks = [mask_files[i] for i in train_indices]
        val_images = [image_files[i] for i in val_indices]
        val_masks = [mask_files[i] for i in val_indices]

        self.train_dataset = FloodSegmentationDataset(
            image_paths=train_images,
            mask_paths=train_masks,
            transform=self._get_train_transform(),
        )

        self.val_dataset = FloodSegmentationDataset(
            image_paths=val_images,
            mask_paths=val_masks,
            transform=self._get_val_transform(),
        )

    def _get_train_transform(self) -> A.Compose:
        """Возвращает трансформации для обучающей выборки.

        Включает аугментации: повороты, отражения, изменения перспективы,
        кропы, изменение яркости, блюр.

        Returns:
            Композиция трансформаций albumentations.
        """
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(self.hparams.image_height, self.hparams.image_width),
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.Perspective(scale=(0.02, 0.05), p=0.2),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=10,
                            sat_shift_limit=20,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=0.2,
                ),
                A.Normalize(mean=self.hparams.mean, std=self.hparams.std),
                ToTensorV2(),
            ]
        )

    def _get_val_transform(self) -> A.Compose:
        """Возвращает трансформации для валидационной выборки.

        Только ресайз и нормализация, без аугментаций.

        Returns:
            Композиция трансформаций albumentations.
        """
        return A.Compose(
            [
                A.Resize(
                    height=self.hparams.image_height, width=self.hparams.image_width
                ),
                A.Normalize(mean=self.hparams.mean, std=self.hparams.std),
                ToTensorV2(),
            ]
        )

    def train_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для обучающей выборки."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для валидационной выборки."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
