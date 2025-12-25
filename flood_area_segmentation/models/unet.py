import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, JaccardIndex


class UNet(pl.LightningModule):
    """UNet модель для бинарной сегментации.

    Использует segmentation-models-pytorch для создания UNet
    с предобученным энкодером.

    Args:
        encoder_name: Название энкодера ('resnet34', 'mobilenet_v2', и др.).
        encoder_weights: Веса для инициализации энкодера ('imagenet' или None).
        in_channels: Количество входных каналов.
        num_classes: Количество классов для сегментации.
        learning_rate: Скорость обучения.
        pos_weight: Вес положительного класса для BCE loss.
        threshold: Порог для бинаризации предсказаний.
        scheduler_factor: Коэффициент уменьшения lr для ReduceLROnPlateau.
        scheduler_patience: Patience для ReduceLROnPlateau.
        scheduler_mode: Режим шедулера ('min' или 'max').
        scheduler_monitor: Метрика для мониторинга шедулером.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        learning_rate: float = 0.001,
        pos_weight: float = 3.0,
        threshold: float = 0.5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 7,
        scheduler_mode: str = "max",
        scheduler_monitor: str = "val_iou",
    ):
        """Инициализирует модель UNet.

        Args:
            encoder_name: Название энкодера ('resnet34', 'mobilenet_v2', и др.).
            encoder_weights: Веса для инициализации энкодера ('imagenet' или None).
            in_channels: Количество входных каналов.
            num_classes: Количество классов для сегментации.
            learning_rate: Скорость обучения.
            pos_weight: Вес положительного класса для BCE loss.
            threshold: Порог для бинаризации предсказаний.
            scheduler_factor: Коэффициент уменьшения lr для ReduceLROnPlateau.
            scheduler_patience: Patience для ReduceLROnPlateau.
            scheduler_mode: Режим шедулера ('min' или 'max').
            scheduler_monitor: Метрика для мониторинга шедулером.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_iou = JaccardIndex(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_accuracy = Accuracy(task="binary")

        self.val_iou = JaccardIndex(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_accuracy = Accuracy(task="binary")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Прямой проход через модель.

        Args:
            images: Входной тензор изображений [B, C, H, W].

        Returns:
            Логиты сегментации [B, 1, H, W].
        """
        return self.model(images)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Общий шаг для обучения и валидации.

        Args:
            batch: Кортеж (изображения, маски).
            stage: Стадия ('train' или 'val').

        Returns:
            Значение функции потерь.
        """
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = (torch.sigmoid(logits) > self.hparams.threshold).int()
        masks_int = masks.int()

        if stage == "train":
            iou = self.train_iou(preds, masks_int)
            f1 = self.train_f1(preds, masks_int)
            accuracy = self.train_accuracy(preds, masks_int)
        else:
            iou = self.val_iou(preds, masks_int)
            f1 = self.val_f1(preds, masks_int)
            accuracy = self.val_accuracy(preds, masks_int)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_iou", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=False, on_epoch=True)

        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Шаг обучения.

        Args:
            batch: Кортеж (изображения, маски).
            batch_idx: Индекс батча.

        Returns:
            Значение функции потерь.
        """
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Шаг валидации.

        Args:
            batch: Кортеж (изображения, маски).
            batch_idx: Индекс батча.
        """
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> dict:
        """Конфигурирует оптимизатор и шедулер.

        Returns:
            Словарь с оптимизатором и шедулером.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.scheduler_mode,
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
