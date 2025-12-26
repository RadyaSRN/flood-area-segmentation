import torch

from flood_area_segmentation.models.unet import UNet


class TestModelOutput:
    """Тесты выходных данных модели."""

    def test_output_shape(self, tmp_model: UNet, dummy_input: torch.Tensor) -> None:
        """Проверяет, что выход модели имеет правильную форму [B, 1, H, W].

        Args:
            tmp_model: Временная модель PyTorch.
            dummy_input: Входной тензор.
        """
        tmp_model.eval()
        with torch.no_grad():
            output = tmp_model(dummy_input)

        batch, channels, height, width = dummy_input.shape
        assert output.shape == (batch, 1, height, width), (
            f"Ожидалась форма ({batch}, 1, {height}, {width}), получена {output.shape}"
        )

    def test_output_dtype(self, tmp_model: UNet, dummy_input: torch.Tensor) -> None:
        """Проверяет, что выход модели имеет тип float32.

        Args:
            tmp_model: Временная модель PyTorch.
            dummy_input: Входной тензор.
        """
        tmp_model.eval()
        with torch.no_grad():
            output = tmp_model(dummy_input)

        assert output.dtype == torch.float32, (
            f"Ожидался dtype torch.float32, получен {output.dtype}"
        )

    def test_output_determinism(
        self, tmp_model: UNet, dummy_input: torch.Tensor
    ) -> None:
        """Проверяет, что модель в eval mode детерминирована.

        Args:
            tmp_model: Временная модель PyTorch.
            dummy_input: Входной тензор.
        """
        tmp_model.eval()
        with torch.no_grad():
            output1 = tmp_model(dummy_input)
            output2 = tmp_model(dummy_input)

        assert torch.allclose(output1, output2), (
            "Модель в eval mode должна давать идентичные результаты"
        )

    def test_output_different_sizes(self, tmp_model: UNet) -> None:
        """Проверяет, что модель работает с разными размерами входа.

        Args:
            tmp_model: Временная модель PyTorch.
        """
        tmp_model.eval()
        sizes = [(128, 128), (256, 256), (512, 512), (256, 384)]

        for height, width in sizes:
            dummy = torch.randn(1, 3, height, width)
            with torch.no_grad():
                output = tmp_model(dummy)

            assert output.shape == (1, 1, height, width), (
                f"Для входа ({height}, {width}) ожидался выход"
                f"(1, 1, {height}, {width}), получен {output.shape}"
            )

    def test_save_hyperparameters(self, tmp_model: UNet) -> None:
        """Проверяет, что гиперпараметры сохраняются в hparams.

        Args:
            tmp_model: Временная модель PyTorch.
        """
        assert hasattr(tmp_model, "hparams"), "Модель должна иметь атрибут hparams"
        assert tmp_model.hparams.encoder_name == "resnet34"
        assert tmp_model.hparams.learning_rate == 0.001
        assert tmp_model.hparams.threshold == 0.5

    def test_configure_optimizers(self, tmp_model: UNet) -> None:
        """Проверяет, что configure_optimizers возвращает правильную структуру.

        Args:
            tmp_model: Временная модель PyTorch.
        """
        result = tmp_model.configure_optimizers()

        assert "optimizer" in result, "Должен быть ключ 'optimizer'"
        assert "lr_scheduler" in result, "Должен быть ключ 'lr_scheduler'"

        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert result["lr_scheduler"]["monitor"] == "val_iou"
