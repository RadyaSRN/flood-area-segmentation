import pytest

from flood_area_segmentation.commands import load_config


class TestConfig:
    """Тесты конфигурации Hydra."""

    def test_load_train_config(self) -> None:
        """Проверяет, что train конфиг загружается без ошибок."""
        cfg = load_config("train")

        assert cfg is not None, "Конфиг не должен быть None"

    def test_train_config_required_fields(self) -> None:
        """Проверяет наличие обязательных полей в train конфиге."""
        cfg = load_config("train")

        assert "model" in cfg, "Конфиг должен содержать 'model'"
        assert "datamodule" in cfg, "Конфиг должен содержать 'datamodule'"
        assert "trainer" in cfg, "Конфиг должен содержать 'trainer'"
        assert "seed" in cfg, "Конфиг должен содержать 'seed'"

    def test_load_infer_config(self) -> None:
        """Проверяет, что infer конфиг загружается без ошибок."""
        cfg = load_config("infer")

        assert cfg is not None, "Конфиг не должен быть None"

    def test_infer_config_required_fields(self) -> None:
        """Проверяет наличие обязательных полей в infer конфиге."""
        cfg = load_config("infer")

        assert "datamodule" in cfg, "Конфиг должен содержать 'datamodule'"
        assert "threshold" in cfg, "Конфиг должен содержать 'threshold'"
        assert hasattr(cfg.datamodule, "mean"), "datamodule должен содержать 'mean'"
        assert hasattr(cfg.datamodule, "std"), "datamodule должен содержать 'std'"

    def test_override_model(self) -> None:
        """Проверяет, что override модели работает."""
        cfg = load_config("train", ["model=mobilenet_unet"])

        assert cfg.model.encoder_name == "mobilenet_v2", (
            "encoder_name должен быть 'mobilenet_v2' для mobilenet_unet"
        )

    def test_override_learning_rate(self) -> None:
        """Проверяет, что override learning_rate работает."""
        cfg = load_config("train", ["model.learning_rate=0.01"])

        assert cfg.model.learning_rate == 0.01, (
            "learning_rate должен быть 0.01 после override"
        )

    def test_override_batch_size(self) -> None:
        """Проверяет, что override batch_size работает."""
        cfg = load_config("train", ["datamodule.batch_size=32"])

        assert cfg.datamodule.batch_size == 32, (
            "batch_size должен быть 32 после override"
        )

    def test_datamodule_defaults(self) -> None:
        """Проверяет дефолтные значения datamodule."""
        cfg = load_config("train")

        assert cfg.datamodule.image_height == 512
        assert cfg.datamodule.image_width == 512
        assert cfg.datamodule.val_split == 0.2

    def test_model_defaults(self) -> None:
        """Проверяет дефолтные значения модели."""
        cfg = load_config("train")

        assert cfg.model.in_channels == 3
        assert cfg.model.num_classes == 1
        assert cfg.model.threshold == 0.5

    def test_invalid_model_raises(self) -> None:
        """Проверяет, что невалидная модель вызывает ошибку."""
        with pytest.raises(ValueError):
            load_config("train", ["model=nonexistent_model"])
