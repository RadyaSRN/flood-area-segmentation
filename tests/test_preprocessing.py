from pathlib import Path

import numpy as np
from PIL import Image

from flood_area_segmentation.cli_impl.infer import preprocess_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TEST_NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
TEST_NORMALIZATION_STD = (0.5, 0.5, 0.5)
TARGET_HEIGHT = 512
TARGET_WIDTH = 512
PIXEL_MAX = 255


class TestPreprocessing:
    """Тесты препроцессинга изображений."""

    def test_output_shape(self, sample_image: Path) -> None:
        """Проверяет, что выход имеет правильную форму NCHW.

        Args:
            sample_image: Путь к тестовому изображению.
        """
        result, _ = preprocess_image(
            sample_image, TARGET_HEIGHT, TARGET_WIDTH, IMAGENET_MEAN, IMAGENET_STD
        )

        assert result.shape == (1, 3, TARGET_HEIGHT, TARGET_WIDTH), (
            f"Ожидалась форма (1, 3, {TARGET_HEIGHT}, {TARGET_WIDTH}), "
            f"получена {result.shape}"
        )

    def test_output_dtype(self, sample_image: Path) -> None:
        """Проверяет, что выход имеет тип float32.

        Args:
            sample_image: Путь к тестовому изображению.
        """
        result, _ = preprocess_image(
            sample_image, TARGET_HEIGHT, TARGET_WIDTH, IMAGENET_MEAN, IMAGENET_STD
        )

        assert result.dtype == np.float32, (
            f"Ожидался dtype float32, получен {result.dtype}"
        )

    def test_original_size_preserved(self, tmp_path: Path) -> None:
        """Проверяет, что оригинальный размер изображения сохраняется.

        Args:
            tmp_path: Временная директория pytest.
        """
        orig_w, orig_h = 640, 480
        img_path = tmp_path / "test.jpg"
        img_array = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        _, original_size = preprocess_image(
            img_path, TARGET_HEIGHT, TARGET_WIDTH, IMAGENET_MEAN, IMAGENET_STD
        )

        assert original_size == (orig_w, orig_h), (
            f"Ожидался размер ({orig_w}, {orig_h}), получен {original_size}"
        )

    def test_normalization_black_image(self, tmp_path: Path) -> None:
        """Проверяет нормализацию на чёрном изображении.

        Args:
            tmp_path: Временная директория pytest.
        """
        img_path = tmp_path / "black.jpg"
        img_array = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        result, _ = preprocess_image(
            img_path,
            TARGET_HEIGHT,
            TARGET_WIDTH,
            TEST_NORMALIZATION_MEAN,
            TEST_NORMALIZATION_STD,
        )

        expected = (0.0 - 0.5) / 0.5
        assert np.allclose(result, expected, atol=1e-5), (
            f"Для чёрного изображения с mean=0.5, std=0.5 ожидалось {expected}"
        )

    def test_normalization_white_image(self, tmp_path: Path) -> None:
        """Проверяет нормализацию на белом изображении.

        Args:
            tmp_path: Временная директория pytest.
        """
        img_path = tmp_path / "white.jpg"
        img_array = np.ones((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8) * 255
        Image.fromarray(img_array).save(img_path)

        result, _ = preprocess_image(
            img_path,
            TARGET_HEIGHT,
            TARGET_WIDTH,
            TEST_NORMALIZATION_MEAN,
            TEST_NORMALIZATION_STD,
        )

        expected = (1.0 - 0.5) / 0.5
        assert np.allclose(result, expected, atol=1e-5), (
            f"Для белого изображения с mean=0.5, std=0.5 ожидалось {expected}"
        )

    def test_resize_different_sizes(self, tmp_path: Path) -> None:
        """Проверяет ресайз для разных целевых размеров.

        Args:
            tmp_path: Временная директория pytest.
        """
        img_path = tmp_path / "test.jpg"
        img_array = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        sizes = [(128, 128), (256, 512), (64, 64)]

        for target_h, target_w in sizes:
            result, _ = preprocess_image(
                img_path, target_h, target_w, IMAGENET_MEAN, IMAGENET_STD
            )
            assert result.shape == (1, 3, target_h, target_w)
