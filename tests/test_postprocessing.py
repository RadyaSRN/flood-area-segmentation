import numpy as np

from flood_area_segmentation.cli_impl.infer import PIXEL_MAX, postprocess_mask

TARGET_HEIGHT = 512
TARGET_WIDTH = 512
THRESHOLD = 0.5


class TestPostprocessing:
    """Тесты постпроцессинга масок."""

    def test_output_binary(self) -> None:
        """Проверяет, что выход содержит только 0 и 255."""
        logits = np.random.randn(1, 1, TARGET_HEIGHT, TARGET_WIDTH).astype(np.float32)
        mask = postprocess_mask(
            logits, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        unique_values = set(np.unique(mask))
        assert unique_values.issubset({0, PIXEL_MAX}), (
            f"Маска должна содержать только {{0, {PIXEL_MAX}}}, "
            f"получено {unique_values}"
        )

    def test_sigmoid_applied(self) -> None:
        """Проверяет, что sigmoid применяется к logits."""
        logits_positive = np.full(
            (1, 1, TARGET_HEIGHT, TARGET_WIDTH), 1.0, dtype=np.float32
        )
        mask = postprocess_mask(
            logits_positive, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        assert np.all(mask == PIXEL_MAX), (
            "При logits=1, sigmoid около 0.73 > threshold=0.5 -> маска должна быть 255"
        )

    def test_threshold_negative_logits(self) -> None:
        """Проверяет, что отрицательные logits дают маску 0."""
        logits = np.full((1, 1, TARGET_HEIGHT, TARGET_WIDTH), -10.0, dtype=np.float32)
        mask = postprocess_mask(
            logits, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        assert np.all(mask == 0), (
            "При logits=-10, sigmoid около 0, threshold=0.5 -> маска должна быть 0"
        )

    def test_threshold_positive_logits(self) -> None:
        """Проверяет, что большие положительные logits дают маску 255."""
        logits = np.full((1, 1, TARGET_HEIGHT, TARGET_WIDTH), 10.0, dtype=np.float32)
        mask = postprocess_mask(
            logits, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        assert np.all(mask == PIXEL_MAX), (
            "При logits=10, sigmoid около 1, threshold=0.5 -> маска должна быть 255"
        )

    def test_resize_to_original(self) -> None:
        """Проверяет ресайз маски к оригинальному размеру."""
        logits = np.random.randn(1, 1, TARGET_HEIGHT, TARGET_WIDTH).astype(np.float32)
        original_size = (480, 640)

        mask = postprocess_mask(logits, original_size, threshold=THRESHOLD)

        assert mask.shape == (640, 480), (
            f"Маска должна иметь размер (640, 480), получено {mask.shape}"
        )

    def test_threshold_boundary(self) -> None:
        """Проверяет граничное значение threshold."""
        logits = np.array([[[[0.0]]]]).astype(np.float32)

        mask_high_threshold = postprocess_mask(logits, (1, 1), threshold=0.6)
        assert mask_high_threshold[0, 0] == 0, (
            "При sigmoid(0)=0.5 и threshold=0.6 -> маска должна быть 0"
        )

        mask_low_threshold = postprocess_mask(logits, (1, 1), threshold=0.4)
        assert mask_low_threshold[0, 0] == PIXEL_MAX, (
            "При sigmoid(0)=0.5 и threshold=0.4 -> маска должна быть 255"
        )

    def test_output_dtype(self) -> None:
        """Проверяет, что выход имеет тип uint8."""
        logits = np.random.randn(1, 1, TARGET_HEIGHT, TARGET_WIDTH).astype(np.float32)
        mask = postprocess_mask(
            logits, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        assert mask.dtype == np.uint8, f"Ожидался dtype uint8, получен {mask.dtype}"

    def test_squeeze_batch_channel(self) -> None:
        """Проверяет, что batch и channel dimensions удаляются."""
        logits = np.random.randn(1, 1, TARGET_HEIGHT, TARGET_WIDTH).astype(np.float32)
        mask = postprocess_mask(
            logits, (TARGET_WIDTH, TARGET_HEIGHT), threshold=THRESHOLD
        )

        assert len(mask.shape) == 2, (
            f"Маска должна быть 2D, получено {len(mask.shape)}D"
        )
