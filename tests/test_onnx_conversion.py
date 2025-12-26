from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from flood_area_segmentation.cli_impl.convert import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    convert_checkpoint_to_onnx,
)
from flood_area_segmentation.models.unet import UNet

TARGET_HEIGHT = 512
TARGET_WIDTH = 512


class TestOnnxConversion:
    """Тесты конвертации в ONNX."""

    def test_onnx_file_created(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что ONNX файл создаётся.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        assert onnx_path.exists(), "ONNX файл должен существовать"

    def test_onnx_valid(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что ONNX модель проходит валидацию.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        model_onnx = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_onnx)

    def test_onnx_input_name(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что input имеет правильное имя.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        model_onnx = onnx.load(str(onnx_path))
        input_name = model_onnx.graph.input[0].name

        assert input_name == ONNX_INPUT_NAME, (
            f"Input должен называться '{ONNX_INPUT_NAME}', получено '{input_name}'"
        )

    def test_onnx_output_name(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что output имеет правильное имя.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        model_onnx = onnx.load(str(onnx_path))
        output_name = model_onnx.graph.output[0].name

        assert output_name == ONNX_OUTPUT_NAME, (
            f"Output должен называться '{ONNX_OUTPUT_NAME}', получено '{output_name}'"
        )

    def test_no_external_data_file(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что нет отдельного .onnx.data файла.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        data_file = onnx_path.with_suffix(".onnx.data")
        assert not data_file.exists(), "Не должно быть отдельного .onnx.data файла"

    def test_onnx_inference_runs(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что ONNX модель работает через ONNX Runtime.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        dummy_input = np.random.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH).astype(
            np.float32
        )

        outputs = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: dummy_input})

        assert len(outputs) == 1
        assert outputs[0].shape == (1, 1, TARGET_HEIGHT, TARGET_WIDTH)

    def test_pytorch_onnx_consistency(
        self, tmp_path: Path, tmp_model: UNet, tmp_checkpoint: Path
    ) -> None:
        """Проверяет, что PyTorch и ONNX дают близкие результаты.

        Args:
            tmp_path: Временная директория pytest.
            tmp_model: Временная модель PyTorch.
            tmp_checkpoint: Временный checkpoint PyTorch.
        """
        onnx_path = tmp_path / "model.onnx"

        convert_checkpoint_to_onnx(
            tmp_checkpoint, onnx_path, TARGET_HEIGHT, TARGET_WIDTH
        )

        dummy_input = np.random.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH).astype(
            np.float32
        )

        tmp_model.eval()
        with torch.no_grad():
            torch_output = tmp_model.model(torch.from_numpy(dummy_input)).numpy()

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_output = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: dummy_input})[0]

        assert np.allclose(torch_output, onnx_output, rtol=1e-2, atol=1e-2), (
            "PyTorch и ONNX должны давать близкие результаты"
        )
