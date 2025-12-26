import subprocess
import sys
from pathlib import Path

import onnx
import torch

from flood_area_segmentation.models.unet import UNet

ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAME = "output"
ONNX_OPSET_VERSION = 18


def convert_checkpoint_to_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    image_height: int = 512,
    image_width: int = 512,
) -> None:
    """Конвертирует checkpoint PyTorch Lightning в ONNX.

    Args:
        checkpoint_path: Путь к checkpoint модели.
        onnx_path: Путь для сохранения ONNX модели.
        image_height: Высота входного изображения.
        image_width: Ширина входного изображения.
    """
    if not checkpoint_path.exists():
        print(f"Ошибка: Checkpoint не найден: {checkpoint_path}")
        sys.exit(1)

    model = UNet.load_from_checkpoint(checkpoint_path, weights_only=False)
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, 3, image_height, image_width)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.model,
        dummy_input,
        str(onnx_path),
        opset_version=ONNX_OPSET_VERSION,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={
            ONNX_INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
            ONNX_OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
        },
    )

    model_onnx = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model_onnx, str(onnx_path))

    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()

    print(f"ONNX модель сохранена: {onnx_path}")


def convert_onnx_to_tensorrt(
    onnx_path: Path,
    trt_path: Path,
    image_height: int = 512,
    image_width: int = 512,
    trt_docker_version: str = "24.12-py3",
) -> None:
    """Конвертирует ONNX модель в TensorRT через Docker.

    Args:
        onnx_path: Путь к ONNX модели.
        trt_path: Путь для сохранения TensorRT engine.
        image_height: Высота входного изображения.
        image_width: Ширина входного изображения.
        trt_docker_version: Версия Docker образа TensorRT.
    """
    real_onnx_path = onnx_path.resolve()
    real_trt_path = trt_path.resolve()

    if not real_onnx_path.is_file():
        print(f"Ошибка: ONNX файл не найден: {real_onnx_path}")
        sys.exit(1)

    trt_dir = real_trt_path.parent
    trt_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = real_onnx_path.parent
    onnx_fname = real_onnx_path.name
    trt_fname = real_trt_path.name

    min_shape = f"1x3x{image_height}x{image_width}"
    opt_shape = f"1x3x{image_height}x{image_width}"
    max_shape = f"4x3x{image_height}x{image_width}"

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{onnx_dir}:/workspace/input:ro",
        "-v",
        f"{trt_dir}:/workspace/output",
        f"nvcr.io/nvidia/tensorrt:{trt_docker_version}",
        "trtexec",
        f"--onnx=/workspace/input/{onnx_fname}",
        f"--saveEngine=/workspace/output/{trt_fname}",
        f"--minShapes={ONNX_INPUT_NAME}:{min_shape}",
        f"--optShapes={ONNX_INPUT_NAME}:{opt_shape}",
        f"--maxShapes={ONNX_INPUT_NAME}:{max_shape}",
        "--fp16",
    ]

    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("\nОшибка при выполнении trtexec в Docker:", file=sys.stderr)
        sys.exit(exc.returncode)

    print(f"TensorRT engine создан: {real_trt_path}")
