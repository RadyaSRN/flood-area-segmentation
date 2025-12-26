import enum
import shutil
import subprocess
import sys
from pathlib import Path

TRITON_REPO_DIRNAME = "triton_model_repository"
TRITON_MODEL_NAME = "flood_segmentation"
CONFIG_PBTXT_NAME = "config.pbtxt"
CONFIG_ONNX_NAME = "config_onnx.pbtxt"
CONFIG_TRT_NAME = "config_trt.pbtxt"
MODEL_ONNX_FILENAME = "model.onnx"
MODEL_TRT_FILENAME = "model.plan"


class ModelType(enum.Enum):
    """Тип модели для Triton Inference Server."""

    ONNX = "onnx"
    TRT = "trt"


def run_triton_server(
    model_type: ModelType,
    model_path: Path,
    container_name: str = "triton_flood_server",
    http_port: int = 8000,
    use_gpus: bool = True,
    triton_docker_version: str = "24.12-py3",
) -> None:
    """Запускает Triton Inference Server с указанной моделью.

    Args:
        model_type: Тип модели (onnx или trt).
        model_path: Путь к файлу модели.
        container_name: Имя Docker контейнера.
        http_port: HTTP порт для Triton.
        use_gpus: Использовать GPU.
        triton_docker_version: Версия Docker образа Triton.
    """
    if model_type not in (ModelType.ONNX, ModelType.TRT):
        print("Ошибка: Неподдерживаемый тип модели. Используйте 'onnx' или 'trt'.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parents[2]
    triton_repo = project_root / TRITON_REPO_DIRNAME

    model_dir = triton_repo / TRITON_MODEL_NAME / "1"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Ошибка: Файл модели не найден: {model_path}")
        sys.exit(1)

    if not model_path.is_file():
        print(f"Ошибка: {model_path} не является файлом.")
        sys.exit(1)

    if model_type == ModelType.ONNX:
        target_model = model_dir / MODEL_ONNX_FILENAME
        config_src = triton_repo / TRITON_MODEL_NAME / CONFIG_ONNX_NAME
    else:
        target_model = model_dir / MODEL_TRT_FILENAME
        config_src = triton_repo / TRITON_MODEL_NAME / CONFIG_TRT_NAME

    shutil.copy(model_path, target_model)

    config_dst = triton_repo / TRITON_MODEL_NAME / CONFIG_PBTXT_NAME
    if not config_src.exists():
        print(f"Ошибка: Конфиг не найден: {config_src}")
        sys.exit(1)
    shutil.copy(config_src, config_dst)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
    ]

    if use_gpus:
        cmd += ["--gpus", "all"]

    cmd += [
        "-p",
        f"{http_port}:8000",
        "-v",
        f"{triton_repo}:/models",
        f"nvcr.io/nvidia/tritonserver:{triton_docker_version}",
        "tritonserver",
        "--model-repository=/models",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\nОшибка при запуске Triton: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        print("\nTriton остановлен.")
