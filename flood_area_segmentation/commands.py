from datetime import datetime
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from flood_area_segmentation.cli_impl.convert import (
    convert_checkpoint_to_onnx,
    convert_onnx_to_tensorrt,
)
from flood_area_segmentation.cli_impl.infer import run_inference
from flood_area_segmentation.cli_impl.train import train_main
from flood_area_segmentation.cli_impl.triton_client import run_triton_client
from flood_area_segmentation.cli_impl.triton_server import ModelType, run_triton_server

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


def load_config(config_name: str, overrides: list[str] | None = None):
    """Загружает конфигурацию Hydra через Compose API.

    Args:
        config_name: Имя конфигурационного файла.
        overrides: Список переопределений в формате Hydra.

    Returns:
        Загруженная конфигурация.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


class Commands:
    """Класс с командами CLI для проекта сегментации затопленных территорий."""

    def train(self, *overrides: str) -> None:
        """Запускает обучение модели.

        Args:
            *overrides: Переопределения конфигурации в формате Hydra.

        Примеры:
            uv run flood-seg train
            uv run flood-seg train model=mobilenet_unet
            uv run flood-seg train trainer.max_epochs=50
        """
        cfg = load_config("train", list(overrides))

        project_root = Path(__file__).resolve().parents[1]
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        output_dir = project_root / "outputs" / timestamp

        train_main(cfg, output_dir)

    def infer(self, onnx: str, image: str, out: str) -> None:
        """Выполняет инференс на изображении через ONNX.

        Args:
            onnx: Путь к ONNX модели.
            image: Путь к входному изображению.
            out: Путь для сохранения маски.

        Примеры:
            uv run flood-seg infer --onnx model.onnx --image img.jpg --out mask.png
        """
        cfg = load_config("infer")
        run_inference(cfg, Path(onnx), Path(image), Path(out))

    def convert_to_onnx(
        self,
        ckpt: str,
        onnx: str,
        height: int = 512,
        width: int = 512,
    ) -> None:
        """Конвертирует checkpoint в ONNX.

        Args:
            ckpt: Путь к checkpoint модели.
            onnx: Путь для сохранения ONNX модели.
            height: Высота входного изображения.
            width: Ширина входного изображения.

        Примеры:
            uv run flood-seg convert-to-onnx --ckpt best.ckpt --onnx model.onnx
        """
        convert_checkpoint_to_onnx(Path(ckpt), Path(onnx), height, width)

    def convert_to_trt(
        self,
        onnx: str,
        trt: str,
        height: int = 512,
        width: int = 512,
        docker_version: str = "24.12-py3",
    ) -> None:
        """Конвертирует ONNX в TensorRT через Docker.

        Args:
            onnx: Путь к ONNX модели.
            trt: Путь для сохранения TensorRT engine.
            height: Высота входного изображения.
            width: Ширина входного изображения.
            docker_version: Версия Docker образа TensorRT.

        Примеры:
            uv run flood-seg convert-to-trt --onnx model.onnx --trt model.plan
        """
        convert_onnx_to_tensorrt(Path(onnx), Path(trt), height, width, docker_version)

    def triton_server(
        self,
        model_type: str,
        model_path: str,
        container_name: str = "triton_flood_server",
        http_port: int = 8000,
        use_gpus: bool = True,
        docker_version: str = "24.12-py3",
    ) -> None:
        """Запускает Triton Inference Server.

        Args:
            model_type: Тип модели ('onnx' или 'trt').
            model_path: Путь к файлу модели.
            container_name: Имя Docker контейнера.
            http_port: HTTP порт для Triton.
            use_gpus: Использовать GPU.
            docker_version: Версия Docker образа Triton.

        Примеры:
            uv run flood-seg triton-server --model_type onnx --model_path model.onnx
            uv run flood-seg triton-server --model_type trt --model_path model.plan
        """
        run_triton_server(
            model_type=ModelType(model_type),
            model_path=Path(model_path),
            container_name=container_name,
            http_port=http_port,
            use_gpus=use_gpus,
            triton_docker_version=docker_version,
        )

    def triton_client(
        self,
        image: str,
        out: str,
        triton_url: str = "localhost:8000",
        model_name: str = "flood_area_segmentation",
    ) -> None:
        """Отправляет изображение на Triton и сохраняет маску.

        Args:
            image: Путь к входному изображению.
            out: Путь для сохранения маски.
            triton_url: URL Triton сервера.
            model_name: Имя модели на сервере.

        Примеры:
            uv run flood-seg triton-client --image img.jpg --out mask.png
        """
        cfg = load_config("infer")
        run_triton_client(
            cfg=cfg,
            image_path=Path(image),
            output_path=Path(out),
            triton_url=triton_url,
            model_name=model_name,
        )


def main() -> None:
    """Главная функция - точка входа CLI."""
    fire.Fire(Commands)
