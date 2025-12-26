import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig
from PIL import Image

ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAME = "output"
PIXEL_MAX = 255


def preprocess_image(
    image_path: Path,
    image_height: int,
    image_width: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Предобрабатывает изображение для инференса.

    Args:
        image_path: Путь к входному изображению.
        image_height: Высота для ресайза.
        image_width: Ширина для ресайза.
        mean: Средние значения для нормализации.
        std: Стандартные отклонения для нормализации.

    Returns:
        Кортеж из предобработанного тензора NCHW и оригинального размера.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение: {image_path}")
        sys.exit(1)

    original_size = (image.shape[1], image.shape[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_width, image_height))

    image = image.astype(np.float32) / PIXEL_MAX

    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean_arr) / std_arr

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return image.astype(np.float32), original_size


def postprocess_mask(
    logits: np.ndarray,
    original_size: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Постобрабатывает выход модели в бинарную маску.

    Args:
        logits: Выходные логиты модели формата NCHW.
        original_size: Размер для ресайза (width, height).
        threshold: Порог бинаризации.

    Returns:
        Бинарная маска (0 или 255).
    """
    probs = 1 / (1 + np.exp(-logits))
    mask = (probs > threshold).astype(np.uint8) * PIXEL_MAX
    mask = mask.squeeze()
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    return mask


def run_inference(
    cfg: DictConfig,
    onnx_path: Path,
    image_path: Path,
    output_path: Path,
) -> None:
    """Выполняет инференс на изображении через ONNX Runtime.

    Args:
        cfg: Конфигурация с параметрами препроцессинга.
        onnx_path: Путь к ONNX модели.
        image_path: Путь к входному изображению.
        output_path: Путь для сохранения маски.
    """
    if not onnx_path.exists():
        print(f"Ошибка: ONNX модель не найдена: {onnx_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"Ошибка: Изображение не найдено: {image_path}")
        sys.exit(1)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    mean = tuple(cfg.datamodule.mean)
    std = tuple(cfg.datamodule.std)

    input_tensor, original_size = preprocess_image(
        image_path,
        cfg.datamodule.image_height,
        cfg.datamodule.image_width,
        mean,
        std,
    )

    outputs = session.run([ONNX_OUTPUT_NAME], {ONNX_INPUT_NAME: input_tensor})
    mask = postprocess_mask(outputs[0], original_size, cfg.threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    print(f"Маска сохранена: {output_path}")
