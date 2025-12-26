# Сегментация затопленных участков при наводнениях

## Описание проекта

### Постановка задачи

Сегментация зон затопления по данным аэрофотосъемки для навигации спасательных операций и оценки ущерба.

#### Формат входных и выходных данных

Входные данные:

- Формат файла: Цветные `.rgb` или `.png` изображения.
- Препроцессинг: Изображения приводятся к фиксированному размеру `height x width`, значения цветов преобразуются в `[0, 1]`, затем значения стандартно нормализуются.
- Тензор модели: `[batch_size, 3, height, width]`.

Выходные данные:

- Тензор модели: `[batch_size, 1, height, width]`.
- Постпроцессинг: Сначала выходы модели приводятся к изначальным размерам изображений. Затем к выходам применяется пороговое преобразование (по умолчанию порог `0.5`, задается в конфиге) для получения бинарных масок (`0` — фон, `1` — вода).
- Формат файла: Одноканальные `.png` маски.

Height и width (по умолчанию оба `512`), а также средние и стандартные отклонения (по умолчанию установлены значения для ImageNet) для нормализации задаются в конфиге.

#### Метрики

В качестве метрик используются стандартные для задачи сегментации — IoU, F1-score (Dice coefficient) и Accuracy. Результаты:

| Model |        Encoder         | Val IoU | Val F1-score | Val Accuracy |
| :---- | :--------------------: | :-----: | :----------: | :----------: |
| U-Net |    ResNet34 (main)     |  0.82   |     0.90     |     0.92     |
| U-Net | MobileNetV2 (baseline) |  0.81   |     0.89     |     0.91     |

#### Валидация и тест

Так как данных не слишком много, тестовый набор не выделяется. Выделяются только обучающий и валидационный наборы, размер валидационного набора задается в конфиге (по умолчанию `20%`). Для детерминированности везде задается `SEED=42` (фиксируется через конфиг).

#### Датасет

Используется датасет [Flood Area Segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation/data) с Kaggle. В данном датасете есть только обучающая выборка, состоящая из `290` элементов. Она разбита на:

- Цветные изображения `.jpg` (их названия — числа) разных разрешений и разных соотношений сторон, которые находятся в папке `data/raw/images`. Папка весит `107.6 МБ`.
- Одноканальные маски `.png` (их названия — числа, соответствующие изображениям из `data/raw/images`), которые находятся в папке `data/raw/masks`. Папка весит `6.5 МБ`.

### Моделирование

#### Бейзлайн

Используется [Unet](https://smp.readthedocs.io/en/latest/models.html#id22) из [Segmentation Models PyTorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) с энкодером [mobilenet_v2](https://smp.readthedocs.io/en/latest/encoders.html) (предобучен на ImageNet). Суммарно `6.6M` обучаемых параметров.

#### Основная модель

Энкодер меняется на [resnet34](https://smp.readthedocs.io/en/latest/encoders.html), cуммарно `24.4M` обучаемых параметров. Гиперпараметры обучения для бейзлайна и основной модели одинаковые (задаются из конфига):

- **Лосс**: `BinaryCrossEntropy` с весом положительного класса (для решения проблемы дисбаланса классов), задаваемым в конфиге (по умолчанию `3.0`).
- **Оптимизатор**: `torch.optim.AdamW`.
- **Число эпох**: `100`, так как данных не так много.
- **Early stopping**: `15` эпох, отслеживается `val_iou`, чтобы избежать переобучения.
- `batch_size=16`.
- `lr=0.001`.
- **lr-шедулер**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=7)` — помогает бороться с плато при обучении.
- **Аугментации (из `Albumentations`)**: Случайные повороты, горизонтальные и вертикальные отражения, изменения перспективы, кропы, а также небольшие визуальные изменения (изменение яркости, контраста, насыщенности, добавление гауссовского и motion блюров).

### Внедрение

Поддерживается инференс из командной строки с использованием ONNX и инференс через Triton Inference Server с использованием ONNX и TensorRT.

## Техническая часть

### Используемые технологии

- uv
- pre-commit
- PyTorch Lightning
- DVC с использованием MinIO
- Hydra
- MLflow
- ONNX
- TensorRT
- Triton Inference Server

### Setup

1. Требования:

- Python версии 3.11 и выше
- Git
- CUDA (опционально, если планируется использование GPU)
- Docker (опционально, нужен для конвертации модели в TensorRT и использования Triton Inference Server)

2. Клонирование репозитория:

```bash
git clone https://github.com/RadyaSRN/flood-area-segmentation.git
cd flood-area-segmentation
```

Полную структуру репозитория можно посмотреть в конце `README.md`.

3. Установка `uv` и зависимостей:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

`--dev` нужен для `pytest` и `pre-commit`.

4. Настройка и запуск `pre-commit`:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

5. Настройка `dvc`:

Настройка credentials для доступа к MinIO:

```bash
uv run dvc remote modify minio access_key_id admin
uv run dvc remote modify minio secret_access_key MLOps_2025_fall_secret
```

6. Загрузка данных из `dvc`:

Можно отдельно загружать данные и модели, в зависимости от того, хотите ли вы делать обучение или просто инференс:

```bash
uv run dvc pull data/raw
uv run dvc pull models/resnet34_unet
uv run dvc pull models/mobilenet_unet
```

7. Опционально можно запустить тесты:

```bash
uv run pytest --verbose --cov=flood_area_segmentation --cov-report=term-missing
```

### Train

1. Запуск сервера MLflow для отслеживания обучения (из отдельной вкладки терминала):

```bash
uv run mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

2. Запуск обучения основной модели:

```bash
uv run flood-seg train
```

Hydra позволяет перенастраивать гиперпараметры из командной строки. Например, чтобы запустить обучение бейзлайн модели с другим размером батча:

```bash
uv run flood-seg train model=mobilenet_unet datamodule.batch_size=32
```

При первом запуске обучения данные сами загружаются из DVC. Обучение шло на NVIDIA RTX 4090 `6.5` минут для основной модели и `3.5` для бейзлайна.

За обучением можно следить через MLflow, открыв адрес из конфига в браузере (по умолчанию он http://127.0.1:8080). Для рана `run_name` в `plots/{run_name}` сохраняются графики метрик и лосса на валидации, на трейне, а также графики сравнения валидации и трейна.

По окончании обучения в папке `outputs/{date}/{time}` будут сохранены чекпоинты `last.ckpt`, `best.ckpt` (является симлинком на полное название, например, `best-epoch=26-val_iou=0.8075.ckpt`), а также папка `.hydra` с конфигами.

### Production preparation

Поддерживается конвертация `.ckpt` в ONNX через команду:

```bash
uv run flood-seg convert-to-onnx --ckpt=<path_to_ckpt> --onnx=<path_to_onnx_model> [--height <h>] [--width <w>]
```

Также поддерживается конвертация из ONNX в TensorRT:

```bash
uv run flood-seg convert-to-trt --onnx=<path_to_onnx_model> --trt=<path_to_trt_model> [--height <h>] [--width <w>] [--docker-version <version>]
```

### Infer

Есть два варианта инференса:

1. Через команду с использованием ONNX:

```bash
uv run flood-seg infer --onnx <path_to_onnx_model> --image <path_to_input_image> --out <path_to_output_image>
```

2. Через Triton Inference Server, он поддерживает и ONNX, и TensorRT. Сначала необходимо запустить его (из отдельной вкладки терминала):

```bash
uv run flood-seg triton-server --model_type <onnx/trt> --model-path <path_to_model> [--container-name <container_name>] [--http-port <http_port>] [--use-gpus <true/false>] [--docker-version <docker_version>]
```

Затем можно делать запрос через клиента:

```bash
uv run flood-seg triton-client --image <path_to_input_image> --out <path_to_output_image> [--triton-url <triton_url>] [--model-name <model_name>]
```

## Полная структура репозитория

```
flood-area-segmentation/
├── .dvc/                              # Конфигурация DVC
│   ├── .gitignore                     # Игнорируемые файлы DVC
│   └── config                         # Настройки remote storage (MinIO)
├── .dvcignore                         # Игнорируемые файлы для DVC
├── .github/
│   └── workflows/
│       └── ci.yml                     # CI пайплайн (pre-commit + pytest)
├── .gitignore                         # Игнорируемые файлы Git
├── .pre-commit-config.yaml            # Конфигурация pre-commit hooks
├── .python-version                    # Версия Python для uv
│
├── configs/                           # Hydra конфигурации
│   ├── train.yaml                     # Главный конфиг обучения
│   ├── infer.yaml                     # Конфиг для инференса
│   ├── datamodule/
│   │   └── flood_data.yaml            # Параметры данных и преобразований
│   ├── logger/
│   │   └── mlflow.yaml                # Настройки MLflow логгера
│   └── model/
│       ├── resnet34_unet.yaml         # Конфиг основной модели
│       └── mobilenet_unet.yaml        # Конфиг baseline модели
│
├── data/                              # Данные (управляются DVC)
│   ├── raw.dvc                        # DVC-файл для отслеживания данных
│   └── raw/
│       ├── images/                    # Исходные изображения (.jpg)
│       └── masks/                     # Маски сегментации (.png)
│
├── models/                            # Обученные модели (управляются DVC)
│   ├── resnet34_unet.dvc              # DVC-файл основной модели
│   ├── resnet34_unet/                 # Основная модель (ckpt, onnx, plan)
│   ├── mobilenet_unet.dvc             # DVC-файл baseline модели
│   └── mobilenet_unet/                # Baseline модель (ckpt, onnx, plan)
│
├── flood_area_segmentation/           # Основной Python-пакет
│   ├── __init__.py
│   ├── commands.py                    # CLI точка входа (fire)
│   ├── callbacks/
│   │   ├── __init__.py
│   │   └── plots_drawer.py            # Callback для сохранения графиков метрик
│   ├── cli_impl/
│   │   ├── __init__.py
│   │   ├── train.py                   # Логика обучения
│   │   ├── convert.py                 # Конвертация ckpt -> ONNX -> TensorRT
│   │   ├── infer.py                   # Инференс через ONNX Runtime
│   │   ├── triton_server.py           # Запуск Triton Inference Server
│   │   └── triton_client.py           # Клиент для Triton
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py              # PyTorch Lightning DataModule
│   └── models/
│       ├── __init__.py
│       └── unet.py                    # Модель UNet (LightningModule)
│
├── tests/                             # Тесты (pytest)
├── triton_model_repository/           # Репозиторий моделей Triton
│
├── pyproject.toml                     # Зависимости и метаданные проекта
├── uv.lock                            # Lock-файл зависимостей
└── README.md                          # Документация проекта
```
