import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from flood_area_segmentation.callbacks.plots_drawer import PlotMetricsCallback


def get_git_commit_id() -> str:
    """Получает идентификатор текущего git коммита.

    Returns:
        Короткий hash коммита или 'unknown' если не удалось получить.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def export_to_onnx(
    model: pl.LightningModule, cfg: DictConfig, ckpt_path: Path, outputs_dir: Path
) -> None:
    """Экспортирует модель в формат ONNX.

    Args:
        model: Обученная модель PyTorch Lightning.
        cfg: Конфигурация Hydra.
        ckpt_path: Путь к чекпоинту модели.
        outputs_dir: Директория для сохранения выходных файлов.
    """
    model_class = type(model)
    best_model = model_class.load_from_checkpoint(ckpt_path, weights_only=False)
    best_model.eval()
    best_model.cpu()

    dummy_input = torch.randn(
        1, 3, cfg.datamodule.image_height, cfg.datamodule.image_width
    )

    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    onnx_path = hydra_output_dir / "model.onnx"

    torch.onnx.export(
        best_model.model,
        dummy_input,
        str(onnx_path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print(f"ONNX модель сохранена: {onnx_path}")

    symlink_path = outputs_dir / "model.onnx"
    symlink_path.unlink(missing_ok=True)
    symlink_path.symlink_to(onnx_path)


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="train",
)
def train_main(cfg: DictConfig) -> None:
    """Главная функция обучения модели.

    Args:
        cfg: Конфигурация Hydra.
    """
    pl.seed_everything(cfg.seed, workers=True)

    print(OmegaConf.to_yaml(cfg))

    git_commit = get_git_commit_id()
    print(f"Git commit: {git_commit}")

    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / cfg.paths.outputs_dir
    plots_dir = project_root / cfg.paths.plots_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    run_name = (
        f"{cfg.model.encoder_name}_"
        f"lr{cfg.model.learning_rate}_"
        f"bs{cfg.datamodule.batch_size}_"
        f"pw{cfg.model.pos_weight}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logger = hydra.utils.instantiate(cfg.logger, run_name=run_name)

    if hasattr(logger, "log_hyperparams"):
        params = OmegaConf.to_container(cfg, resolve=True)
        params["git_commit"] = git_commit
        logger.log_hyperparams(params)

    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=hydra_output_dir,
        filename="best-{epoch:02d}-{val_iou:.4f}",
        save_top_k=1,
        save_last=True,
        monitor="val_iou",
        mode="max",
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        verbose=True,
    )

    plots_callback = PlotMetricsCallback(out_dir=str(plots_dir))

    callbacks = [checkpoint_callback, early_stopping_callback, plots_callback]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(hydra_output_dir),
        **cfg.trainer,
    )

    trainer.fit(model, datamodule)

    best_ckpt_path = Path(checkpoint_callback.best_model_path)
    if best_ckpt_path.exists() and best_ckpt_path.is_file():
        symlink_path = outputs_dir / "best.ckpt"
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(best_ckpt_path)
        print(f"Лучший чекпоинт: {best_ckpt_path}")

    last_ckpt_path = hydra_output_dir / "last.ckpt"
    if last_ckpt_path.exists() and last_ckpt_path.is_file():
        symlink_last = outputs_dir / "last.ckpt"
        symlink_last.unlink(missing_ok=True)
        symlink_last.symlink_to(last_ckpt_path)

    if (
        cfg.production.export_onnx
        and best_ckpt_path.exists()
        and best_ckpt_path.is_file()
    ):
        export_to_onnx(model, cfg, best_ckpt_path, outputs_dir)
