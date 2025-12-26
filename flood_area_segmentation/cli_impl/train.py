import subprocess
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
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


def train_main(cfg: DictConfig, output_dir: Path) -> None:
    """Главная функция обучения модели.

    Args:
        cfg: Конфигурация Hydra.
        output_dir: Директория для выходных файлов.
    """
    pl.seed_everything(cfg.seed, workers=True)

    print(OmegaConf.to_yaml(cfg))

    git_commit = get_git_commit_id()
    print(f"Git commit: {git_commit}")

    project_root = Path(__file__).resolve().parents[2]
    plots_dir = project_root / cfg.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
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
        default_root_dir=str(output_dir),
        **cfg.trainer,
    )

    trainer.fit(model, datamodule)

    best_ckpt_path = Path(checkpoint_callback.best_model_path)
    if best_ckpt_path.exists() and best_ckpt_path.is_file():
        symlink_path = output_dir / "best.ckpt"
        symlink_path.unlink(missing_ok=True)
        symlink_path.symlink_to(best_ckpt_path)
        print(f"Лучший чекпоинт: {best_ckpt_path}")
        print(f"Симлинк: {symlink_path}")
