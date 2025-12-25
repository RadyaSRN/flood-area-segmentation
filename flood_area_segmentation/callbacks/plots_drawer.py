from pathlib import Path

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class PlotMetricsCallback(Callback):
    """Callback для сохранения графиков метрик после обучения.

    Извлекает историю метрик из MLflow и сохраняет графики для всех 8 метрик:
    - train_loss, val_loss
    - train_iou, val_iou
    - train_f1, val_f1
    - train_accuracy, val_accuracy

    Args:
        out_dir: Директория для сохранения графиков.
    """

    def __init__(self, out_dir: str = "./plots"):
        """Инициализирует callback.

        Args:
            out_dir: Директория для сохранения графиков.
        """
        super().__init__()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, trainer, pl_module) -> None:
        """Сохраняет графики метрик после завершения обучения.

        Args:
            trainer: Trainer PyTorch Lightning.
            pl_module: LightningModule модель.
        """
        logger = trainer.logger

        try:
            run_id = logger.run_id
            client = logger.experiment
        except AttributeError:
            print(
                "[PlotMetricsCallback] Warning: MLFlowLogger не используется, "
                "отрисовка графиков пропускается."
            )
            return

        try:
            run_name = client.get_run(run_id).data.tags.get("mlflow.runName")
        except AttributeError:
            return

        run_out_dir = self.out_dir / run_name if run_name else self.out_dir
        run_out_dir.mkdir(parents=True, exist_ok=True)

        metrics_config = {
            "train_loss": {"title": "Train Loss", "ylabel": "Loss", "xlabel": "Step"},
            "val_loss": {
                "title": "Validation Loss",
                "ylabel": "Loss",
                "xlabel": "Epoch",
            },
            "train_iou": {"title": "Train IoU", "ylabel": "IoU", "xlabel": "Step"},
            "val_iou": {"title": "Validation IoU", "ylabel": "IoU", "xlabel": "Epoch"},
            "train_f1": {"title": "Train F1 Score", "ylabel": "F1", "xlabel": "Step"},
            "val_f1": {
                "title": "Validation F1 Score",
                "ylabel": "F1",
                "xlabel": "Epoch",
            },
            "train_accuracy": {
                "title": "Train Accuracy",
                "ylabel": "Accuracy",
                "xlabel": "Step",
            },
            "val_accuracy": {
                "title": "Validation Accuracy",
                "ylabel": "Accuracy",
                "xlabel": "Epoch",
            },
        }

        for metric_name, config in metrics_config.items():
            try:
                history = client.get_metric_history(run_id, key=metric_name)
                if not history:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                values = [metric.value for metric in history]
                steps = [metric.step for metric in history]

                ax.plot(steps, values, label=metric_name, linewidth=2)
                ax.set_xlabel(config["xlabel"])
                ax.set_ylabel(config["ylabel"])
                ax.set_title(config["title"])
                ax.legend()
                ax.grid(True, alpha=0.3)

                fig.savefig(run_out_dir / f"{metric_name}.png", dpi=150)
                plt.close(fig)

            except Exception as error:
                print(
                    f"[PlotMetricsCallback] Ошибка при отрисовке {metric_name}: {error}"
                )

        self._plot_comparison(
            client, run_id, run_out_dir, "loss", "Loss", ["train_loss", "val_loss"]
        )
        self._plot_comparison(
            client, run_id, run_out_dir, "iou", "IoU", ["train_iou", "val_iou"]
        )
        self._plot_comparison(
            client, run_id, run_out_dir, "f1", "F1 Score", ["train_f1", "val_f1"]
        )
        self._plot_comparison(
            client,
            run_id,
            run_out_dir,
            "accuracy",
            "Accuracy",
            ["train_accuracy", "val_accuracy"],
        )

        print(f"[PlotMetricsCallback] Графики сохранены в: {run_out_dir}")

    def _plot_comparison(
        self,
        client,
        run_id: str,
        out_dir: Path,
        name: str,
        ylabel: str,
        metric_names: list[str],
    ) -> None:
        """Создаёт сводный график для сравнения train и val метрик.

        Args:
            client: MLflow client.
            run_id: ID запуска MLflow.
            out_dir: Директория для сохранения.
            name: Имя графика.
            ylabel: Подпись оси Y.
            metric_names: Список имён метрик для отрисовки.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False

            for metric_name in metric_names:
                history = client.get_metric_history(run_id, key=metric_name)
                if history:
                    values = [m.value for m in history]
                    steps = [m.step for m in history]
                    label = "Train" if "train" in metric_name else "Validation"
                    ax.plot(steps, values, label=label, linewidth=2)
                    has_data = True

            if has_data:
                ax.set_xlabel("Step / Epoch")
                ax.set_ylabel(ylabel)
                ax.set_title(f"Train vs Validation {ylabel}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(out_dir / f"{name}_comparison.png", dpi=150)

            plt.close(fig)

        except Exception as error:
            print(
                f"[PlotMetricsCallback] Ошибка при отрисовке {name}_comparison: {error}"
            )
