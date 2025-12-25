import sys

import fire

from flood_area_segmentation.cli_impl.train import train_main


class Commands:
    """Класс с командами CLI для проекта.

    Предоставляет команды для обучения, инференса и других операций.
    """

    def train(self) -> None:
        """Запускает обучение модели.

        Использует Hydra для конфигурации. Параметры можно переопределять
        через аргументы командной строки в формате Hydra:
            python -m flood_area_segmentation.commands train model=mobilenet_unet
        """
        sys.argv.remove("train")
        train_main()


def main() -> None:
    """Главная функция - точка входа CLI."""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
