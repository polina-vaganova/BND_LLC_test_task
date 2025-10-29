# BND_LLC — тестовое задание

## О проекте

Скрипт детектирует людей на видео и сохраняет аннотированное видео.
Выделенные объекты содержат название класса (person) и уверенность (confidence).

## Требования проекту

- pip
- GNU Make
- Python 3.8+

## Подготовка проекта

1) Пропишите команду: `make check`, чтобы проверить наличие нужных инструментов: *GNU Make*, *Python*.
2) Пропишите команду `make install`, чтобы создать виртуальное окружение *Python* и установить необходимые *Python* пакеты.

## Запуск программы

Производить из корня проекта.

1) Для начала активируйте виртуальное окружение командой: `.venv\Scripts\activate` для OS *Windows* или `source .venv/bin/activate` для OS *MacOS*/*Ubuntu*.
2) Комманда для запуска: `python src/main.py --input data/video/raw/crowd.mp4 --output data/video/processed/detecterd_crowd.mp4 --model yolov8n.pt --conf 0.25`

где:
    - `python` - Ваш интерпретатор *Python*.
    - `src\main.py` - точка запуска программы.
    - `--input data/video/raw/crowd.mp4` - входное видео для обработки.
    - `--output data/video/processed/detecterd_crowd.mp4` - обработанное видео.
    - `--model yolov8n.pt` - настроенная модель для обработки видео.
    - `--conf 0.25` - доверительный порог модели. Принимает значения от `0` до `1`.

## Очистка проекта

Для очистки проекта:

1) Выйдите из виртуалного окружения *Python* командой: `.venv\Scripts\deactivate` для OS *Windows* или `deactivate .venv` для OS *MacOS*/*Ubuntu*.
2) Использовать команду: `make clear`.
