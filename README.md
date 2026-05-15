# ih-prep: Data Preparation for Information-Theoretic Analysis

Библиотека `ih-prep` превращает «сырой» pandas DataFrame в матрицу целочисленных имён, с которой работает вычислительное ядро IH-анализа ([`ih-lib`](https://github.com/AlGoldin87/ih-lib)).

**Никакого one-hot encoding. Никакого заполнения пропусков.**

Категории, числа и пропуски становятся просто целыми числами — равноправными и готовыми к анализу. Это первый и необходимый шаг в пайплайне IH-анализа. Узнать больше о методе можно в [статье на Хабре](https://habr.com/ru/articles/XXXXXX/).

## Установка

    pip install git+https://github.com/AlGoldin87/ih-prep.git

## Быстрый старт

    import pandas as pd
    from ih_prep import prepare_data

    # Загрузите ваш датасет
    df = pd.read_csv('your_data.csv')

    # Подготовьте данные
    data, info = prepare_data(df, target='target_column', sharpness=0.25)

    # data — это матрица int32, готовая для ih-lib
    # info — метаданные о подготовке

## Назначение

- **Дискретизация** количественных признаков на интервалы с заданной `резкостью (sharpness)`.
- **Кодирование** категориальных признаков в целочисленные имена.
- **Обработка пропусков** как отдельного, информативного значения.

## Экосистема IH-анализа

- **[`ih-lib`](https://github.com/AlGoldin87/ih-lib)** — вычислительное ядро (энтропия, R(Y|X)).
- **[`ih-coverage`](https://github.com/AlGoldin87/ih-coverage)** — автоматический подбор оптимальной резкости (ICC).
