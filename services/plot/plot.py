import os
import time

import pandas as pd

# В контейнере обычно нет графического интерфейса.
# Поэтому используем неинтерактивный backend, чтобы matplotlib мог сохранять картинки в файл.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_env(name: str, default: str) -> str:
    """
    Читает переменную окружения и подставляет значение по умолчанию.
    """
    return os.getenv(name, default)


def main() -> None:
    metric_log_path = get_env("METRIC_LOG_PATH", "/app/logs/metric_log.csv")
    plot_path = get_env("PLOT_PATH", "/app/logs/error_distribution.png")

    plot_sleep_seconds = int(get_env("PLOT_SLEEP_SECONDS", "10"))
    hist_bins = int(get_env("HIST_BINS", "30"))

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    while True:
        try:
            # Если логов ещё нет, ждём следующей итерации.
            if not os.path.exists(metric_log_path):
                time.sleep(plot_sleep_seconds)
                continue

            df = pd.read_csv(metric_log_path)

            # Если данных нет (только заголовок), строить нечего.
            if df.empty or "absolute_error" not in df.columns:
                time.sleep(plot_sleep_seconds)
                continue

            errors = df["absolute_error"].dropna()
            if errors.empty:
                time.sleep(plot_sleep_seconds)
                continue

            # Строим гистограмму распределения абсолютных ошибок.
            plt.figure()
            plt.hist(errors, bins=hist_bins)
            plt.title("Error distribution (absolute_error)")
            plt.xlabel("absolute_error")
            plt.ylabel("count")
            plt.tight_layout()

            # Перезаписываем картинку при каждом обновлении.
            plt.savefig(plot_path)
            plt.close()

        except Exception:
            # В учебном проекте важно не падать, а продолжать попытки строить график.
            pass

        time.sleep(plot_sleep_seconds)


if __name__ == "__main__":
    main()
