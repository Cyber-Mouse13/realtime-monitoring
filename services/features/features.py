import json
import os
import time
from datetime import datetime

import numpy as np
import pika
from sklearn.datasets import load_diabetes


def get_env(name: str, default: str) -> str:
    """
    Читает переменную окружения и подставляет значение по умолчанию.
    Это позволяет конфигурировать сервис через docker-compose.yml без правок кода.
    """
    return os.getenv(name, default)


def connect_with_retry(host: str, port: int, retries: int = 60, sleep_seconds: int = 2) -> pika.BlockingConnection:
    """
    Подключается к RabbitMQ с повторными попытками.

    Это нужно, потому что при запуске docker-compose брокер RabbitMQ может стартовать
    чуть позже, чем контейнер features попытается к нему подключиться.
    """
    last_error = None
    for _ in range(retries):
        try:
            params = pika.ConnectionParameters(host=host, port=port, heartbeat=60)
            return pika.BlockingConnection(params)
        except Exception as e:
            last_error = e
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Cannot connect to RabbitMQ: {last_error}")


def main() -> None:
    # Настройки подключения к RabbitMQ и имена очередей.
    rabbitmq_host = get_env("RABBITMQ_HOST", "localhost")
    rabbitmq_port = int(get_env("RABBITMQ_PORT", "5672"))

    features_queue = get_env("FEATURES_QUEUE", "X")
    y_true_queue = get_env("Y_TRUE_QUEUE", "y_true")

    # Задержка между итерациями генерации данных (по заданию поток должен быть наблюдаемым).
    sleep_seconds = int(get_env("SLEEP_SECONDS", "10"))

    # Загружаем готовый датасет из sklearn.
    # X: numpy array формы (n_samples, n_features)
    # y: numpy array формы (n_samples,)
    X, y = load_diabetes(return_X_y=True)

    # Подключаемся к RabbitMQ и объявляем очереди.
    connection = connect_with_retry(rabbitmq_host, rabbitmq_port)
    channel = connection.channel()
    channel.queue_declare(queue=features_queue, durable=True)
    channel.queue_declare(queue=y_true_queue, durable=True)

    rng = np.random.default_rng()

    # delivery_mode=2 делает сообщение persistent (RabbitMQ будет сохранять его на диск).
    props = pika.BasicProperties(delivery_mode=2)

    # Бесконечный цикл генерации сообщений.
    while True:
        # Случайно выбираем одно наблюдение.
        idx = int(rng.integers(low=0, high=len(X)))

        # Формируем уникальный идентификатор наблюдения.
        # По заданию используем timestamp текущего времени.
        message_id = datetime.timestamp(datetime.now())

        # Сообщение с признаками.
        message_features = {
            "id": message_id,
            "body": X[idx].tolist()
        }

        # Сообщение с истинным ответом.
        message_y_true = {
            "id": message_id,
            "body": float(y[idx])
        }

        # Публикуем признаки и истинный ответ в разные очереди.
        channel.basic_publish(
            exchange="",
            routing_key=features_queue,
            body=json.dumps(message_features).encode("utf-8"),
            properties=props
        )

        channel.basic_publish(
            exchange="",
            routing_key=y_true_queue,
            body=json.dumps(message_y_true).encode("utf-8"),
            properties=props
        )

        # Пауза между итерациями.
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
