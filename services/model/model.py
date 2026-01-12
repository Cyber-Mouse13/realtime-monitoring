import json
import os
import time

import numpy as np
import pika
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


def get_env(name: str, default: str) -> str:
    """
    Читает переменную окружения и подставляет значение по умолчанию.
    """
    return os.getenv(name, default)


def connect_with_retry(host: str, port: int, retries: int = 60, sleep_seconds: int = 2) -> pika.BlockingConnection:
    """
    Подключение к RabbitMQ с повторными попытками.
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
    rabbitmq_host = get_env("RABBITMQ_HOST", "localhost")
    rabbitmq_port = int(get_env("RABBITMQ_PORT", "5672"))

    features_queue = get_env("FEATURES_QUEUE", "X")
    y_pred_queue = get_env("Y_PRED_QUEUE", "y_pred")

    # Обучаем модель на готовом датасете sklearn.
    # Для учебного проекта это удобно: не нужно хранить модель как отдельный артефакт.
    X_train, y_train = load_diabetes(return_X_y=True)
    model = LinearRegression()
    model.fit(X_train, y_train)

    connection = connect_with_retry(rabbitmq_host, rabbitmq_port)
    channel = connection.channel()

    channel.queue_declare(queue=features_queue, durable=True)
    channel.queue_declare(queue=y_pred_queue, durable=True)

    # Ограничиваем число сообщений без подтверждения.
    channel.basic_qos(prefetch_count=1)

    props = pika.BasicProperties(delivery_mode=2)

    def on_message(ch, method, properties, body: bytes) -> None:
        """
        Обработчик сообщений с признаками.

        Формат входящего сообщения из очереди X:
        {
            "id": <timestamp>,
            "body": [f1, f2, f3]
        }

        Формат исходящего сообщения в очередь y_pred:
        {
            "id": <тот же id>,
            "body": <предсказание модели>
        }
        """
        try:
            message = json.loads(body.decode("utf-8"))
            message_id = message["id"]
            features = message["body"]

            x = np.array(features, dtype=float).reshape(1, -1)
            y_pred = float(model.predict(x)[0])

            message_y_pred = {
                "id": message_id,
                "body": y_pred
            }

            ch.basic_publish(
                exchange="",
                routing_key=y_pred_queue,
                body=json.dumps(message_y_pred).encode("utf-8"),
                properties=props
            )

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception:
            # В случае ошибки не подтверждаем сообщение, чтобы оно могло быть переотправлено.
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=features_queue, on_message_callback=on_message, auto_ack=False)
    channel.start_consuming()


if __name__ == "__main__":
    main()
