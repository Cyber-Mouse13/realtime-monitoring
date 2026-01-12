import csv
import json
import os
import time
from typing import Dict, Optional, Any

import pika


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


def ensure_csv_with_header(path: str) -> None:
    """
    Создаёт CSV-файл и записывает заголовок, если файла ещё нет.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "y_true", "y_pred", "absolute_error"])
            f.flush()


def append_metric_row(path: str, row: Dict[str, Any]) -> None:
    """
    Дозаписывает одну строку метрик в CSV.
    flush() нужен, чтобы файл обновлялся в реальном времени.
    """
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row["id"], row["y_true"], row["y_pred"], row["absolute_error"]])
        f.flush()


def main() -> None:
    rabbitmq_host = get_env("RABBITMQ_HOST", "localhost")
    rabbitmq_port = int(get_env("RABBITMQ_PORT", "5672"))

    y_true_queue = get_env("Y_TRUE_QUEUE", "y_true")
    y_pred_queue = get_env("Y_PRED_QUEUE", "y_pred")

    metric_log_path = get_env("METRIC_LOG_PATH", "/app/logs/metric_log.csv")
    buffer_ttl_seconds = int(get_env("BUFFER_TTL_SECONDS", "600"))

    ensure_csv_with_header(metric_log_path)

    connection = connect_with_retry(rabbitmq_host, rabbitmq_port)
    channel = connection.channel()

    channel.queue_declare(queue=y_true_queue, durable=True)
    channel.queue_declare(queue=y_pred_queue, durable=True)

    # Буфер для сопоставления сообщений по id.
    buffer: Dict[str, Dict[str, Optional[float]]] = {}
    buffer_ts: Dict[str, float] = {}

    channel.basic_qos(prefetch_count=10)

    def try_finalize(message_id: str) -> None:
        """
        Если для одного id получены и y_true, и y_pred, то:
        - считаем absolute_error
        - пишем строку в CSV
        - удаляем запись из буфера
        """
        if message_id not in buffer:
            return

        y_true_val = buffer[message_id].get("y_true")
        y_pred_val = buffer[message_id].get("y_pred")

        if y_true_val is None or y_pred_val is None:
            return

        absolute_error = abs(y_true_val - y_pred_val)

        append_metric_row(metric_log_path, {
            "id": message_id,
            "y_true": y_true_val,
            "y_pred": y_pred_val,
            "absolute_error": absolute_error
        })

        del buffer[message_id]
        if message_id in buffer_ts:
            del buffer_ts[message_id]

    def cleanup_old_records() -> None:
        """
        Очищает записи из буфера, которые висят дольше TTL.
        Это предотвращает бесконечный рост памяти при потере сообщений.
        """
        now = time.time()
        to_delete = []
        for message_id, ts in buffer_ts.items():
            if now - ts > buffer_ttl_seconds:
                to_delete.append(message_id)

        for message_id in to_delete:
            if message_id in buffer:
                del buffer[message_id]
            del buffer_ts[message_id]

    def on_y_true(ch, method, properties, body: bytes) -> None:
        """
        Обработчик сообщений из очереди y_true.
        """
        try:
            msg = json.loads(body.decode("utf-8"))
            message_id = str(msg["id"])
            y_true_val = float(msg["body"])

            if message_id not in buffer:
                buffer[message_id] = {"y_true": None, "y_pred": None}
                buffer_ts[message_id] = time.time()

            buffer[message_id]["y_true"] = y_true_val

            try_finalize(message_id)
            cleanup_old_records()

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def on_y_pred(ch, method, properties, body: bytes) -> None:
        """
        Обработчик сообщений из очереди y_pred.
        """
        try:
            msg = json.loads(body.decode("utf-8"))
            message_id = str(msg["id"])
            y_pred_val = float(msg["body"])

            if message_id not in buffer:
                buffer[message_id] = {"y_true": None, "y_pred": None}
                buffer_ts[message_id] = time.time()

            buffer[message_id]["y_pred"] = y_pred_val

            try_finalize(message_id)
            cleanup_old_records()

            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=y_true_queue, on_message_callback=on_y_true, auto_ack=False)
    channel.basic_consume(queue=y_pred_queue, on_message_callback=on_y_pred, auto_ack=False)

    channel.start_consuming()


if __name__ == "__main__":
    main()
