# Realtime Monitoring (sklearn.load_diabetes + AMQP + docker-compose)

Проект реализует упрощённый мониторинг качества модели в режиме "почти реального времени".
Все сервисы запускаются через docker-compose и обмениваются сообщениями через RabbitMQ (AMQP).

Источник данных:
- Сервис features использует готовый датасет `sklearn.datasets.load_diabetes`.

Модель:
- Сервис model обучает `LinearRegression` на том же датасете при старте контейнера.

## Архитектура
features ──▶ очередь X ──▶ model ──▶ очередь y_pred ──┐
features ──▶ очередь y_true ──────────────────────────┤
                                                       ▼
                                                   metric
                                                       │
                                                       ▼
                                                logs/metric_log.csv
                                                       │
                                                       ▼
                                                     plot
                                                       │
                                                       ▼
                                            logs/error_distribution.png

## Запуск
Требования:
- Docker
- Docker Compose (команда `docker compose`)

Команда:
docker compose up --build

## Проверка результата
tail -f logs/metric_log.csv
ls -lh logs/error_distribution.png

## RabbitMQ UI
http://localhost:15672
Логин/пароль: guest / guest
