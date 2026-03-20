# Fashion-MNIST ML Pipeline (CI/CD + DVC)

Пример полного жизненного цикла разработки модели машинного обучения:  
подготовка данных, обучение модели, тестирование, контейнеризация и автоматизация CI/CD.

Проект демонстрирует **MLOps pipeline** с использованием:
- FastAPI (API для инференса модели)
- DVC (версионирование данных и воспроизводимость)
- Docker (контейнеризация)
- GitHub Actions (CI/CD)
- pytest (тестирование)

---

# Dataset

Используется датасет **Fashion-MNIST**:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

Описание:

- 70 000 изображений одежды
- размер изображений: **28×28**
- grayscale
- **10 классов одежды**

Классы:
- 0 - T-shirt/top
- 1 - Trouser
- 2 - Pullover
- 3 - Dress
- 4 - Coat
- 5 - Sandal
- 6 - Shirt
- 7 - Sneaker
- 8 - Bag
- 9 - Ankle boot

Данные версионируются с помощью **DVC** и хранятся вне Git.

---

# DVC Remote Storage (DagsHub)

Для хранения данных и артефактов модели используется **DVC remote storage**, размещенный на платформе **DagsHub**.

DagsHub предоставляет S3-совместимое хранилище, которое позволяет:

- хранить большие файлы (датасеты и модели) вне Git
- версионировать данные вместе с кодом
- воспроизводить ML pipeline
- автоматически получать данные в CI/CD

В данном проекте DagsHub используется как **удалённое хранилище для DVC**.

### Использование DagsHub в CI

В CI pipeline доступ к хранилищу DagsHub настраивается через секреты GitHub:

```
DAGSHUB_ACCESS_KEY
DAGSHUB_SECRET_KEY
```

В pipeline выполняется настройка remote:

```yaml
dvc remote modify origin --local access_key_id ${{ secrets.DAGSHUB_ACCESS_KEY }}
dvc remote modify origin --local secret_access_key ${{ secrets.DAGSHUB_SECRET_KEY }}
```

После этого pipeline может скачать данные:

```
dvc pull
```

---

# CI/CD Pipeline

CI pipeline выполняет:

1. установку зависимостей
2. загрузку данных через DVC
3. воспроизводимое обучение модели
4. запуск тестов
5. сборку Docker образа
6. публикацию образа в DockerHub

CD pipeline:

1. запускает контейнер
2. выполняет функциональное тестирование API (`scenario.json`)

---

# API

После запуска доступен Swagger UI:

```

http://localhost:8000/docs

```

Эндпоинты:

| endpoint | описание |
|--------|--------|
| `/health` | проверка состояния сервиса |
| `/predict` | предсказание по массиву пикселей |
| `/predict/image` | предсказание по изображению |
| `/predict/random` | случайный тестовый пример |

---

# Установка и запуск

## Создание виртуального окружения

```bash
python -m venv venv
```

Активировать:

Linux / macOS

```bash
source venv/bin/activate
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск через Docker

Сборка образа:

```bash
docker build -t fashion-mnist-api .
```

Запуск контейнера:

```bash
docker run -p 8000:8000 fashion-mnist-api
```

## Docker Compose

```bash
docker-compose up --build
```

---

# Docker Image

Docker образ публикуется в DockerHub:

```
https://hub.docker.com/r/<username>/bd-lab-1-6
```

# DevSecOps Metadata

Pipeline генерирует файл:

```
dev_sec_ops.yml
```

Он содержит:

- Docker image
- последние коммиты
- результаты тестов

