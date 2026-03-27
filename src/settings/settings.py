from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # API
    app_name: str = "Fashion MNIST API"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # DagsHub
    dagshub_access_key: Optional[str] = None
    dagshub_secret_key: Optional[str] = None

    # Docker
    docker_image: str = "bd-lab-1-6:latest"

    model_config = {
        "env_file": ".env",
    }


settings = Settings()