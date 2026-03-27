from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # API
    app_name: str = "Fashion MNIST API"

    # Qdrant
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)

    # DVC / DagsHub
    dagshub_access_key: Optional[str] = None
    dagshub_secret_key: Optional[str] = None

    # Docker
    docker_image: str = "bd-lab-1-6:latest"

    class ConfigDict:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()