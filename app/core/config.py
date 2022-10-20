import secrets
from typing import Any, List, Union

from pydantic import AnyHttpUrl, BaseSettings, HttpUrl, validator


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    #LOCAL DEVELOPMENT
    SERVER_HOST: AnyHttpUrl = "http://127.0.0.1:8000/"
    #SERVER_HOST: AnyHttpUrl = "https://deilasbackendattempt.azurewebsites.net"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost", "https://localhost:5002", "https://localhost:5001", "https://deilafrontend.azurewebsites.net"]
    PROJECT_NAME: str = "DEILA"

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        case_sensitive = True

settings = Settings()
