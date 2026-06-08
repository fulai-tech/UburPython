"""应用配置（pydantic-settings，读 .env）。

阈值类参数（SIM_THRESHOLD、TOP_K 等）集中在此，禁止散落魔法数（规范 §八）。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8080
    app_debug: bool = False

    es_node: str = "http://localhost:9200"
    es_audio_index: str = "audio_materials"
    es_tag_vectors_index: str = "tag_vectors"

    comm_grpc_host: str = "bionode-test.fulai.tech"
    comm_grpc_port: int = 443
    comm_grpc_use_tls: bool = True  # 443 走 TLS；内网明文可设 false

    sim_threshold: float = 0.7  # 内容形态向量模糊命中阈值（规范 §五-2）

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # 与 ES dense_vector.dims 一致

    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file_name: str = "uburnode.log"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"

    @property
    def comm_grpc_target(self) -> str:
        return f"{self.comm_grpc_host}:{self.comm_grpc_port}"

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
