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
    es_audio_index: str = "somni_audio_materials"
    es_tag_vectors_index: str = "somni_audio_tag_dictionary"

    mongo_uri: str = ""
    mongo_db: str = "Fullive"
    mongo_materials_collection: str = "somni_audio_materials"
    mongo_tag_dictionary_collection: str = "somni_audio_tag_dictionary"

    comm_grpc_host: str = "bionode-test.fulai.tech"
    comm_grpc_port: int = 443
    comm_grpc_use_tls: bool = True  # 443 走 TLS；内网明文可设 false

    sim_threshold: float = 0.7  # 内容形态向量模糊命中阈值（规范 §五-2）
    search_sleep_stage_filter_enabled: bool = True  # 检索步骤 1 是否按睡眠阶段过滤

    embedding_backend: str = "onnx"  # onnx | torch | qwen_api（DashScope/OpenAI 兼容）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # 与 ES dense_vector.dims 一致
    embedding_onnx_dir: str = "models/onnx/bge-small-zh-v1.5"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_timeout_sec: float = 30.0
    embedding_api_batch_size: int = 10

    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file_name: str = "uburnode.log"
    log_rotation: str = "10 MB"
    log_retention: str = "7 days"

    # Mongo → ES 差异同步（服务内定时 + scripts/sync_es_from_comm.py 手动）
    sync_enabled: bool = True
    sync_interval_days: int = 7
    sync_page_size: int = 100
    sync_backup_dir: str = "data/sync_backup"
    sync_backup_filename: str = "somni_audio_materials_backup.json"
    sync_tag_dictionary_backup_filename: str = "somni_audio_tag_dictionary_backup.json"

    @property
    def embedding_onnx_path(self) -> Path:
        return Path(self.embedding_onnx_dir) / "model.onnx"

    @property
    def embedding_tokenizer_dir(self) -> Path:
        return Path(self.embedding_onnx_dir)

    @property
    def sync_backup_path(self) -> Path:
        return Path(self.sync_backup_dir) / self.sync_backup_filename

    @property
    def sync_tag_dictionary_backup_path(self) -> Path:
        return Path(self.sync_backup_dir) / self.sync_tag_dictionary_backup_filename

    @property
    def comm_grpc_target(self) -> str:
        return f"{self.comm_grpc_host}:{self.comm_grpc_port}"

    @property
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
