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

    # 功能手板 gRPC（与 HTTP 同进程；false 时跳过该端口）
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50050
    grpc_enabled: bool = True

    # 量产 gRPC
    somni_grpc_port: int = 50055
    somni_grpc_enabled: bool = True

    es_node: str = "http://localhost:9200"
    es_audio_index: str = "somni_audio_materials"
    es_tag_vectors_index: str = "somni_audio_tag_dictionary"
    es_connections_per_node: int = 50  # 异步连接池大小，高并发检索需调大
    es_request_timeout_sec: float = 30.0
    search_max_concurrency: int = 25  # 同时执行的检索流水线上限，防止打满 ES

    # 量产 ES（与手板向量/节点隔离）
    somni_es_node: str = ""
    somni_es_audio_index: str = "somni_audio_materials"
    somni_es_tag_vectors_index: str = "somni_audio_tag_dictionary"

    mongo_uri: str = ""
    mongo_db: str = "Fullive"
    mongo_materials_collection: str = "somni_audio_materials"
    mongo_tag_dictionary_collection: str = "somni_audio_tag_dictionary"

    # 量产 Mongo
    somni_mongo_uri: str = ""
    somni_mongo_db: str = "Somni"
    somni_mongo_materials_collection: str = "somni_audio_materials"
    somni_mongo_tag_dictionary_collection: str = "somni_audio_tag_dictionary"
    somni_mongo_answers_collection: str = "somni_quiz_answers"

    sim_threshold: float = 0.7  # 内容形态向量模糊命中阈值（规范 §五-2）
    # 多路文本检索厌恶硬剔除阈值；≥ 该值 penalty=1.0 丢弃候选
    strong_dislike_sim_threshold: float = 0.85
    search_sleep_stage_filter_enabled: bool = True  # 检索步骤 1 是否按睡眠阶段过滤
    es_dictionary_mget_batch_size: int = 500  # 标签词典向量批量读取的单批最大 ID 数

    embedding_backend: str = "onnx"  # onnx | torch | qwen_api（DashScope/OpenAI 兼容）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512  # 与 ES dense_vector.dims 一致
    embedding_onnx_dir: str = "models/onnx/bge-small-zh-v1.5"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_timeout_sec: float = 30.0
    embedding_api_batch_size: int = 10
    embedding_text_cache_size: int = 4096  # 请求侧文本→向量 LRU，降低重复 ONNX
    embedding_onnx_pool_size: int = 4  # ONNX InferenceSession 池大小，提高并发 encode
    embedding_onnx_intra_op_threads: int = 1  # 单 session 内线程数；池化后宜保持较小避免 CPU 过订阅

    log_level: str = "INFO"
    log_dir: str = "logs"
    # 按日命名 YYYY-MM-DD_ubur_log；超大小则同日再开新文件，见 app.core.logging
    log_rotation_size: str = "100 MB"
    log_retention: str = "7 days"

    # Mongo → ES 差异同步（服务内定时 + scripts/sync_es_from_comm.py 手动）
    sync_enabled: bool = True
    sync_interval_days: int = 7
    sync_page_size: int = 100
    sync_backup_dir: str = "data/sync_backup"
    sync_backup_filename: str = "somni_audio_materials_backup.json"
    sync_tag_dictionary_backup_filename: str = "somni_audio_tag_dictionary_backup.json"

    # 功能手板 Redis（空 URL 表示关闭）
    redis_url: str = ""
    # 量产 Redis（与手板隔离）
    somni_redis_url: str = ""
    # 连接池需覆盖 HTTP 并发峰值；redis-py 默认仅 100，高并发易 Too many connections
    redis_max_connections: int = 512
    search_cache_max_size: int = 2048
    search_cache_ttl_sec: int = 604800  # 7 天
    # CUD 后延时重建睡眠阶段候选缓存，窗口内多次写入只重建一次
    sleep_stage_cache_rewarm_delay_sec: float = 5.0

    default_page_size: int = 20
    max_page_size: int = 200
    fetch_all_hard_limit: int = 5000

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
    def log_dir_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def effective_somni_es_node(self) -> str:
        """量产 ES 节点；未配置时回退手板（仅本地兜底，生产应显式配置）。"""
        return self.somni_es_node or self.es_node


@lru_cache
def get_settings() -> Settings:
    return Settings()
