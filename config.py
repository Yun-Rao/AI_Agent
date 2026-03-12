"""
统一配置文件
所有环境变量和常量在此处读取，其他模块从这里导入
"""

import os

# HuggingFace 镜像站，解决国内下载慢/失败的问题
# 若已在系统环境变量中设置 HF_ENDPOINT 则以系统变量为准
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 模型已缓存到本地后，禁止联网检查更新，避免每次调用都超时等待
# 首次下载模型时需要注释掉这两行，下载完成后再恢复
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ============================================================
# Kimi API
# ============================================================
KIMI_API_KEY: str = os.environ.get("MOONSHOT_API_KEY", "")
if not KIMI_API_KEY:
    raise RuntimeError(
        "\n========================================\n"
        "  未找到系统环境变量 KIMI_API_KEY ！\n"
        "========================================\n"
        "请在 Windows 环境变量中新建：\n"
        "  变量名: KIMI_API_KEY\n"
        "  变量值: sk-xxxxxxxxxxxxxxxxxxxxxxxx\n"
        "设置后请重新打开终端再启动服务。\n"
    )

KIMI_BASE_URL: str = "https://api.moonshot.cn/v1"
KIMI_MODEL: str = "kimi-k2-turbo-preview"
KIMI_TEMPERATURE: float = 0.8
KIMI_MAX_TOKENS: int = 1000
# kimi-k2-turbo-preview 原生支持视觉输入，直接复用主模型
KIMI_VISION_MODEL: str = "kimi-k2-turbo-preview"

# ============================================================
# Redis（短期记忆）
# ============================================================
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
SHORT_TERM_MAX_TURNS: int = int(os.environ.get("SHORT_TERM_MAX_TURNS", "10"))
SHORT_TERM_TTL: int = int(os.environ.get("SHORT_TERM_TTL", str(60 * 60 * 24)))

# ============================================================
# ChromaDB（长期记忆）
# ============================================================
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
LONG_TERM_MIGRATE_THRESHOLD: int = int(os.environ.get("LONG_TERM_MIGRATE_THRESHOLD", "10"))
LONG_TERM_RECALL_K: int = int(os.environ.get("LONG_TERM_RECALL_K", "3"))

# ============================================================
# MySQL（预留 - 暂未启用）
# 启用：设置环境变量 MYSQL_ENABLED=true 并填写 MYSQL_URL
# ============================================================
MYSQL_ENABLED: bool = os.environ.get("MYSQL_ENABLED", "false").lower() == "true"
MYSQL_URL: str = os.environ.get(
    "MYSQL_URL",
    "mysql+pymysql://user:password@localhost:3306/irene_ai"
)

# ============================================================
# RAG（预留 - 暂未启用）
# 启用：设置环境变量 RAG_ENABLED=true 并准备好知识库文件
# ============================================================
RAG_ENABLED: bool = os.environ.get("RAG_ENABLED", "true").lower() == "true"
RAG_KNOWLEDGE_DIR: str = os.environ.get("RAG_KNOWLEDGE_DIR", "./knowledge")
RAG_RECALL_K: int = int(os.environ.get("RAG_RECALL_K", "3"))        # 最终送入模型的条数
RAG_RECALL_CANDIDATES: int = int(os.environ.get("RAG_RECALL_CANDIDATES", "20"))  # 向量召回候选数量，Reranker 从中精排
RAG_RERANKER_MODEL: str = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")  # 重排序模型