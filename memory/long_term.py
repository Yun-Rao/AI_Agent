"""
长期记忆模块 - 基于 ChromaDB
职责：
  - 将从 Redis 溢出的旧对话向量化后永久存储
  - 根据当前用户输入，语义召回相关的历史记忆
  - 数据默认持久化到磁盘，重启不丢失
"""

import logging
import uuid
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    LONG_TERM_RECALL_K,
)

logger = logging.getLogger(__name__)

# ── Embedding 模型 ──────────────────────────────────────────
# 使用本地 HuggingFace 模型，无需 API Key，首次运行自动下载
# 模型：BAAI/bge-base-zh-v1.5（支持中文，轻量快速）
# 如需替换为其他 embedding 服务，只需改这里
def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": "cpu"},
        # bge 模型官方推荐开启 normalize，有助于提升检索精度
        encode_kwargs={"normalize_embeddings": True},
    )


class LongTermMemory:
    """
    ChromaDB 长期记忆。
    Collection 命名规则：irene_memory_<session_id>
    每条记录存储一轮对话摘要，附带 metadata：
      - session_id
      - role_pair: "user | assistant"
      - turn_index: 第几轮
    """

    def __init__(self) -> None:
        try:
            self._embeddings: HuggingFaceEmbeddings = _build_embeddings()
            logger.info("✦ ChromaDB 长期记忆初始化，持久化目录：%s", CHROMA_PERSIST_DIR)
        except Exception as e:
            logger.error("ChromaDB 初始化失败：%s", e)
            raise RuntimeError(f"ChromaDB 初始化失败：{e}")

    # ── 公开接口 ──────────────────────────────

    def save(self, session_id: str, messages: List[dict]) -> None:
        """
        将一批消息存入长期记忆。
        messages 格式：[{"role": "user"|"assistant", "content": "..."}]
        相邻的 user+assistant 两条合并为一轮存储，减少 chunk 数量。
        """
        if not messages:
            return

        store = self._get_store(session_id)
        docs = []

        # 两两配对（user + assistant = 1轮）
        i = 0
        turn = 0
        while i < len(messages):
            user_msg = messages[i] if messages[i]["role"] == "user" else None
            asst_msg = messages[i + 1] if (i + 1 < len(messages) and messages[i + 1]["role"] == "assistant") else None

            if user_msg and asst_msg:
                combined = f"用户：{user_msg['content']}\n伊蕾娜：{asst_msg['content']}"
                docs.append(Document(
                    page_content=combined,
                    metadata={
                        "session_id": session_id,
                        "turn": turn,
                        "type": "dialogue",
                    }
                ))
                i += 2
            else:
                # 奇数条消息兜底：单独存
                msg = messages[i]
                docs.append(Document(
                    page_content=f"{msg['role']}：{msg['content']}",
                    metadata={
                        "session_id": session_id,
                        "turn": turn,
                        "type": "single",
                    }
                ))
                i += 1
            turn += 1

        # 为每个 doc 生成唯一 ID，防止重复存入
        ids = [str(uuid.uuid4()) for _ in docs]
        store.add_documents(docs, ids=ids)
        logger.info("✦ 长期记忆：session=%s 存入 %d 轮对话", session_id, len(docs))

    def recall(self, session_id: str, query: str) -> List[dict]:
        """
        根据当前用户输入，召回语义相关的历史记忆。
        返回格式：[{"role": "system", "content": "【相关历史记忆】..."}]
        若无相关记忆则返回空列表。
        """
        store = self._get_store(session_id)

        try:
            results = store.similarity_search(query, k=LONG_TERM_RECALL_K)
        except Exception as e:
            logger.warning("长期记忆召回失败：%s", e)
            return []

        if not results:
            return []

        # 将召回的历史整合为一条 system 消息注入上下文
        recalled_text = "\n---\n".join([doc.page_content for doc in results])
        summary = (
            "【来自长期记忆的相关历史对话，仅供参考，请自然融入回答，无需直接引用】\n"
            + recalled_text
        )

        logger.info("✦ 长期记忆：session=%s 召回 %d 条相关记录", session_id, len(results))
        return [{"role": "system", "content": summary}]

    def clear(self, session_id: str) -> None:
        """清除该 session 的全部长期记忆"""
        store = self._get_store(session_id)
        store.delete_collection()
        logger.info("✦ 长期记忆：session=%s 已清空", session_id)

    # ── 私有方法 ──────────────────────────────

    def _get_store(self, session_id: str) -> Chroma:
        """每个 session 使用独立的 collection"""
        return Chroma(
            collection_name=f"irene_memory_{session_id}",
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )