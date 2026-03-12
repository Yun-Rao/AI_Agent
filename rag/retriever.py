"""
RAG 知识库检索模块
职责：
  - 管理独立的知识库 ChromaDB collection（与记忆库隔离）
  - 提供 PDF 导入入口（调用 pdf_loader 处理图文混合内容）
  - 提供语义检索接口，供 Agent Tool 调用
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    RAG_ENABLED,
    RAG_KNOWLEDGE_DIR,
    RAG_RECALL_K,
    RAG_RECALL_CANDIDATES,
    CHROMA_PERSIST_DIR,
)
from rag.reranker import get_reranker

logger = logging.getLogger(__name__)

# 知识库使用独立 collection，与记忆库（irene_memory_*）完全隔离
KNOWLEDGE_COLLECTION = "irene_knowledge"


def _build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class RAGRetriever:
    """
    RAG 知识库管理与检索。
    支持 PDF 导入（含图文混合内容），语义检索。
    """

    def __init__(self) -> None:
        if not RAG_ENABLED:
            logger.info("✦ RAG 未启用（RAG_ENABLED=false）")
            self._store = None
            return

        self._embeddings = _build_embeddings()
        self._store = Chroma(
            collection_name=KNOWLEDGE_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("✦ RAG 知识库初始化完成，collection=%s", KNOWLEDGE_COLLECTION)
        # PDF 自动导入改为在 app.py lifespan 中异步执行，不在此处阻塞

    # ── 导入接口 ────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str) -> int:
        """
        导入单个 PDF 文件（支持图文混合）。
        返回存入的 chunk 数量。
        已导入过的文件（根据 source metadata 去重）会跳过。
        """
        if not RAG_ENABLED or self._store is None:
            logger.warning("RAG 未启用，ingest 操作跳过")
            return 0

        # 检查是否已导入（避免重复）
        if self._is_ingested(pdf_path):
            logger.info("✦ PDF 已导入过，跳过：%s", pdf_path)
            return 0

        from rag.pdf_loader import PDFLoader
        loader = PDFLoader()
        chunks = loader.load(pdf_path)

        if not chunks:
            logger.warning("PDF 解析结果为空：%s", pdf_path)
            return 0

        self._store.add_documents(chunks)
        logger.info("✦ PDF 导入完成：%s，共 %d 个 chunk", pdf_path, len(chunks))
        return len(chunks)

    # ── 检索接口 ────────────────────────────────────────────

    def recall(self, query: str) -> List[dict]:
        """
        根据查询语句语义检索知识库。
        返回格式：[{"role": "system", "content": "【知识库参考】..."}]
        RAG 未启用或无结果时返回空列表。
        """
        if not RAG_ENABLED or self._store is None:
            return []

        try:
            # 第一阶段：向量召回较多候选（RAG_RECALL_CANDIDATES 条）
            candidates: List[Document] = self._store.similarity_search(
                query, k=RAG_RECALL_CANDIDATES
            )
        except Exception as e:
            logger.warning("RAG 检索失败：%s", e)
            return []

        if not candidates:
            return []

        # 去重：同一 source+page 只保留一个（避免图文都召回）
        candidates = self._deduplicate(candidates)

        # 第二阶段：Reranker 精排，取 top RAG_RECALL_K
        reranker = get_reranker()
        results = reranker.rerank(query, candidates, top_k=RAG_RECALL_K)

        knowledge_text = "\n---\n".join([
            f"[来源：{doc.metadata.get('source','未知')} 第{doc.metadata.get('page','')}页]\n{doc.page_content}"
            for doc in results
        ])

        summary = "【知识库参考资料，请结合角色风格自然融入回答，无需直接引用】\n" + knowledge_text
        logger.info("✦ RAG 召回 %d 个候选 → Reranker 精排后保留 %d 条，query=%s",
                    len(candidates), len(results), query[:20])
        return [{"role": "system", "content": summary}]

    def get_stats(self) -> dict:
        """返回知识库统计信息"""
        if not RAG_ENABLED or self._store is None:
            return {"enabled": False}
        count = self._store._collection.count()
        return {"enabled": True, "chunks": count, "collection": KNOWLEDGE_COLLECTION}

    # ── 私有方法 ────────────────────────────────────────────

    def _is_ingested(self, pdf_path: str) -> bool:
        """检查该文件是否已存入知识库（按 source metadata 判断）"""
        if self._store is None:
            return False
        try:
            results = self._store.similarity_search(
                "test", k=1,
                filter={"source": pdf_path}
            )
            return len(results) > 0
        except Exception:
            return False

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """
        同一页的文字和图片描述可能同时被召回，
        按 source+page 去重，优先保留文字 chunk。
        """
        seen = set()
        result = []
        # 先过一遍文字，再过图片（保证文字优先）
        for doc in sorted(docs, key=lambda d: d.metadata.get("content_type", "") == "image"):
            key = (doc.metadata.get("source", ""), doc.metadata.get("page", ""))
            if key not in seen:
                seen.add(key)
                result.append(doc)
        return result

    async def auto_ingest_async(self, knowledge_dir: str) -> None:
        """
        异步版自动扫描导入，在后台线程中执行，不阻塞主线程。
        每个 PDF 之间让出事件循环，保证服务启动后可以立即响应请求。
        """
        dir_path = Path(knowledge_dir)
        if not dir_path.exists():
            logger.info("✦ 知识库目录不存在，跳过自动导入：%s", knowledge_dir)
            return

        pdfs = list(dir_path.glob("**/*.pdf"))
        if not pdfs:
            logger.info("✦ 知识库目录暂无 PDF 文件：%s", knowledge_dir)
            return

        logger.info("✦ 后台导入：发现 %d 个 PDF，开始异步处理...", len(pdfs))
        for pdf in pdfs:
            # 在线程池中运行同步的 ingest_pdf，避免阻塞事件循环
            await asyncio.get_event_loop().run_in_executor(
                None, self.ingest_pdf, str(pdf)
            )
            # 每个 PDF 处理完后让出事件循环，保证其他请求可以正常处理
            await asyncio.sleep(0)

        logger.info("✦ 后台导入完成，共处理 %d 个 PDF", len(pdfs))

    def _auto_ingest(self, knowledge_dir: str) -> None:
        """启动时自动扫描目录，导入未处理的 PDF 文件（同步版，已弃用）"""
        dir_path = Path(knowledge_dir)
        if not dir_path.exists():
            logger.info("✦ 知识库目录不存在，跳过自动导入：%s", knowledge_dir)
            return

        pdfs = list(dir_path.glob("**/*.pdf"))
        if not pdfs:
            logger.info("✦ 知识库目录暂无 PDF 文件：%s", knowledge_dir)
            return

        logger.info("✦ 发现 %d 个 PDF，开始自动导入...", len(pdfs))
        for pdf in pdfs:
            self.ingest_pdf(str(pdf))


# ── 单例 ────────────────────────────────────────────────────
_rag_instance: RAGRetriever | None = None

def get_rag_retriever() -> RAGRetriever:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGRetriever()
    return _rag_instance