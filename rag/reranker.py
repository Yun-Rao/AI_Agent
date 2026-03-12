"""
重排序模块 - 基于 BAAI/bge-reranker-base
职责：
  - 对向量召回的候选 chunk 进行精排
  - 使用交叉编码器直接对"问题+文档"打分，比向量相似度更精准
  - 首次使用自动下载模型（约 400MB），之后本地缓存

向量召回 vs 重排序的区别：
  - 向量召回：问题和文档分别编码成向量，计算相似度，速度快但精度一般
  - 重排序：问题和文档拼在一起输入模型打分，理解上下文关系，精度更高
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from langchain_core.documents import Document

from config import RAG_RERANKER_MODEL

logger = logging.getLogger(__name__)


class Reranker:
    """
    基于 CrossEncoder 的重排序器。
    懒加载：首次调用 rerank() 时才加载模型，不影响启动速度。
    """

    def __init__(self) -> None:
        self._model: Optional["CrossEncoder"] = None  # 懒加载

    def _load_model(self):
        """首次调用时加载 CrossEncoder 模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                RAG_RERANKER_MODEL,
                max_length=512,     # 超过 512 token 的文本截断
            )
            logger.info("✦ Reranker 模型加载完成：%s", RAG_RERANKER_MODEL)
        except Exception as e:
            logger.error("Reranker 模型加载失败：%s", e)
            raise RuntimeError(f"Reranker 加载失败：{e}")

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:
        """
        对候选文档重排序，返回得分最高的 top_k 个。

        参数：
          query：用户查询
          docs：向量召回的候选文档列表
          top_k：最终保留的文档数量

        返回：
          按相关性从高到低排列的 Document 列表（最多 top_k 个）
        """
        if not docs:
            return []

        # 候选数量不超过 top_k 时直接返回，不需要重排
        if len(docs) <= top_k:
            return docs

        self._load_model()
        assert self._model is not None

        # 构造 (query, doc_content) 对，CrossEncoder 接受这种格式
        pairs: List[Tuple[str, str]] = [
            (query, doc.page_content) for doc in docs
        ]

        try:
            scores: List[float] = self._model.predict(pairs).tolist()
        except Exception as e:
            logger.warning("Reranker 打分失败，降级为原始顺序：%s", e)
            return docs[:top_k]

        # 按得分降序排列，取 top_k
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored[:top_k]]

        logger.info(
            "✦ Reranker：%d 个候选 → top%d，最高分=%.4f 最低分=%.4f",
            len(docs), top_k,
            scored[0][0] if scored else 0,
            scored[top_k - 1][0] if len(scored) >= top_k else 0,
        )
        return top_docs


# ── 单例 ────────────────────────────────────────────────────
_reranker_instance: Reranker | None = None

def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance