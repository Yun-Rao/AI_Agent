"""
记忆管理器 - 统一调度短期（Redis）和长期（ChromaDB）记忆
同时提供 LangChain Tool 封装，供 Agent 按需调用
"""

import logging
from typing import List

from langchain_core.tools import Tool

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    统一记忆调度器。
    提供两种使用方式：
      1. 直接调用（get_context / save_turn）：用于每轮对话的基础上下文
      2. Tool 封装（as_tool）：供 Agent 按需主动检索记忆
    """

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    # ── 基础接口（每轮必调用）────────────────────────────────

    def get_short_term(self, session_id: str) -> List[dict]:
        """获取短期记忆（最近 N 轮），用于构建基础上下文"""
        return self.short_term.get(session_id)

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """
        保存本轮对话，自动处理溢出迁移：
          Redis 追加 → 检测溢出 → 迁移到 ChromaDB → 裁剪 Redis
        """
        self.short_term.append(session_id, "user", user_msg)
        self.short_term.append(session_id, "assistant", assistant_msg)

        overflow = self.short_term.get_overflow(session_id)
        if overflow:
            logger.info(
                "✦ session=%s 短期溢出 %d 条，迁移长期记忆",
                session_id, len(overflow)
            )
            self.long_term.save(session_id, overflow)
            self.short_term.trim(session_id)

    def clear(self, session_id: str) -> None:
        """清空该 session 所有记忆"""
        self.short_term.clear(session_id)
        self.long_term.clear(session_id)
        logger.info("✦ session=%s 所有记忆已清空", session_id)

    # ── Tool 封装（供 Agent 主动调用）────────────────────────

    def as_tool(self, session_id: str) -> Tool:
        """
        将长期记忆检索包装为 LangChain Tool。
        Agent 判断需要查找历史记忆时主动调用此工具。
        短期记忆（近期对话）已在 messages 里，不需要工具召回。
        """
        def _search_memory(query: str) -> str:
            results = self.long_term.recall(session_id, query)
            if not results:
                return "没有找到相关的历史记忆。"
            # recall 返回的是 [{"role": "system", "content": "..."}]
            return results[0]["content"]

        return Tool(
            name="search_memory",
            func=_search_memory,
            description=(
                "搜索伊蕾娜与用户过去的历史对话记忆。"
                "当用户提到'你还记得吗'、'我们之前聊过'、'你之前说过'等，"
                "或者需要回忆超过近期对话范围的历史内容时使用。"
                "输入：想要查找的历史话题或关键词。"
            ),
        )


# ── 单例 ────────────────────────────────────────────────────
_manager_instance: MemoryManager | None = None

def get_memory_manager() -> MemoryManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = MemoryManager()
    return _manager_instance