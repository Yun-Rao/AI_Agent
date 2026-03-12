"""
短期记忆模块 - 基于 Redis
职责：
  - 存储最近 N 轮对话（可配置）
  - 超出上限时返回溢出记录，供迁移到长期记忆
  - 支持 TTL 过期（默认 24 小时不活跃则清除）
"""

import json
import logging
from typing import List

import redis
from redis import Redis

from config import REDIS_URL, SHORT_TERM_MAX_TURNS, SHORT_TERM_TTL

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Redis 短期记忆。
    每个 session 使用独立的 key：irene:session:<session_id>
    内部存储为 JSON 列表，每条消息格式：{"role": "user"|"assistant", "content": "..."}
    """

    KEY_PREFIX = "irene:session:"

    def __init__(self) -> None:
        try:
            self._client: Redis = redis.from_url(REDIS_URL, decode_responses=True)
            self._client.ping()
            logger.info("✦ Redis 短期记忆连接成功：%s", REDIS_URL)
        except Exception as e:
            logger.error("Redis 连接失败：%s", e)
            raise RuntimeError(
                f"无法连接 Redis（{REDIS_URL}），请确认 Redis 服务已启动。\n错误：{e}"
            )

    # ── 公开接口 ──────────────────────────────

    def get(self, session_id: str) -> List[dict]:
        """获取该 session 的全部短期历史"""
        raw = self._client.get(self._key(session_id))
        if not raw:
            return []
        return json.loads(str(raw))

    def append(self, session_id: str, role: str, content: str) -> List[dict]:
        """
        追加一条消息，返回追加后的完整历史。
        同时刷新过期时间。
        """
        messages = self.get(session_id)
        messages.append({"role": role, "content": content})
        self._save(session_id, messages)
        return messages

    def get_overflow(self, session_id: str) -> List[dict]:
        """
        返回超出短期记忆窗口的旧消息（用于迁移到长期记忆）。
        若无溢出则返回空列表。
        """
        messages = self.get(session_id)
        max_messages = SHORT_TERM_MAX_TURNS * 2  # 每轮 1问+1答
        if len(messages) <= max_messages:
            return []
        return messages[:-max_messages]  # 最旧的部分

    def trim(self, session_id: str) -> None:
        """
        裁剪历史，只保留最近 SHORT_TERM_MAX_TURNS 轮。
        在迁移完溢出数据后调用。
        """
        messages = self.get(session_id)
        max_messages = SHORT_TERM_MAX_TURNS * 2
        if len(messages) > max_messages:
            self._save(session_id, messages[-max_messages:])

    def clear(self, session_id: str) -> None:
        """清空该 session 的短期记忆"""
        self._client.delete(self._key(session_id))

    def refresh_ttl(self, session_id: str) -> None:
        """刷新过期时间"""
        self._client.expire(self._key(session_id), SHORT_TERM_TTL)

    # ── 私有方法 ──────────────────────────────

    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    def _save(self, session_id: str, messages: List[dict]) -> None:
        key = self._key(session_id)
        self._client.set(key, json.dumps(messages, ensure_ascii=False))
        self._client.expire(key, SHORT_TERM_TTL)