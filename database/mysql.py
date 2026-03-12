"""
MySQL 持久化模块
职责：
  - 永久归档每一轮对话（原始记录仓库）
  - 支持跨设备恢复：根据 session_id 取回历史记录
  - 支持按 session 清除记录

启用步骤：
  1. 安装依赖：pip install sqlalchemy pymysql
  2. 启动 MySQL 并创建数据库：
       CREATE DATABASE irene_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  3. 设置环境变量：
       MYSQL_ENABLED=true
       MYSQL_URL=mysql+pymysql://用户名:密码@localhost:3306/irene_ai
  4. 首次启动时会自动建表，无需手动执行 DDL

表结构（自动创建）：
  conversations：每条消息一行，包含 session_id / role / content / created_at
  sessions：session 元信息，记录首次创建时间
"""

import logging
from datetime import datetime
from typing import List, Optional

from config import MYSQL_ENABLED, MYSQL_URL

logger = logging.getLogger(__name__)

# ── 按需导入 SQLAlchemy ──────────────────────────────────────
if MYSQL_ENABLED:
    try:
        from sqlalchemy import (
            create_engine,
            BigInteger, String, Text, DateTime, Enum
        )
        from sqlalchemy.orm import declarative_base, Session, sessionmaker, Mapped, mapped_column
        import enum

        engine = create_engine(
            MYSQL_URL,
            pool_pre_ping=True,       # 连接断开时自动重连
            pool_recycle=3600,        # 1小时回收连接，防止 MySQL 断开空闲连接
            echo=False,               # 不打印 SQL，减少日志噪音
        )
        Base = declarative_base()
        SessionLocal = sessionmaker(bind=engine)

        # ── ORM 模型 ─────────────────────────────────────────

        class RoleEnum(str, enum.Enum):
            user = "user"
            assistant = "assistant"

        class Conversation(Base):
            __tablename__ = "conversations"
            id:         Mapped[int]      = mapped_column(BigInteger, primary_key=True, autoincrement=True)
            session_id: Mapped[str]      = mapped_column(String(64), nullable=False, index=True)
            role:       Mapped[RoleEnum] = mapped_column(Enum(RoleEnum), nullable=False)
            content:    Mapped[str]      = mapped_column(Text, nullable=False)
            created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

        class SessionMeta(Base):
            __tablename__ = "sessions"
            session_id:  Mapped[str]               = mapped_column(String(64), primary_key=True)
            name:        Mapped[Optional[str]]      = mapped_column(String(200), nullable=True)   # 对话标题，默认为第一条用户消息
            created_at:  Mapped[datetime]           = mapped_column(DateTime, default=datetime.utcnow)
            last_active: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=True)

        # 自动建表
        Base.metadata.create_all(engine)
        logger.info("✦ MySQL 已连接并完成建表检查：%s", MYSQL_URL)

    except ImportError:
        raise RuntimeError(
            "MYSQL_ENABLED=true 但缺少依赖，请执行：\n"
            "pip install sqlalchemy pymysql"
        )
    except Exception as e:
        raise RuntimeError(f"MySQL 连接失败：{e}\n请检查 MYSQL_URL 和 MySQL 服务状态")
else:
    logger.info("✦ MySQL 未启用（MYSQL_ENABLED=false）")


class MySQLStorage:
    """
    MySQL 对话记录持久化。
    MYSQL_ENABLED=false 时所有操作为空实现（no-op），不影响主流程。
    """

    # ── 写入 ────────────────────────────────────────────────

    def save_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """永久保存一轮对话（user + assistant 各一行）"""
        if not MYSQL_ENABLED:
            return
        try:
            with SessionLocal() as db:
                # 更新或插入 session 元信息
                meta = db.get(SessionMeta, session_id)
                if meta is None:
                    # 首次创建：用第一条用户消息前 30 字作为默认名称
                    default_name = user_msg[:30] + ("…" if len(user_msg) > 30 else "")
                    db.add(SessionMeta(session_id=session_id, name=default_name))
                else:
                    meta.last_active = datetime.utcnow()

                # 写入两条消息
                db.add(Conversation(session_id=session_id, role=RoleEnum.user,      content=user_msg))
                db.add(Conversation(session_id=session_id, role=RoleEnum.assistant, content=assistant_msg))
                db.commit()
        except Exception as e:
            logger.error("MySQL save_turn 失败：%s", e)

    # ── 查询 ────────────────────────────────────────────────

    def get_history(self, session_id: str, limit: int = 100) -> List[dict]:
        """
        取回该 session 的历史记录，按时间正序排列。
        用于跨设备恢复：新设备打开时用 session_id 从 MySQL 拉取历史，
        重新写入 Redis 短期记忆，实现无缝继续对话。
        """
        if not MYSQL_ENABLED:
            return []
        try:
            with SessionLocal() as db:
                rows = (
                    db.query(Conversation)
                    .filter(Conversation.session_id == session_id)
                    .order_by(Conversation.created_at.desc(), Conversation.id.desc())
                    .limit(limit)
                    .all()
                )
                # 倒序取出后翻转为正序
                return [
                    {"role": r.role.value, "content": r.content}
                    for r in reversed(rows)
                ]
        except Exception as e:
            logger.error("MySQL get_history 失败：%s", e)
            return []

    def list_sessions(self) -> List[dict]:
        """列出所有 session 及其最后活跃时间（可用于管理界面）"""
        if not MYSQL_ENABLED:
            return []
        try:
            with SessionLocal() as db:
                rows = db.query(SessionMeta).order_by(SessionMeta.last_active.desc()).all()
                return [
                    {
                        "session_id": r.session_id,
                        "name": r.name or r.session_id[:8],
                        "created_at": r.created_at.isoformat() if r.created_at else "",
                        "last_active": r.last_active.isoformat() if r.last_active else "",
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error("MySQL list_sessions 失败：%s", e)
            return []

    # ── 删除 ────────────────────────────────────────────────

    def rename_session(self, session_id: str, name: str) -> bool:
        """修改 session 名称，成功返回 True"""
        if not MYSQL_ENABLED:
            return False
        try:
            with SessionLocal() as db:
                meta = db.get(SessionMeta, session_id)
                if meta is None:
                    return False
                meta.name = name[:200]
                db.commit()
                return True
        except Exception as e:
            logger.error("MySQL rename_session 失败：%s", e)
            return False

    def clear(self, session_id: str) -> None:
        """清除该 session 的所有 MySQL 记录"""
        if not MYSQL_ENABLED:
            return
        try:
            with SessionLocal() as db:
                db.query(Conversation).filter(Conversation.session_id == session_id).delete()
                db.query(SessionMeta).filter(SessionMeta.session_id == session_id).delete()
                db.commit()
                logger.info("✦ MySQL：session=%s 记录已清除", session_id)
        except Exception as e:
            logger.error("MySQL clear 失败：%s", e)


# ── 单例 ────────────────────────────────────────────────────
_mysql_instance: MySQLStorage | None = None

def get_mysql_storage() -> MySQLStorage:
    global _mysql_instance
    if _mysql_instance is None:
        _mysql_instance = MySQLStorage()
    return _mysql_instance