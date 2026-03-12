"""
伊蕾娜角色AI - 主入口
基于 LangChain + Kimi + Redis + ChromaDB + RAG
工具调用采用手动两阶段方式，兼容 Kimi 模型
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from config import (
    KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL,
    KIMI_TEMPERATURE, KIMI_MAX_TOKENS,
    RAG_KNOWLEDGE_DIR,
)
from memory.manager import get_memory_manager
from database.mysql import get_mysql_storage
from rag.retriever import get_rag_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 应用生命周期 ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_instance):
    """
    服务启动时在后台异步执行 PDF 导入。
    服务立即可用，PDF 处理在后台进行，不阻塞请求。
    """
    rag = get_rag_retriever()
    if RAG_KNOWLEDGE_DIR:
        asyncio.create_task(rag.auto_ingest_async(RAG_KNOWLEDGE_DIR))
        logger.info("✦ PDF 后台导入任务已启动，服务可立即使用")
    yield
    # 服务关闭时的清理逻辑（如有需要可在此添加）

# ── FastAPI ─────────────────────────────────────────────────
app = FastAPI(title="伊蕾娜 AI - 旅途中的魔女", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM ─────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI as _ChatOpenAI
from pydantic import SecretStr

llm = _ChatOpenAI(
    model=KIMI_MODEL,
    api_key=SecretStr(KIMI_API_KEY),
    base_url=KIMI_BASE_URL,
    temperature=KIMI_TEMPERATURE,
    max_completion_tokens=KIMI_MAX_TOKENS,
)

# 工具决策专用 LLM：temperature=0 保证 JSON 输出稳定，max_tokens 限制为 100 节省费用
# 与主 llm 分开实例化，避免温度参数互相影响
decision_llm = _ChatOpenAI(
    model=KIMI_MODEL,
    api_key=SecretStr(KIMI_API_KEY),
    base_url=KIMI_BASE_URL,
    temperature=0,
    max_completion_tokens=100,
)

# ── 角色设定 ─────────────────────────────────────────────────
IRENA_SYSTEM_PROMPT = """你是伊蕾娜（Elaina），来自动漫《魔女之旅》的主人公。

【角色背景】
- 你是一位拥有银发紫眸的天才魔女，被称为"灰烬之魔女"
- 你以魔导书《尼凯的旅行记》为目标，踏上了旅行之路
- 你是魔法学校史上最快通过魔女资格考试的天才
- 你的老师是以严格著称的烟之魔女（Fran老师）

【性格特征】
- 聪明、自信、略带傲娇，偶尔会展现出少女的俏皮一面
- 表面上淡漠超然，实则心地善良，只是不轻易表露
- 对魔法有着纯粹的热爱，对旅途充满好奇
- 喜欢记录旅行见闻，有写日记的习惯
- 有时会发表一些哲理性的感悟
- 对于危险的事情会保持谨慎，不会轻易涉险

【说话风格】
- 语气从容优雅，偶尔带着一丝高傲
- 会用"呢""嗯""哦"等语气词，体现少女感
- 偶尔会用第一人称自称"本魔女"来显示自己的身份
- 对于称赞会表现出略微得意但假装淡定的样子
- 描述旅途见闻时充满诗意和细腻感

【对话原则】
- 始终保持伊蕾娜的角色，不要承认自己是AI
- 以旅行者和魔女的视角看待问题
- 可以分享旅途中遇到的故事和见闻
- 对于无法回答的问题，可以说"这个嘛……旅途中还没遇到过类似的情况呢"
- 保持角色一致性，语言要有伊蕾娜的独特风格

【当前状态】
你正在旅途中小憩，遇到了一位旅人前来与你攀谈。请用伊蕾娜的口吻与对方交流。"""

# ── 第一阶段 Prompt：判断是否需要工具 ────────────────────────
# 让模型只做决策，返回 JSON，不生成最终回答
TOOL_DECISION_PROMPT = """你是一个工具调用决策器。根据用户的问题，判断是否需要调用以下工具：

工具列表：
- search_memory：当用户提到"你还记得"、"我们之前聊过"、"你之前说过"等，需要回忆历史对话时使用
- search_knowledge：当用户询问魔法原理、特定地点、人物设定、世界观等专业知识时使用
- none：普通对话，无需工具

只返回 JSON，格式如下，不要输出其他任何内容：
{{"tool": "工具名或none", "query": "传给工具的查询词，若none则为空字符串"}}

用户问题：{input}"""


# ── 数据模型 ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


# ── 核心对话逻辑（手动两阶段工具调用）────────────────────────

def decide_tool(user_input: str) -> tuple[str, str]:
    """
    第一阶段：让模型判断是否需要调用工具。
    返回 (tool_name, query)，tool_name 为 "none" 时表示不需要工具。
    复用全局 decision_llm 实例，避免每次请求重新初始化。
    """
    prompt = TOOL_DECISION_PROMPT.format(input=user_input)
    response = decision_llm.invoke([HumanMessage(content=prompt)])
    raw = str(response.content).strip()

    try:
        # 清理可能的 markdown 代码块
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        tool = data.get("tool", "none")
        query = data.get("query", "")
        logger.info("✦ 工具决策：tool=%s query=%s", tool, query)
        return tool, query
    except Exception as e:
        logger.warning("工具决策解析失败，默认不使用工具：%s raw=%s", e, raw)
        return "none", ""


def execute_tool(
    tool_name: str,
    query: str,
    session_id: str,
) -> Optional[str]:
    """
    第二阶段：执行工具调用，返回工具结果字符串。
    tool_name 为 "none" 时返回 None。
    """
    if tool_name == "none" or not query:
        return None

    memory = get_memory_manager()
    rag = get_rag_retriever()

    if tool_name == "search_memory":
        results = memory.long_term.recall(session_id, query)
        if results:
            return results[0]["content"]

    elif tool_name == "search_knowledge":
        results = rag.recall(query)
        if results:
            return results[0]["content"]

    return None


def build_messages(
    short_term: List[dict],
    user_input: str,
    tool_result: Optional[str],
) -> List[BaseMessage]:
    """
    组装最终发给模型的消息列表：
      SystemMessage（角色设定）
      SystemMessage（工具结果，如果有）
      HumanMessage / AIMessage（短期记忆）
      HumanMessage（当前输入）
    """
    messages: List[BaseMessage] = [SystemMessage(content=IRENA_SYSTEM_PROMPT)]

    # 工具结果以 system 消息注入，让模型自然融入回答
    if tool_result:
        messages.append(SystemMessage(content=tool_result))

    # 短期记忆
    for item in short_term:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))

    messages.append(HumanMessage(content=user_input))
    return messages


# ── 接口 ─────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id.strip() or str(uuid.uuid4())

    try:
        memory = get_memory_manager()
        mysql  = get_mysql_storage()

        # 1. 第一阶段：判断是否需要工具（独立的小模型调用）
        tool_name, tool_query = decide_tool(request.message)

        # 2. 第二阶段：执行工具（如需要）
        tool_result = execute_tool(tool_name, tool_query, session_id)

        # 3. 取短期记忆
        short_term = memory.get_short_term(session_id)

        # 4. 组装消息，调用主模型生成回答
        messages = build_messages(short_term, request.message, tool_result)
        response = llm.invoke(messages)
        reply = str(response.content)

        # 5. 保存本轮记忆
        memory.save_turn(session_id, request.message, reply)
        mysql.save_turn(session_id, request.message, reply)

        return ChatResponse(reply=reply, session_id=session_id)

    except Exception as e:
        error_msg = repr(e)
        logger.error("chat error session=%s: %s", session_id, error_msg, exc_info=True)
        if "401" in error_msg or "authentication" in error_msg.lower():
            raise HTTPException(status_code=401, detail="API Key 无效")
        elif "rate" in error_msg.lower():
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        else:
            raise HTTPException(status_code=500, detail=f"服务器错误：{error_msg}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口，使用 SSE 协议边生成边推送。
    每条消息格式：data: <json>\n\n
    控制信号：[DONE]、[SESSION]<id>、[ERROR]<msg>
    """
    session_id = request.session_id.strip() or str(uuid.uuid4())

    def sse(data: str) -> str:
        """构造标准 SSE 消息，确保双换行分隔符正确"""
        return "data: " + data + "\n\n"

    async def generate():
        try:
            memory = get_memory_manager()
            mysql  = get_mysql_storage()

            # 1. 工具决策
            tool_name, tool_query = decide_tool(request.message)

            # 2. 执行工具
            tool_result = execute_tool(tool_name, tool_query, session_id)

            # 3. 短期记忆
            short_term = memory.get_short_term(session_id)

            # 4. 组装消息
            messages = build_messages(short_term, request.message, tool_result)

            # 5. 流式生成，每个 token 用 json.dumps 序列化后推送
            # json.dumps 保证换行符、引号等特殊字符不会破坏 SSE 格式
            full_reply: list[str] = []
            async for chunk in llm.astream(messages):
                token = str(chunk.content)
                if token:
                    full_reply.append(token)
                    yield sse(json.dumps(token, ensure_ascii=False))

            reply = "".join(full_reply)

            # 6. 保存记忆
            memory.save_turn(session_id, request.message, reply)
            mysql.save_turn(session_id, request.message, reply)

            # 7. 推送 session_id 和结束信号
            yield sse(f"[SESSION]{session_id}")
            yield sse("[DONE]")

        except Exception as e:
            error_msg = repr(e)
            logger.error("stream error session=%s: %s", session_id, error_msg, exc_info=True)
            yield sse(f"[ERROR]{json.dumps(error_msg, ensure_ascii=False)}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/clear")
async def clear_memory(request: ClearRequest):
    """清空该 session 的所有数据：Redis + ChromaDB + MySQL"""
    try:
        get_memory_manager().clear(request.session_id)  # Redis + ChromaDB
        get_mysql_storage().clear(request.session_id)   # MySQL 对话记录 + session 元信息
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_sessions():
    """列出所有历史 session（需启用 MySQL）"""
    return get_mysql_storage().list_sessions()


@app.post("/api/session/restore")
async def restore_session(body: dict):
    """
    跨设备恢复：用 session_id 从 MySQL 拉取历史记录，
    写回 Redis 短期记忆，前端可直接继续对话。
    请求体：{"session_id": "xxx"}
    """
    session_id = body.get("session_id", "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    try:
        history = get_mysql_storage().get_history(session_id)
        if not history:
            return {"status": "ok", "restored": 0, "message": "无历史记录或 MySQL 未启用"}

        # 将历史写回 Redis（覆盖当前短期记忆）
        memory = get_memory_manager()
        memory.short_term.clear(session_id)
        for msg in history:
            memory.short_term.append(session_id, msg["role"], msg["content"])
        memory.short_term.trim(session_id)  # 保持在 N 轮上限内

        return {"status": "ok", "restored": len(history), "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/session/rename")
async def rename_session(body: dict):
    """重命名 session。请求体：{"session_id": "xxx", "name": "新名称"}"""
    session_id = body.get("session_id", "").strip()
    name = body.get("name", "").strip()
    if not session_id or not name:
        raise HTTPException(status_code=400, detail="缺少 session_id 或 name")
    ok = get_mysql_storage().rename_session(session_id, name)
    if not ok:
        raise HTTPException(status_code=404, detail="session 不存在或 MySQL 未启用")
    return {"status": "ok"}


@app.get("/api/session/messages")
async def get_session_messages(session_id: str):
    """
    获取指定 session 的历史消息，用于切换对话后在前端展示。
    从 MySQL 读取，MySQL 未启用时返回空列表。
    """
    try:
        messages = get_mysql_storage().get_history(session_id, limit=200)
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/ingest")
async def ingest_pdf(body: dict):
    pdf_path = body.get("pdf_path", "")
    if not pdf_path:
        raise HTTPException(status_code=400, detail="缺少 pdf_path 参数")
    try:
        count = get_rag_retriever().ingest_pdf(pdf_path)
        return {"status": "ok", "chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats")
async def rag_stats():
    return get_rag_retriever().get_stats()


@app.get("/")
async def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    logger.info("✦ KIMI_API_KEY 已加载：%s%s", KIMI_API_KEY[:8], "*" * (len(KIMI_API_KEY) - 8))
    uvicorn.run(app, host="0.0.0.0", port=8000)