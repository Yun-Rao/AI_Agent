# 灰之魔女 · 伊蕾娜 AI

基于 LangChain + Kimi + Redis + ChromaDB + MySQL 构建的角色扮演 AI，扮演动漫《魔女之旅》中的伊蕾娜。

---

## 功能特性

- **角色扮演**：完整还原伊蕾娜的性格、说话风格与世界观设定
- **流式输出**：逐 token 实时显示，打字机效果
- **短期记忆**：Redis 缓存最近 10 轮对话，保持上下文连贯
- **长期记忆**：ChromaDB 语义向量存储，支持历史对话召回
- **知识库 RAG**：导入 PDF 文档，支持图文混合内容，向量召回 + Reranker 精排
- **两阶段工具决策**：独立决策模型判断是否需要检索，兼容 Kimi 模型
- **对话历史管理**：MySQL 永久归档，侧边栏切换历史对话，支持重命名与删除
- **跨设备恢复**：通过 session_id 从 MySQL 恢复历史对话

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| LLM | Kimi（kimi-k2-turbo-preview） |
| 短期记忆 | Redis |
| 长期记忆 / 知识库 | ChromaDB |
| Embedding 模型 | BAAI/bge-base-zh-v1.5（本地） |
| Reranker 模型 | BAAI/bge-reranker-base（本地） |
| PDF 解析 | PyMuPDF |
| 永久归档 | MySQL 8.0 |
| 前端 | 原生 HTML + CSS + JS（SSE 流式） |

---

## 目录结构

```
irene-ai-v2/
├── app.py                  # 主入口，FastAPI 接口
├── config.py               # 统一配置，读取环境变量
├── index.html              # 前端界面
├── requirements.txt        # Python 依赖
├── knowledge/              # PDF 知识库目录，启动时自动导入
├── memory/
│   ├── short_term.py       # Redis 短期记忆
│   ├── long_term.py        # ChromaDB 长期记忆
│   └── manager.py          # 记忆管理器
├── rag/
│   ├── retriever.py        # RAG 检索（向量召回 + Reranker 精排）
│   ├── pdf_loader.py       # PDF 解析（图文混合）
│   └── reranker.py         # CrossEncoder 重排序
└── database/
    └── mysql.py            # MySQL 永久归档
```

---

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Redis

```bash
docker run -d --name redis-irene -p 6379:6379 --restart always redis
```

### 3. 启动 MySQL

```bash
docker run -d --name mysql-irene -e MYSQL_ROOT_PASSWORD=你的密码 -e MYSQL_DATABASE=irene_ai -e MYSQL_CHARSET=utf8mb4 -p 3100:3306 --restart always mysql:8.0
```

> 端口 3306/3307 被系统保留时换用 3100。

### 4. 设置环境变量

在系统环境变量中添加以下配置（Windows：此电脑 → 属性 → 高级系统设置 → 环境变量）：

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `MOONSHOT_API_KEY` | Kimi API Key | `sk-xxxxxxxx` |
| `MYSQL_ENABLED` | 是否启用 MySQL | `true` |
| `MYSQL_URL` | MySQL 连接地址 | `mysql+pymysql://root:密码@localhost:3100/irene_ai` |
| `HF_ENDPOINT` | HuggingFace 镜像（可选） | `https://hf-mirror.com` |

---

## 启动

```bash
python app.py
```

启动后访问 http://localhost:8000

正常启动日志：

```
✦ KIMI_API_KEY 已加载：sk-1zJ2X****
✦ MySQL 已连接并完成建表检查
✦ PDF 后台导入任务已启动，服务可立即使用
```

> MySQL 表会在首次启动时自动创建，无需手动执行 DDL。

---

## 知识库导入

### 自动导入
将 PDF 文件放入 `knowledge/` 目录，启动时自动在后台导入，已导入的文件会跳过。

### 手动导入（API）
```bash
curl -X POST http://localhost:8000/api/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "knowledge/data.pdf"}'
```

### 查看知识库状态
```bash
curl http://localhost:8000/api/rag/stats
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 普通对话（一次性返回） |
| POST | `/api/chat/stream` | 流式对话（SSE） |
| POST | `/api/clear` | 清空 session 所有数据 |
| GET | `/api/sessions` | 列出所有历史 session |
| POST | `/api/session/restore` | 跨设备恢复历史对话 |
| POST | `/api/session/rename` | 重命名 session |
| GET | `/api/session/messages` | 获取 session 历史消息 |
| POST | `/api/rag/ingest` | 手动导入 PDF |
| GET | `/api/rag/stats` | 知识库统计 |

---

## 对话流程

```
用户输入
    ↓
工具决策（decision_llm，temperature=0）
    ├── search_memory    → ChromaDB 语义召回历史对话
    ├── search_knowledge → 向量召回（20条）→ Reranker 精排（top3）
    └── none             → 跳过检索
    ↓
组装上下文（工具结果 + Redis 短期记忆 10轮）
    ↓
主模型流式生成（llm.astream）
    ↓
逐 token SSE 推送前端
    ↓
生成完成后保存记忆
    ├── Redis（短期，超出迁移 ChromaDB）
    ├── ChromaDB（长期语义记忆）
    └── MySQL（永久归档）
```

---

## 重新启动项目

每次启动前确认 Docker 容器正在运行：

```bash
docker ps
```

看到 `redis-irene` 和 `mysql-irene` 后直接运行：

```bash
python app.py
```

Docker Desktop 设置了 `--restart always`，重启电脑后容器会自动恢复。

---

## 注意事项

- Embedding 和 Reranker 模型首次运行时自动下载，之后缓存在本地，无需重复下载
- Kimi 偶发 502/429，LangChain 会自动重试，不影响最终结果
- MySQL 的 `sessions` 表如果在加入 `name` 字段前已存在，需手动执行：
  ```sql
  ALTER TABLE irene_ai.sessions ADD COLUMN name VARCHAR(200) NULL AFTER session_id;
  ```
