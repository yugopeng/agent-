# 智扫通 · 扫地机器人智能客服 Agent

基于 **LangChain / LangGraph** 构建的 RAG + ReAct Agent 智能客服系统。面向扫地机器人与扫拖一体机器人场景，支持知识库问答、天气与环境适配建议、个性化使用报告生成三大能力，并通过**中间件动态切换系统提示词**实现「客服」与「报告写手」双人格。

---

## 目录

- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [核心机制：动态提示词切换](#核心机制动态提示词切换)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [工具清单](#工具清单)
- [运行方式](#运行方式)
- [开发指南](#开发指南)
- [已知问题](#已知问题)

---

## 核心特性

| 能力 | 说明 |
|---|---|
| **知识库问答（RAG）** | 本地文档 → 分块 → 向量化 → Chroma 检索 → LLM 总结，回答严格基于参考资料，不允许编造 |
| **ReAct 自主推理** | 遵循「思考 → 行动 → 观察 → 再思考」循环，自主决定调用哪个工具、调用几次 |
| **双人格动态切换** | 通过中间件在运行时切换系统提示词，客服场景与报告生成场景使用完全不同的行为约束 |
| **个性化报告生成** | 拉取用户 ID、月份、外部使用记录（清洁效率 / 耗材 / 对比），生成 Markdown 格式使用报告与保养建议 |
| **增量知识库** | 基于文件 MD5 指纹去重，重复文件自动跳过，避免向量库重复入库 |
| **可观测性** | 工具调用前后日志埋点、模型调用前状态日志，全流程可追踪 |
| **流式输出** | 基于 `stream_mode="values"` 逐块吐出，前端可打字机式渲染 |

---

## 技术栈

| 层次 | 组件 |
|---|---|
| Agent 框架 | LangChain `create_agent`、LangGraph `Runtime` / middleware |
| 对话模型 | DeepSeek（`langchain-deepseek`） |
| 向量模型 | 阿里云 DashScope `text-embedding-v4` |
| 向量数据库 | Chroma（`langchain-chroma`） |
| 文本切分 | `RecursiveCharacterTextSplitter` |
| 文档解析 | `PyPDFLoader`、`TextLoader` |
| 前端 | Streamlit |
| 服务端（预留） | FastAPI + SSE |
| 配置 | YAML + 系统环境变量 |

**Python 要求**：3.10+（使用了 `str | None` 等 PEP 604 语法）

---

## 系统架构

```
                         ┌──────────────────────────┐
                         │   用户提问 (query)        │
                         └────────────┬─────────────┘
                                      │
                   context = {"report": False}
                                      │
                    ┌─────────────────▼──────────────────┐
                    │        ReactAgent (LangGraph)      │
                    │                                    │
                    │  ┌──────────────────────────────┐  │
                    │  │ before_model: log_before_model│ │  ← 日志埋点
                    │  └──────────────┬───────────────┘  │
                    │                 │                   │
                    │  ┌──────────────▼───────────────┐  │
                    │  │ dynamic_prompt:               │ │  ← 动态提示词
                    │  │   report_prompt_switch        │ │
                    │  │   False → main_prompt          │ │
                    │  │   True  → report_prompt        │ │
                    │  └──────────────┬───────────────┘  │
                    │                 │                   │
                    │  ┌──────────────▼───────────────┐  │
                    │  │      LLM (DeepSeek)           │ │
                    │  └──────────────┬───────────────┘  │
                    │                 │                   │
                    │  ┌──────────────▼───────────────┐  │
                    │  │ wrap_tool_call: monitor_tool  │ │  ← 工具拦截
                    │  │  工具 == fill_context_for_report│ │
                    │  │      → context["report"]=True  │ │
                    │  └──────────────┬───────────────┘  │
                    └─────────────────┼──────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
        ┌─────▼──────┐        ┌───────▼──────┐        ┌───────▼──────┐
        │rag_summarize│        │ get_weather  │        │fetch_external│
        │  Chroma 检索│        │get_user_loc..│        │   _data      │
        └─────┬──────┘        └──────────────┘        └──────────────┘
              │
     ┌────────▼────────┐
     │ 向量库 (Chroma) │ ← data/*.txt / *.pdf 分块入库
     └─────────────────┘
```

---

## 目录结构

```
AI大模型RAG与智能体开发_Agent项目/
├── agent/
│   ├── react_agent.py              # Agent 装配与流式执行入口
│   └── tools/
│       ├── agent_tools.py          # 7 个工具的定义
│       └── middleware.py           # 3 个中间件（日志/监控/提示词切换）
├── config/
│   ├── agent.yml                   # 外部数据路径
│   ├── chroma.yml                  # 向量库、切分参数
│   ├── prompts.yml                 # 提示词文件路径映射
│   └── rag.yml                     # 模型名称
├── data/                           # 知识库语料（自动入库）
│   ├── external/records.csv        # 用户使用记录（报告生成数据源）
│   ├── 扫地机器人100问.pdf
│   ├── 扫拖一体机器人100问.txt
│   ├── 故障排除.txt
│   ├── 维护保养.txt
│   └── 选购指南.txt
├── model/
│   └── factory.py                  # 模型工厂（Chat / Embedding）
├── prompts/
│   ├── main_prompt.txt             # 客服人格提示词
│   ├── rag_summarize.txt           # RAG 总结提示词
│   └── report_prompt.txt           # 报告写手人格提示词
├── rag/
│   ├── rag_service.py              # 检索 + 总结链
│   └── vector_store.py             # Chroma 封装与增量入库
├── server/
│   └── main.py                     # FastAPI + SSE（预留）
├── utils/
│   ├── config_handler.py           # YAML 配置加载
│   ├── file_handler.py             # MD5 / 文件扫描 / PDF&TXT 解析
│   ├── logger_handler.py           # 统一日志器
│   ├── path_tool.py                # 绝对路径工具
│   └── prompt_loader.py            # 提示词文件读取
├── app.py                          # Streamlit 前端入口
├── .env.example                    # 环境变量模板
├── .gitignore
└── requirements.txt
```

---

## 核心机制：动态提示词切换

这是本项目最有设计感的部分。需求上有两套完全不同的行为：**客服答疑** 和 **写使用报告**。如果用关键词匹配 `if "报告" in query` 来分流，误判率会很高。

项目采用的是「**模型负责语义判断，中间件负责确定性执行**」的方案：

### 1. 请求初始化打标记

```python
# agent/react_agent.py:28
self.agent.stream(input_dict, stream_mode="values", context={"report": False})
```

`context` 是 LangGraph 的运行时上下文，随本次调用存在，天然做到请求级隔离——上一个问题进了报告模式，不会污染下一个问题。

### 2. 空壳工具作为信号

```python
# agent/tools/agent_tools.py:109
@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
```

它**不干任何实际工作**，唯一价值是给中间件一个可拦截的事件。主提示词里写死约束：识别到报告意图时，必须先调它，再调 `fetch_external_data`。

### 3. 中间件拦截并翻转标记

```python
# agent/tools/middleware.py:12
@wrap_tool_call
def monitor_tool(request, handler):
    result = handler(request)
    if request.tool_call['name'] == "fill_context_for_report":
        request.runtime.context["report"] = True   # ← 翻转开关
    return result
```

`runtime.context` 是**同一个 dict 引用**，在这里写入，另一个中间件能立刻读到。

### 4. 每次模型调用前重建提示词

```python
# agent/tools/middleware.py:47
@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    if request.runtime.context.get("report", False):
        return load_report_prompts()    # 报告写手人格
    return load_system_prompts()        # 客服人格
```

`@dynamic_prompt` 的**返回值会替换本次模型调用的 system prompt**。

### 执行时序

```
第 1 轮  prompt=客服    → LLM 判断「这是报告需求」→ 调 get_user_id / get_current_month
第 2 轮  prompt=客服    → 调 fill_context_for_report → 中间件置 report=True
第 3 轮  prompt=报告写手 ← 已切换！→ 调 fetch_external_data
第 4 轮  prompt=报告写手 → 生成 Markdown 报告
```

> ⚠️ 注意切换发生在**下一轮**模型调用，因为 `dynamic_prompt` 只在调用前执行。这正是主提示词要求 `fill_context_for_report` 必须**前置**于 `fetch_external_data` 的原因。

---

## 快速开始

### 1. 克隆并安装依赖

```bash
git clone <your-repo-url>
cd AI大模型RAG与智能体开发_Agent项目

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

项目**不在代码中硬编码任何密钥**，全部通过系统环境变量读取：

| 环境变量 | 用途 | 获取方式 |
|---|---|---|
| `deepseek_apikey` | DeepSeek 对话模型 | <https://platform.deepseek.com> |
| `DASHSCOPE_API_KEY` | 阿里云向量模型 | <https://dashscope.console.aliyun.com> |

**方式 A：系统环境变量（推荐，最简单）**

```bash
# Windows CMD
setx deepseek_apikey "sk-xxxxxxxx"
setx DASHSCOPE_API_KEY "sk-xxxxxxxx"

# macOS / Linux
export deepseek_apikey="sk-xxxxxxxx"
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

**方式 B：`.env` 文件**（需在主入口顶部加 `from dotenv import load_dotenv; load_dotenv()`）

```bash
cp .env.example .env
# 编辑 .env 填入真实 key
```

> 🔒 `.env` 已被 `.gitignore` 排除，**切勿提交**。

### 3. 构建知识库

首次运行前需要把 `data/` 下的文档向量化入库：

```bash
python -m rag.vector_store
```

该命令会扫描 `data/` 目录，按 MD5 去重后切分、向量化并写入 `chroma_db/`。重复执行不会重复入库。

> 💡 后续新增文档后重跑此命令即可增量入库。

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## 配置说明

所有配置集中在 `config/` 目录，修改后无需改代码。

### `config/rag.yml`

```yaml
chat_model_name: deepseek-chat        # 对话模型
embedding_model_name: text-embedding-v4  # 向量模型
```

### `config/chroma.yml`

```yaml
collection_name: agent                # Chroma 集合名
persist_directory: chroma_db          # 向量库持久化目录
k: 3                                  # 检索返回条数
data_path: data                       # 知识库源目录
md5_hex_store: md5.text               # MD5 去重记录文件
allow_knowledge_file_type: ["txt", "pdf"]

chunk_size: 200                       # 分块大小
chunk_overlap: 20                     # 分块重叠
separators: ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
```

### `config/prompts.yml`

```yaml
main_prompt_path: prompts/main_prompt.txt            # 客服人格
rag_summarize_prompt_path: prompts/rag_summarize.txt # RAG 总结
report_prompt_path: prompts/report_prompt.txt        # 报告写手人格
```

### `config/agent.yml`

```yaml
external_data_path: data/external/records.csv  # 用户使用记录数据源
```

---

## 工具清单

Agent 可自主调用以下 7 个工具：

| 工具 | 入参 | 出参 | 说明 |
|---|---|---|---|
| `rag_summarize` | `query` | 专业资料文本 | 从 Chroma 检索知识库，交给 LLM 总结 |
| `get_weather` | `city` | 天气字符串 | 获取城市天气、湿度、降雨概率、AQI |
| `get_user_location` | — | 城市名 | 获取用户所在城市 |
| `get_user_id` | — | 用户 ID | 获取当前用户标识 |
| `get_current_month` | — | `YYYY-MM` | 获取当前月份 |
| `fetch_external_data` | `user_id`, `month` | 结构化记录 | 拉取指定用户指定月份的使用数据 |
| `fill_context_for_report` | — | 固定字符串 | **空壳工具**，仅用于触发提示词切换 |

> `get_weather` / `get_user_location` / `get_user_id` / `get_current_month` 目前是 **Mock 实现**（随机或固定值），接入真实业务系统时替换其函数体即可，Agent 侧无需改动。

---

## 运行方式

### Streamlit 界面

```bash
streamlit run app.py
```

特性：会话级 Agent 实例复用、历史消息保持、逐字打字机效果（`app.py:35` 每次 sleep 10ms）。

### 直接在代码中调用

```python
from agent.react_agent import ReactAgent

agent = ReactAgent()
for chunk in agent.execute_stream("帮我生成上个月的使用报告"):
    print(chunk, end="", flush=True)
```

### FastAPI 服务（预留）

```bash
uvicorn server.main:app --reload --port 8000
```

接口：`POST /api/chat/stream`，SSE 流式返回。详见 [已知问题](#已知问题)。

---

## 开发指南

### 新增知识库文档

1. 把 `.txt` / `.pdf` 放入 `data/`
2. 运行 `python -m rag.vector_store`

MD5 去重会跳过未变更的文件，只入库新文件。

### 新增工具

```python
# agent/tools/agent_tools.py
@tool(description="清晰描述工具能力、入参格式、使用场景，LLM 据此决定是否调用")
def my_tool(param: str) -> str:
    return "..."
```

然后在两处注册：

1. `react_agent.py:14` 的 `tools=[...]` 列表
2. 对应的提示词文件（`main_prompt.txt` 和/或 `report_prompt.txt`）中补充工具说明——**这一步不能省**，LLM 靠提示词理解工具用途。

### 新增中间件

```python
@before_model          # 模型调用前
@after_model           # 模型调用后
@wrap_tool_call        # 包裹工具调用
@dynamic_prompt        # 动态替换 system prompt
```

注册到 `react_agent.py:16` 的 `middleware=[...]` 即可。

### 调整人格行为

直接编辑 `prompts/*.txt`。文件在每次模型调用时重新读取，修改后**无需重启**服务。

---

## 已知问题

| 问题 | 位置 | 说明 |
|---|---|---|
| FastAPI 接口签名不匹配 | `server/main.py:28` | 调用 `execute_stream(req.query, thread_id=...)`，但 `execute_stream` 当前签名为 `(self, query: str)`，且 yield 的是 `str` 而非 `dict`，`json.dumps(event)` 会得到 JSON 字符串。该接口需与 Agent 对齐后才能使用。 |
| 提示词重复读盘 | `utils/prompt_loader.py` | 每次模型调用都重新读取文件，多轮工具调用下存在冗余 IO，可加缓存优化。 |
| 模式单向切换 | `middleware.py:27` | `report` 只会 `False → True`，同一次 `stream()` 内无法切回客服人格。若模型误判调用了 `fill_context_for_report`，本轮后续回答都将使用报告人格。 |
| `data/` 为示例语料 | `data/` | 当前语料为公开示例文档，若替换为内部资料，请在 `.gitignore` 中放开 `data/` 的排除规则。 |

---

## 安全须知

- 所有密钥通过环境变量注入，**禁止**写入任何会提交到仓库的文件
- `.gitignore` 已排除 `.env`、`venv/`、`logs/`、`chroma_db/`、`.idea/`
- 推送前建议执行：

```bash
git grep -niE "sk-|api[_-]?key *=|password" --cached
git check-ignore -v venv/ .env logs/
```

- 如不慎泄露：**立即到平台吊销旧 key 并重新生成**，再用 `git filter-repo` 清理历史——仅删除文件无效，历史记录中仍然存在。

---

## License

仅用于学习与技术交流。
