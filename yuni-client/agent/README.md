# Yuni Agent

Yuni 的本地 Python 服务，使用 FastAPI、LangGraph、SQLite 和 Alembic。提供伴侣资料、聊天队列与事件推送、长期记忆、文件工具、提醒、图像生成和语音回复。

## 安装与启动

从 `yuni-client` 目录执行：

```powershell
uv sync --project agent --extra dev
npm run agent:dev
```

开发服务监听 `127.0.0.1:43127`，使用固定的本机开发 Session Token。配套前端的连接配置见[客户端说明](../README.md)。

普通开发模式的数据和日志位于系统临时目录的 `yuni-agent-dev/data` 与 `yuni-agent-dev/logs`。IDE 入口 `run.py` 使用仓库内被 Git 忽略的 `output/first-round/` 数据目录；两种启动方式的数据独立。

## 模型配置

在前端「设置 → 模型配置」中保存服务商地址、模型名称和自己的 API Key。聊天模型需要支持 OpenAI 兼容的工具调用及 JSON 结构化输出。

- 聊天模型用于 Planner、Replyer 和记忆整理。
- Embedding 可选；缺省时长期记忆使用本地 BM25 检索，表达选择使用加权回退。
- 图像生成和语音合成仅在启用且配置齐备时开放，对应工具、参数及提示词按需提供。
- 语音支持 MiniMax 官方原生接口和 OpenAI 兼容 TTS；MiniMax 可读取账号音色列表。

凭据保存在数据目录的 `auth.json`、`embedding-auth.json`、`image-auth.json`、`voice-auth.json` 中。Key 为本机明文配置，不提交 Git；API 响应、数据库、日志和导出备份不应包含 Key。保存配置时 Key 留空可保留同地址的已有 Key，适用的扩展也可复用同地址聊天模型的 Key。

## 对话与文件权限

`POST /api/chat/messages` 接收输入，后台按会话串行处理；`GET /api/events` 提供可重放的全局 SSE。切换页面不会取消后台生成。

Planner 根据当前对话选择工具或 `reply`，Replyer 生成角色台词，实际交付回执返回 Planner；调用 `stop` 结束本轮。长期记忆与表达学习使用独立后台任务，SQLite 原文是事实来源，向量索引可重建。

文件工具仅访问已授权目录，相对路径以当前默认工作目录为基准。写入和敏感操作通过确认流程执行，并保存调用记录。变更启动工作目录后，旧项标为历史工作目录，已有文件和撤销状态保留。

表情库初始为空；个人表情素材、参考音频及克隆素材不随源码提供。

## Sidecar 接口

生产入口通过 stdin 接收一行启动 JSON，stdout 输出 `ready` 握手，日志输出到 stderr 和指定日志目录：

```json
{
  "dataDir": "C:/Users/me/AppData/Roaming/Yuni",
  "logDir": "C:/Users/me/AppData/Local/Yuni/logs",
  "sessionToken": "replace-with-a-random-token-of-at-least-32-bytes",
  "allowedOrigins": ["http://tauri.localhost"]
}
```

Tauri 自动启动和打包此服务的集成尚未完成。

## 测试

从 `yuni-client` 目录执行：

```powershell
npm run agent:test
```

测试源码随仓库保留；测试生成的数据、报告和日志不入库。

## 来源

部分 Planner / Replyer 提示词、表达学习及分条规则参考 [MaiBot 的 ff0fe425 版本](https://github.com/Mai-with-u/MaiBot/tree/ff0fe42500f5f7581e9655d1e78379407233207a)。涉及 `agent/planner.py`、`agent/chat_agent.py`、`context/planner_builder.py`、`model/reply.py` 与 `expressions/`；具体差异和来源注释保留在代码中。
