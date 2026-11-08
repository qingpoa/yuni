# Yuni

Yuni 是一个本地优先的 AI 陪伴项目，使用 React + TypeScript 构建界面，Python FastAPI + LangGraph 提供本地 Agent，SQLite 保存伴侣资料、聊天和记忆。

目前可以从源码运行 Web 界面与本地 Agent。Tauri 2 桌面工程已建立，Agent sidecar 生命周期与安装包集成尚未完成。

## 功能

- 伴侣资料、自定义头像、聊天历史和断线事件补收。
- Planner / Replyer 对话、长期记忆和表达学习。
- 授权目录内的文本文件工具、写入确认与本地提醒。
- 可选图像生成与语音回复，支持 MiniMax 和 OpenAI 兼容 TTS。
- 模型服务商预设、音色选择与试听、本地数据备份和导入。

只需配置聊天模型即可开始使用。语音、生图和向量检索均为可选扩展；没有配置向量模型时，长期记忆使用本地关键词检索。聊天服务需要支持工具调用与 JSON 结构化输出，模型请求由本地 Agent 发往你配置的服务商。

## 目录

```text
yuni-client/
  src/           React 界面
  public/        应用静态资源
  agent/
    src/         本地 Agent、数据库迁移和模型适配
    tests/       自动化测试源码
    scripts/     可选的素材与表达库维护工具
  src-tauri/     Tauri 桌面工程
```

## 从源码运行

需要 Node.js 22.12+、Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。以下命令均从仓库根目录开始。

安装依赖：

```powershell
cd yuni-client
npm ci
uv sync --project agent --extra dev
```

创建本地环境配置：

```powershell
Copy-Item .env.example .env.local
```

将 `.env.local` 中的内容设置为：

```dotenv
VITE_APP_API=local
VITE_AGENT_BASE_URL=http://127.0.0.1:43127
VITE_AGENT_SESSION=dev-session-token-not-for-production-0001
```

在两个终端中分别运行（工作目录均为 `yuni-client`）：

```powershell
npm run agent:dev
```

```powershell
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

打开 <http://127.0.0.1:5173/>，创建伴侣，然后在「设置 → 模型配置」中填写聊天模型配置。开发 Session Token 仅用于本机联调，正式桌面启动应生成随机令牌。

仅预览界面时，将 `VITE_APP_API` 设为 `mock`，只启动前端即可。Mock 数据保存在浏览器中。

## 检查与构建

在 `yuni-client` 中执行：

```powershell
npm run lint
npm run build
npm run agent:test
```

`npm run build` 生成 Web 静态资源。桌面构建还需要 Rust、Windows C++ 编译工具和 WebView2；当前桌面工程尚未把 Python Agent 打包为可独立运行的 sidecar。

## 数据与仓库范围

- 普通开发 Agent 使用系统临时目录下的 `yuni-agent-dev/data`，模型凭据保存在该数据目录的本地配置文件中。
- API Key 以明文存于本机配置文件。不要提交凭据、个人数据库、导出备份、聊天附件或专属克隆音色素材。
- 本仓库保留源码、自动化测试源码、依赖锁、构建配置和必要使用说明；内部开发记录、设计原型、浏览器测试产物、日志及构建目录由 Git 忽略。
- 表情库初始为空，个人使用的表情包、参考音频与克隆素材不随源码提供。

代码中引用的上游项目包括 [MaiBot](https://github.com/Mai-with-u/MaiBot)。相关来源已在代码或 Agent 说明中注明。
