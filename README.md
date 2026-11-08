# Yuni

**一个本地优先的 AI 陪伴项目，让你按照自己的想法创建、了解并陪伴一个独一无二的角色。**

Yuni 将聊天界面、本地 Agent 和记忆管理放在一起：你可以设定伴侣的人设与相处边界，在日常对话中积累记忆，也可以按需接入语音、图像生成和本地文件工具。

> **项目状态：开发中。** 目前可以从源码运行 Web 界面和本地 Agent。Tauri 桌面窗口已建立，但 Agent 的 sidecar 启动及安装包集成尚未完成，暂不能作为开箱即用的桌面安装包使用。

## 功能一览

| 功能 | 说明 |
| --- | --- |
| **创建伴侣** | 设置名称、头像、人设、称呼和互动边界，随时调整资料。 |
| **日常聊天** | 基于人物资料、上下文和记忆生成回复；支持历史记录、连续发送和断线后补收事件。 |
| **长期记忆** | 整理与管理聊天中形成的记忆，支持查看、编辑、删除及历史记录。 |
| **表达学习** | 从用户的聊天表达中学习风格，在适合的场景下提供参考。 |
| **主动聊天** | 可设置主动聊天频率与免打扰时段，也可以随时关闭。 |
| **语音回复** | 可选接入 MiniMax 原生语音或兼容 OpenAI TTS 的服务，选择音色并试听。 |
| **图像生成** | 可选接入兼容的生图服务，让伴侣按请求生成图片或自拍。 |
| **本地工具** | 在授权目录内查找、读取和编辑文本文件，也可以设置本地提醒；写入等操作需要确认。 |
| **数据管理** | 在本地查看聊天和记忆，并支持导出、导入与清空数据。 |

## 模型配置

在「**设置 → 模型配置**」中配置自己的模型服务。只需配置聊天模型即可开始使用：

| 用途 | 是否必需 | 作用 |
| --- | --- | --- |
| 聊天模型 | 必需 | 支持日常对话、任务规划和记忆整理；服务需要兼容 OpenAI 格式的工具调用和 JSON 结构化输出。 |
| 向量检索（Embedding） | 可选 | 增强长期记忆和表达的语义检索。未配置时，长期记忆使用本地关键词检索，表达选择使用加权回退。 |
| 图像生成 | 可选 | 启用图片生成能力，需使用兼容的图像生成服务。 |
| 语音合成 | 可选 | 启用语音回复、音色选择与试听；MiniMax 用户可以读取账号中的可用音色。 |

未启用的可选模型功能不会向 Planner 开放对应工具或语音参数。模型请求由本地 Agent 发送到你配置的服务商，**使用外部服务时可能产生费用**，可用模型、音色和接口以服务商实际提供的为准。

## 快速开始

以下步骤以 **Windows PowerShell** 为例。需要安装：

- [Node.js](https://nodejs.org/) 22.12 或更新版本，以及 npm。
- Python 3.11 或更新版本。
- Python 包管理工具 [uv](https://docs.astral.sh/uv/)。
- 一个支持工具调用的聊天模型服务及其 API Key。

### 1. 获取代码并安装依赖

```powershell
git clone https://github.com/qingpoa/yuni.git
cd yuni/yuni-client

npm ci
uv sync --project agent --extra dev
```

### 2. 配置前端与本地 Agent 的连接

在 `yuni-client` 目录下复制示例配置：

```powershell
Copy-Item .env.example .env.local
```

编辑 `.env.local`，设置以下内容：

```dotenv
VITE_APP_API=local
VITE_AGENT_BASE_URL=http://127.0.0.1:43127
VITE_AGENT_SESSION=dev-session-token-not-for-production-0001
```

这里的 Session Token 是**仅供本机开发使用的固定值**，用于前端连接本地 Agent；它不是模型服务商的 API Key。请勿将模型 API Key 写入 `VITE_*` 环境变量。

### 3. 分别启动 Agent 和前端

打开两个 PowerShell 窗口，分别进入 `yuni-client` 目录。

第一个窗口启动本地 Agent：

```powershell
npm run agent:dev
```

第二个窗口启动前端：

```powershell
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

在浏览器中打开 <http://127.0.0.1:5173/>，创建伴侣后，进入「设置 → 模型配置」填写聊天模型的服务地址、模型名和 API Key，即可开始聊天。完成后停止两个终端即可关闭服务。

### 只想预览界面？

将 `.env.local` 中的 `VITE_APP_API` 设为 `mock`，然后只启动前端：

```powershell
npm run dev
```

Mock 模式使用浏览器中的演示数据，不连接真实模型，也不需要启动 Python Agent。

## 数据与隐私

- 伴侣资料、聊天记录和记忆由本地 SQLite 保存。普通 Agent 开发模式的数据目录为系统临时目录下的 `yuni-agent-dev/data`。
- 模型 API Key 保存在本机 Agent 数据目录的配置文件中，**以明文存储**。请勿分享这些文件或将它们提交到 Git。
- 本地保存并不意味着模型推理在本机完成：使用外部模型服务时，相关请求会发送到你配置的服务商。
- 文件工具仅能访问已授权的目录，文件写入等操作仍需单独确认。可在设置中添加或撤销目录授权。
- 表情库初始为空；个人表情素材、参考音频和克隆音色素材不包含在本仓库中。
- 建议定期通过设置页导出备份，并妥善保存备份文件。

## 常见问题

**已经启动前端，但页面提示本地服务没有准备好？**

确认 Agent 终端仍在运行，且 `.env.local` 中设置了 `VITE_APP_API=local`、正确的本地地址和开发 Session Token。修改 `.env.local` 后需要重启前端开发服务器。

**没有 Embedding、语音或生图模型能使用吗？**

可以。聊天模型是必需项；其他模型都是可选扩展。未配置 Embedding 时，长期记忆仍可使用本地关键词检索。

**配置了 API Key 还是无法聊天？**

检查服务地址、模型名、Key 和账户状态，并确认服务支持 OpenAI 兼容的工具调用。设置页提供常用服务商预设，实际可用模型请以对应服务商为准。

**能否直接下载桌面安装包？**

当前尚未完成 Python Agent 的桌面 sidecar 集成，请先按照本文的 Web + 本地 Agent 方式从源码运行。

## 开发与验证

在 `yuni-client` 目录中执行：

```powershell
npm run lint        # 前端静态检查
npm run build       # 构建前端静态资源
npm run agent:test  # 运行 Agent 自动化测试
```

项目主要目录：

```text
yuni-client/
├── src/               React 页面、组件、状态管理与 API 客户端
├── public/            前端静态资源
├── agent/
│   ├── src/           本地 Agent、模型适配与数据库迁移
│   ├── tests/         Agent 自动化测试
│   └── scripts/       可选的素材与表达库维护脚本
└── src-tauri/         Tauri 桌面工程
```

前端细节见 [Yuni Client](yuni-client/README.md)，本地 Agent 的运行方式和实现说明见 [Yuni Agent](yuni-client/agent/README.md)。

## 致谢与来源

Yuni 的部分 Planner / Replyer 提示词、表达学习和分条规则参考了 [MaiBot](https://github.com/Mai-with-u/MaiBot)，具体来源与实现范围见 [Agent 说明](yuni-client/agent/README.md#来源)和代码注释。
