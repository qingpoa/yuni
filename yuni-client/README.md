# Yuni Client

React + TypeScript + Vite 前端，配合本地 Python Agent 使用。完整源码启动步骤见[仓库说明](../README.md)。

## 安装与运行

```powershell
npm ci
Copy-Item .env.example .env.local
```

本地 Agent 模式需要在 `.env.local` 中设置：

```dotenv
VITE_APP_API=local
VITE_AGENT_BASE_URL=http://127.0.0.1:43127
VITE_AGENT_SESSION=dev-session-token-not-for-production-0001
```

分别打开两个终端：

```powershell
uv sync --project agent --extra dev
npm run agent:dev
```

```powershell
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

访问 <http://127.0.0.1:5173/>。设置页只需配置聊天模型，向量检索、图像生成与语音合成为可选扩展。

仅查看演示界面时，将 `VITE_APP_API` 改为 `mock`，无需启动 Agent。演示数据保存在浏览器 `localStorage`。

## 检查

```powershell
npm run lint
npm run build
npm run agent:test
```

`npm run build` 输出 Web 资源至 `dist/`。

## 桌面工程

`src-tauri/` 包含 Tauri 2 窗口与原生插件配置。安装 Rust、Windows C++ 编译工具与 WebView2 后可使用 `npm run desktop:dev` 调试桌面窗口。Python Agent 的 sidecar 启动与安装包集成仍待完成，当前源码运行请先使用上述 Web + Agent 方式。

API Key 仅提交给本地 Agent，不保存在前端持久化状态或浏览器存储中。本地环境覆盖文件、运行数据和构建产物不提交 Git。
