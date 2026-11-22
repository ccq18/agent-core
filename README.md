# agent-core

独立、轻量、可复用的纯 TypeScript / Node.js（ESM）内嵌 Agent 运行时。

底层复用 [`@earendil-works/pi-ai`](https://github.com/earendil-works/pi)、`@earendil-works/pi-agent-core` 与 `@earendil-works/pi-mcp`，上层以极薄胶水层提供面向宿主应用的完整 Agent Harness，零 TUI / CLI / Bun / SQLite / WASM 绑定，可直接嵌入桌面端、服务端或任意 Node.js 进程。

## 核心能力

- **多协议模型路由（`ModelRouter`）**：支持业务别名（如 `auto`）与运行时热更新，显式分派 `openai-completions`、`openai-responses`、`anthropic-messages`、`google-generative-ai`，原生支持自定义网关与 Thinking 思考流。
- **物理隔离的树状会话（`JsonlTreeSessionStore`）**：按 `<dataDir>/sessions/<sessionId>.jsonl` 独立存储，与本机 CLI 工具完全隔离；支持树状 `branch()` 分支切换、可插拔 `SessionStorageAdapter` 与压缩检查点投影。
- **Plan / Build 模式与 Todo 规划**：保护 Prompt Cache 稳定前缀，通过尾部 `<system-update>` 提醒与权限叠加实现只读规划模式；内置 `todowrite` / `todoread` 与实时 `plan_update` 事件。
- **默认全许可的权限门控（`PermissionEngine`）**：默认策略全许可（`allow`）不拦截；当显式配置 `ask` / `deny` 规则或切入 `plan` 模式时，支持通配符匹配、异步审批挂起（`once` / `always` / `reject` + 纠偏反馈）与持久允许规则。
- **两段式上下文压缩（`CompactionEngine`）**：基于真实 usage 锚点与估算触发自动/手动/溢出压缩，安全切点不破坏 `tool_result` 配对，自动生成结构化检查点摘要。
- **隔离上下文子代理（`SubagentManager`）**：内置 `explore`（只读）与 `general` 预设，支持同步与后台异步委托、`maxSteps` 强制文字收口、`maxDepth` 防套娃与父会话级联取消。
- **统一工具注册、输出截断、Rules / Skills 与 MCP**：超限工具输出自动落盘截断（默认 2,000 行 / 50 KiB）；支持全局与项目 `AGENTS.md` / `CLAUDE.md` 规则发现、`SKILL.md` 元数据索引与按需加载（`load_skill`）；通过 `@earendil-works/pi-mcp` 桥接 stdio / Streamable HTTP MCP 工具。
- **统一流式事件契约（`AgentCoreEvent`）**：将底层事件归约为单调递增的强类型事件流（`turn_*`、`thinking_*`、`text_*`、`plan_update`、`tool_*`、`approval_*`、`subagent_*`、`compaction_*`）。

## 环境要求

- Node.js `>= 22.19.0`（纯 ESM）

## 快速开始

```ts
import { createAgentRuntime } from 'agent-core';

const runtime = createAgentRuntime({
  dataDir: './.agent-data',
  models: {
    defaultAlias: 'auto',
    profiles: [
      {
        alias: 'auto',
        modelId: 'gpt-4.1',
        providerId: 'openai-gateway',
        protocol: 'openai-completions',
        baseUrl: 'https://api.openai.com/v1',
        apiKey: process.env.OPENAI_API_KEY,
        contextWindow: 128_000,
        maxTokens: 4096,
        reasoning: true,
        defaultThinkingLevel: 'medium',
      },
    ],
  },
  tools: [
    {
      name: 'echo',
      label: 'Echo',
      description: 'Return the provided message.',
      readOnly: true,
      parameters: {
        type: 'object',
        properties: { text: { type: 'string' } },
        required: ['text'],
      },
      execute: async (params) => ({
        content: [{ type: 'text', text: String(params.text) }],
      }),
    },
  ],
});

const session = await runtime.createSession({ modelAlias: 'auto' });

session.subscribe((event) => {
  if (event.type === 'text_delta') process.stdout.write(event.delta);
});

const summary = await session.prompt('请先制定一个两步计划，再调用 echo 工具。');
console.log('\nDone:', summary.state);

await runtime.dispose();
```

## 开发与测试

```bash
npm install
npm run typecheck
npm run build
npm test
```

## License

MIT
