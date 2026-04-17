# copilot-ai-provider

A [Vercel AI SDK](https://sdk.vercel.ai) provider backed by the [GitHub Copilot SDK](https://github.com/github/copilot-sdk). Use `generateText` and `streamText` with GitHub Copilot models while keeping tool-call follow-ups as **non-premium** requests.

## Installation

```bash
npm install @github/copilot-sdk @ai-sdk/provider ai
```

## Usage

```ts
import { generateText, tool } from "ai";
import { z } from "zod";
import { createCopilotProvider } from "./src/index.js";

const provider = createCopilotProvider({ logLevel: "warning" });
await provider.start();

const model = provider.chat(undefined, {
  toolExecutors: {
    get_weather: async ({ city }) => ({ temp: 72, condition: "sunny" }),
  },
});

const { text } = await generateText({
  model,
  tools: {
    get_weather: tool({
      description: "Get the current weather for a city.",
      inputSchema: z.object({ city: z.string() }),
    }),
  },
  messages: [{ role: "user", content: "What's the weather in Paris?" }],
});

console.log(text);
await provider.stop();
```

### Streaming

```ts
import { streamText } from "ai";

const result = streamText({ model, tools, messages });
for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### Specifying a model

Pass a model ID string to `provider.chat()`. Omit it (or pass `undefined`) to let the Copilot CLI use its default.

```ts
const model = provider.chat("gpt-4.1", { toolExecutors: { … } });
```

Override at runtime via the `COPILOT_MODEL` environment variable (used by the demo scripts).

## How tool executors work

`generateText` accepts tool schemas but not `execute` functions — those live in the Vercel AI SDK's own tool loop, which never runs here. Instead, pass executor functions to `provider.chat()`:

```ts
const model = provider.chat(undefined, {
  toolExecutors: {
    my_tool: async (args) => { /* return result */ },
  },
});
```

`buildCopilotTools` zips the schemas (from the `generateText` call) with the executors (from the constructor) to produce `defineTool` definitions that the Copilot CLI invokes directly.

## Demo scripts

```bash
npm run demo           # Raw Copilot SDK (baseline)
npm run demo:vercel    # generateText + streamText via this provider
```

Both run a 3-tool data pipeline: `lookup_data` → `calculate_stats` → `compare_results`.

### Verifying premium requests with mitmproxy

```bash
# 1. Install and start mitmproxy
brew install mitmproxy
mitmweb --listen-port 8080 --web-port 8081

# 2. Trust the CA cert (macOS)
sudo security add-trusted-cert -d -r trustRoot \
  -k /Library/Keychains/System.keychain ~/.mitmproxy/mitmproxy-ca-cert.pem

# 3. Run with proxy
npm run demo:vercel:proxy

# 4. Open http://localhost:8081, filter api.github.com/copilot
#    Round 1  → X-Initiator: user  (premium)
#    Rounds 2+ → X-Initiator: agent (non-premium)
```

## API

### `createCopilotProvider(options?)`

Creates a provider instance. `options` are forwarded to `CopilotClient`.

| Method | Description |
|---|---|
| `start()` | Authenticate and start the Copilot CLI process |
| `stop()` | Gracefully shut down |
| `chat(modelId?, options?)` | Return a `LanguageModelV3` for use with `generateText` / `streamText` |

### `CopilotModelOptions`

| Field | Type | Description |
|---|---|---|
| `toolExecutors` | `Record<string, (args: unknown) => Promise<unknown>>` | Executor functions keyed by tool name |
