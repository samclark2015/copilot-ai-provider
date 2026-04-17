/**
 * Vercel AI SDK provider backed by the GitHub Copilot SDK.
 *
 * ## Architecture
 *
 * The Copilot CLI uses an **agent loop**: it receives a prompt, calls tools as needed, and
 * iterates internally until it produces a final answer. This is structurally different from the
 * Vercel AI SDK's **controller loop**, where `generateText` calls `doGenerate` repeatedly until
 * no more tool calls are returned.
 *
 * We resolve this mismatch by running the Copilot agent loop **in full** inside a single
 * `doGenerate` call. `doGenerate` returns only the final text content with `finishReason: "stop"`,
 * so the Vercel AI SDK never fires its own tool loop. All tool iterations happen inside the Copilot
 * CLI — only the initial user message counts as a premium Copilot request.
 *
 * ## Tool executor bridge
 *
 * `LanguageModelV3CallOptions.tools` carries tool schemas only, not `execute` functions. To wire
 * up executors, pass them to `provider.chat()` as `toolExecutors`. `buildCopilotTools` then zips
 * the per-call schemas with the pre-registered executors to produce `defineTool` definitions.
 */

import {
    CopilotClient,
    defineTool,
    approveAll,
    type CopilotClientOptions,
} from "@github/copilot-sdk";
import type {
    LanguageModelV3,
    LanguageModelV3CallOptions,
    LanguageModelV3GenerateResult,
    LanguageModelV3StreamResult,
    LanguageModelV3StreamPart,
    LanguageModelV3Usage,
    LanguageModelV3FinishReason,
    LanguageModelV3FunctionTool,
    LanguageModelV3Message,
} from "@ai-sdk/provider";

/** Map of tool name → async executor. Passed to `provider.chat()` alongside tool schemas. */
export type ToolExecutors = Record<string, (args: unknown) => Promise<unknown>>;

// ---------------------------------------------------------------------------
// Prompt helpers
// ---------------------------------------------------------------------------

/**
 * Extracts plain text from a content-parts array, ignoring non-text parts.
 * The wide `ReadonlyArray<{ type: string }>` input type accepts any role's `.content` directly.
 */
function extractTextFromParts(parts: ReadonlyArray<{ type: string }>): string {
    return parts
        .filter((p): p is { type: "text"; text: string } => p.type === "text" && "text" in p)
        .map((p) => p.text)
        .join("");
}

/**
 * Flattens a Vercel AI message array into a single `userPrompt` string and an optional
 * `systemContent` string.
 *
 * The Copilot SDK's `session.send()` takes a single prompt string, so multi-turn history is
 * serialised as "User: …\n\nAssistant: …" turns. For the common single-turn case this is just
 * the user message text.
 */
function extractPrompt(messages: LanguageModelV3Message[]): {
    systemContent: string | undefined;
    userPrompt: string;
} {
    let systemContent: string | undefined;
    const turns: string[] = [];

    for (const msg of messages) {
        if (msg.role === "system") {
            systemContent = msg.content;
        } else if (msg.role === "user") {
            const text = extractTextFromParts(msg.content);
            if (text) turns.push(`User: ${text}`);
        } else if (msg.role === "assistant") {
            const text = extractTextFromParts(msg.content);
            if (text) turns.push(`Assistant: ${text}`);
        } else if (msg.role === "tool") {
            for (const part of msg.content) {
                if (part.type === "tool-result") {
                    const output = part.output;
                    // `output.value` is the text field for type === "text" results
                    const text = output.type === "text" ? output.value : JSON.stringify(output);
                    turns.push(`Tool result (${part.toolName}): ${text}`);
                }
            }
        }
    }

    return {
        systemContent,
        userPrompt: turns.join("\n\n"),
    };
}

// ---------------------------------------------------------------------------
// Usage helper
// ---------------------------------------------------------------------------

interface TokenCounts {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
}

/**
 * Converts raw token counts into the nested `LanguageModelV3Usage` shape.
 * Zero values become `undefined` to signal "not tracked" rather than "zero tokens".
 */
function makeUsage({ input, output, cacheRead, cacheWrite }: TokenCounts): LanguageModelV3Usage {
    return {
        inputTokens: {
            total: input || undefined,
            // noCache = total input minus what came from cache
            noCache: input - cacheRead || undefined,
            cacheRead: cacheRead || undefined,
            cacheWrite: cacheWrite || undefined,
        },
        outputTokens: { total: output || undefined, text: undefined, reasoning: undefined },
    };
}

// ---------------------------------------------------------------------------
// Finish reason
// ---------------------------------------------------------------------------

/**
 * The provider always reports `"stop"` because `doGenerate` runs the full Copilot agent loop and
 * only returns after the CLI has produced its final answer — there are never tool-call chunks in
 * the response for the Vercel AI SDK to act on.
 */
const STOP_REASON: LanguageModelV3FinishReason = { unified: "stop", raw: "stop" };

// ---------------------------------------------------------------------------
// Core model implementation
// ---------------------------------------------------------------------------

/**
 * Vercel AI `LanguageModelV3` backed by a `CopilotClient` session.
 *
 * Construct via `CopilotProvider.chat()` rather than directly.
 */
class CopilotLanguageModel implements LanguageModelV3 {
    readonly specificationVersion = "v3" as const;
    readonly provider = "github-copilot";
    readonly supportedUrls: Record<string, RegExp[]> = {};

    constructor(
        readonly modelId: string,
        private readonly client: CopilotClient,
        private readonly executors: ToolExecutors = {}
    ) {}

    /**
     * Builds Copilot `defineTool` definitions by zipping `options.tools` schemas (from the Vercel
     * AI call site) with the pre-registered `executors` (from the model constructor).
     *
     * `{ ...t.inputSchema }` spreads the JSONSchema7 object into a plain `Record<string, unknown>`,
     * which is what `defineTool` expects for `parameters`.
     */
    private buildCopilotTools(tools: (LanguageModelV3FunctionTool | { type: string })[]) {
        return tools
            .filter((t): t is LanguageModelV3FunctionTool => t.type === "function")
            .map((t) =>
                defineTool(t.name, {
                    description: t.description ?? "",
                    parameters: { ...t.inputSchema },
                    handler: async (args) => {
                        const exec = this.executors[t.name];
                        if (exec) {
                            try {
                                return JSON.stringify(await exec(args));
                            } catch (err) {
                                return `Tool ${t.name} error: ${String(err)}`;
                            }
                        }
                        return `No executor registered for tool "${t.name}"`;
                    },
                })
            );
    }

    /**
     * Runs the full Copilot agent loop synchronously and returns the final text response.
     *
     * Because the CLI iterates tool calls internally, this method always returns
     * `finishReason: "stop"` — the Vercel AI SDK never sees tool-call content and never triggers
     * its own tool loop. Only the first LLM request is a premium Copilot request.
     */
    async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
        const { systemContent, userPrompt } = extractPrompt(options.prompt);
        const copilotTools = this.buildCopilotTools(options.tools ?? []);

        const counts: TokenCounts = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

        const session = await this.client.createSession({
            model: this.modelId === "default" ? undefined : this.modelId,
            tools: copilotTools,
            systemMessage: systemContent ? { mode: "append", content: systemContent } : undefined,
            onPermissionRequest: approveAll,
            infiniteSessions: { enabled: false },
        });

        session.on("assistant.usage", (event) => {
            const d = event.data;
            counts.input += d.inputTokens ?? 0;
            counts.output += d.outputTokens ?? 0;
            counts.cacheRead += d.cacheReadTokens ?? 0;
            counts.cacheWrite += d.cacheWriteTokens ?? 0;
        });

        const reply = await session.sendAndWait({ prompt: userPrompt });
        await session.disconnect();

        return {
            content: [{ type: "text", text: reply?.data.content ?? "" }],
            finishReason: STOP_REASON,
            usage: makeUsage(counts),
            warnings: [],
            request: { body: { model: this.modelId, prompt: userPrompt } },
            response: { body: reply?.data ?? null },
        };
    }

    /**
     * Streaming variant of `doGenerate`. Text deltas arrive via `assistant.message_delta` events
     * and are forwarded as `text-delta` stream parts. The `session.idle` event fires once all tool
     * iterations are complete and the CLI has no more work to do — that is the signal to close the
     * stream.
     */
    async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
        const { systemContent, userPrompt } = extractPrompt(options.prompt);
        const copilotTools = this.buildCopilotTools(options.tools ?? []);
        const client = this.client;
        // Resolve sentinel before entering the ReadableStream closure
        const modelId = this.modelId === "default" ? undefined : this.modelId;

        const stream = new ReadableStream<LanguageModelV3StreamPart>({
            start(controller) {
                void (async () => {
                    try {
                        controller.enqueue({ type: "stream-start", warnings: [] });

                        const counts: TokenCounts = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
                        let textStarted = false;
                        let finished = false;

                        const session = await client.createSession({
                            model: modelId,
                            tools: copilotTools,
                            streaming: true,
                            systemMessage: systemContent ? { mode: "append", content: systemContent } : undefined,
                            onPermissionRequest: approveAll,
                            infiniteSessions: { enabled: false },
                        });

                        session.on("assistant.usage", (event) => {
                            const d = event.data;
                            counts.input += d.inputTokens ?? 0;
                            counts.output += d.outputTokens ?? 0;
                            counts.cacheRead += d.cacheReadTokens ?? 0;
                            counts.cacheWrite += d.cacheWriteTokens ?? 0;
                        });

                        session.on("assistant.message_delta", (event) => {
                            const delta = event.data.deltaContent;
                            if (!delta) return;
                            if (!textStarted) {
                                controller.enqueue({ type: "text-start", id: "text-0" });
                                textStarted = true;
                            }
                            controller.enqueue({ type: "text-delta", id: "text-0", delta });
                        });

                        // session.idle fires after the CLI finishes all tool iterations
                        session.on("session.idle", async () => {
                            if (finished) return;
                            finished = true;
                            if (textStarted) {
                                controller.enqueue({ type: "text-end", id: "text-0" });
                            }
                            controller.enqueue({
                                type: "finish",
                                finishReason: STOP_REASON,
                                usage: makeUsage(counts),
                            });
                            controller.close();
                            await session.disconnect();
                        });

                        await session.send({ prompt: userPrompt });
                    } catch (error) {
                        controller.enqueue({ type: "error", error });
                        controller.close();
                    }
                })();
            },
        });

        return { stream };
    }
}

// ---------------------------------------------------------------------------
// Provider factory
// ---------------------------------------------------------------------------

/** Options forwarded directly to `CopilotClient`. */
export interface CopilotProviderOptions extends CopilotClientOptions {
    // intentionally empty — forwards all CopilotClientOptions
}

export interface CopilotModelOptions {
    /**
     * Async executor functions keyed by tool name. These are registered at model-construction time
     * and bridged into Copilot `defineTool` handlers when `doGenerate` / `doStream` runs.
     *
     * Pass these alongside the matching tool schemas in `generateText({ tools: { … } })`.
     */
    toolExecutors?: ToolExecutors;
}

export interface CopilotProvider {
    /** Start the underlying `CopilotClient` (authenticates and spawns the CLI). */
    start(): Promise<void>;
    /** Gracefully shut down the `CopilotClient`. */
    stop(): Promise<unknown[]>;
    /**
     * Return a `LanguageModelV3` for use with `generateText` / `streamText`.
     *
     * @param modelId - Copilot model identifier (e.g. `"gpt-4.1"`). Omit to let the CLI choose
     *   its default.
     * @param options - Per-model options, including tool executors.
     */
    chat(modelId?: string, options?: CopilotModelOptions): CopilotLanguageModel;
}

/**
 * Create a Vercel AI SDK–compatible provider backed by the GitHub Copilot SDK.
 *
 * ```ts
 * const provider = createCopilotProvider({ logLevel: "warning" });
 * await provider.start();
 *
 * const model = provider.chat(undefined, {
 *   toolExecutors: { my_tool: async (args) => { … } },
 * });
 *
 * const { text } = await generateText({ model, tools: { … }, messages: […] });
 * await provider.stop();
 * ```
 */
export function createCopilotProvider(options?: CopilotProviderOptions): CopilotProvider {
    const client = new CopilotClient(options);
    return {
        start: () => client.start(),
        stop: () => client.stop(),
        chat(modelId?: string, modelOptions?: CopilotModelOptions) {
            // "default" sentinel lets the session creation skip the model field entirely
            return new CopilotLanguageModel(modelId ?? "default", client, modelOptions?.toolExecutors ?? {});
        },
    };
}
