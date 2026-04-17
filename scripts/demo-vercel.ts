/**
 * Vercel AI SDK demo using the GitHub Copilot provider.
 *
 * Runs the same 3-tool data pipeline as demo.ts, but accessed through standard
 * `generateText` / `streamText` calls from the Vercel AI SDK.
 *
 * ## How it works
 *
 * The Copilot CLI handles the entire agent loop internally: it receives the prompt, calls
 * tools, and iterates until it reaches a final answer. `doGenerate` runs that loop in full and
 * returns only the text result with `finishReason: "stop"`, so the Vercel AI SDK never fires its
 * own tool loop. Only the first LLM request should count as a premium Copilot request.
 *
 * ## Verifying premium requests
 *
 *   npm run demo:vercel:proxy   (routes through mitmproxy)
 *
 * Open http://localhost:8081, filter api.github.com/copilot, and check X-Initiator:
 *   Round 1  → X-Initiator: user  (premium)
 *   Rounds 2+ → X-Initiator: agent (non-premium)
 */

import { generateText, streamText, tool } from "ai";
import { z } from "zod";
import { createCopilotProvider } from "../src/copilot-provider.js";

// ---------------------------------------------------------------------------
// In-memory dataset — same values as demo.ts for easy comparison
// ---------------------------------------------------------------------------

const salesData: Record<string, number[]> = {
    sales_q1: [120, 145, 132, 167, 189, 201, 178, 156, 143, 165, 190, 210],
    sales_q2: [198, 223, 241, 215, 267, 289, 256, 278, 301, 289, 312, 334],
    sales_q3: [310, 298, 325, 341, 358, 372, 389, 401, 367, 345, 378, 412],
};

// Counts tool invocations across the current run; reset before each demo section
let toolCallCount = 0;

// ---------------------------------------------------------------------------
// Tool implementations
// These are passed as `toolExecutors` to `provider.chat()` and wired into
// Copilot's `defineTool` handlers. The matching schemas live in the `tools`
// object passed to `generateText` / `streamText`.
// ---------------------------------------------------------------------------

async function lookupData({ dataset }: { dataset: string }) {
    const values = salesData[dataset] ?? [];
    console.log(`    [Tool ${++toolCallCount}] lookup_data("${dataset}") → ${values.length} values`);
    return { dataset, values, count: values.length };
}

async function calculateStats({ numbers, label }: { numbers: number[]; label: string }) {
    const total = numbers.reduce((a, b) => a + b, 0);
    const mean = parseFloat((total / numbers.length).toFixed(2));
    console.log(`    [Tool ${++toolCallCount}] calculate_stats("${label}") → mean=${mean}, total=${total}`);
    return { label, total, mean, min: Math.min(...numbers), max: Math.max(...numbers) };
}

async function compareResults({
    a,
    b,
}: {
    a: { label: string; total: number; mean: number };
    b: { label: string; total: number; mean: number };
}) {
    const diff = a.total - b.total;
    const pct = parseFloat(((Math.abs(diff) / b.total) * 100).toFixed(1));
    const higher = diff > 0 ? a.label : b.label;
    console.log(`    [Tool ${++toolCallCount}] compare_results("${a.label}" vs "${b.label}") → ${higher} wins by ${Math.abs(diff)}`);
    return { higher, lower: diff > 0 ? b.label : a.label, difference: Math.abs(diff), percentChange: pct };
}

// ---------------------------------------------------------------------------
// Tool schemas — shared by both generateText and streamText runs
// Schemas describe shape to the model; executors (above) provide the logic.
// ---------------------------------------------------------------------------

const toolSchemas = {
    lookup_data: tool({
        description: "Look up raw sales values for a named dataset (e.g. 'sales_q1').",
        inputSchema: z.object({ dataset: z.string() }),
    }),
    calculate_stats: tool({
        description: "Compute total, mean, min, and max for a list of numbers.",
        inputSchema: z.object({
            numbers: z.array(z.number()),
            label: z.string(),
        }),
    }),
    compare_results: tool({
        description: "Compare two stat results and determine which is higher and by how much.",
        inputSchema: z.object({
            a: z.object({ label: z.string(), total: z.number(), mean: z.number() }),
            b: z.object({ label: z.string(), total: z.number(), mean: z.number() }),
        }),
    }),
};

const DEMO_PROMPT =
    "Using the available tools, look up sales_q1 and sales_q2, " +
    "calculate statistics for each, then compare them. Which quarter performed better and by what percentage?";

// ---------------------------------------------------------------------------
// Demo runners
// ---------------------------------------------------------------------------

/**
 * Runs a non-streaming `generateText` call. The Copilot agent loop completes before this
 * function returns; the result is a single text string.
 */
async function runGenerateText(provider: ReturnType<typeof createCopilotProvider>, modelId: string | undefined) {
    console.log("\n--- generateText (non-streaming) ---");
    toolCallCount = 0;

    const model = provider.chat(modelId, {
        toolExecutors: {
            lookup_data: lookupData as (args: unknown) => Promise<unknown>,
            calculate_stats: calculateStats as (args: unknown) => Promise<unknown>,
            compare_results: compareResults as (args: unknown) => Promise<unknown>,
        },
    });

    const start = Date.now();

    const { text, usage } = await generateText({
        model,
        tools: toolSchemas,
        messages: [{ role: "user", content: DEMO_PROMPT }],
    });

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    console.log("\nFinal response:");
    console.log(text);
    console.log(`\nTool calls: ${toolCallCount} | Input tokens: ${usage.inputTokens} | Output tokens: ${usage.outputTokens} | Time: ${elapsed}s`);
}

/**
 * Runs a streaming `streamText` call. Text arrives incrementally via `result.textStream`;
 * usage is read after the stream closes.
 */
async function runStreamText(provider: ReturnType<typeof createCopilotProvider>, modelId: string | undefined) {
    console.log("\n--- streamText (streaming) ---");
    toolCallCount = 0;

    const model = provider.chat(modelId, {
        toolExecutors: {
            lookup_data: lookupData as (args: unknown) => Promise<unknown>,
            calculate_stats: calculateStats as (args: unknown) => Promise<unknown>,
            compare_results: compareResults as (args: unknown) => Promise<unknown>,
        },
    });

    const start = Date.now();

    const result = streamText({
        model,
        tools: toolSchemas,
        messages: [{ role: "user", content: DEMO_PROMPT }],
    });

    process.stdout.write("\nStreaming response: ");
    for await (const chunk of result.textStream) {
        process.stdout.write(chunk);
    }
    process.stdout.write("\n");

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    const usage = await result.usage;
    console.log(`\nTool calls: ${toolCallCount} | Input tokens: ${usage.inputTokens} | Output tokens: ${usage.outputTokens} | Time: ${elapsed}s`);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
    console.log("=".repeat(60));
    console.log("  Vercel AI SDK + GitHub Copilot Provider Demo");
    console.log("=".repeat(60));
    console.log("Tools execute inside the Copilot CLI agent loop.");
    console.log("Only the first LLM request should be premium.\n");

    const provider = createCopilotProvider({ logLevel: "warning" });
    await provider.start();

    // Pass undefined to let the CLI pick its default model, or override via COPILOT_MODEL env var
    const modelId = process.env.COPILOT_MODEL;
    console.log(`Model: ${modelId ?? "(Copilot default)"}\n`);

    await runGenerateText(provider, modelId);
    await runStreamText(provider, modelId);

    await provider.stop();

    console.log("\n" + "=".repeat(60));
    console.log("  Verification");
    console.log("=".repeat(60));
    console.log("To confirm only 1 premium request was consumed:");
    console.log("  1. Check github.com/settings/copilot/usage");
    console.log("  2. Or run with mitmproxy:");
    console.log("     npm run demo:vercel:proxy");
    console.log("     Open http://localhost:8081, filter api.github.com/copilot");
    console.log("     Round 1: X-Initiator: user  (premium)");
    console.log("     Rounds 2+: X-Initiator: agent (non-premium)");
    console.log("=".repeat(60));
}

main().catch((err) => {
    console.error("Error:", err);
    process.exit(1);
});
