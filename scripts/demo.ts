/**
 * GitHub Copilot SDK - Premium Request Test
 *
 * Hypothesis: The Copilot CLI (which backs this SDK) sets X-Initiator: agent for
 * tool-call follow-up requests, unlike vscode.lm which always sets X-Initiator: user.
 *
 * This means only the initial user prompt should consume a premium Copilot request,
 * and subsequent tool-call iterations should be non-premium "agent" requests.
 *
 * Context: https://github.com/posit-dev/positron/issues/12016
 *
 * To verify headers, run with mitmproxy:
 *   npm run demo:proxy
 *
 * Then inspect requests to api.github.com/copilot/ for the X-Initiator header.
 * Expected: first request = "user", follow-ups = "agent"
 */

import { CopilotClient, approveAll, defineTool, type SessionEvent } from "@github/copilot-sdk";
import { z } from "zod";

// --- Tools requiring multiple sequential calls ---

// Step 1: Look up data
const lookupDataTool = defineTool("lookup_data", {
    description: "Look up raw data for a given dataset name. Returns raw numbers.",
    parameters: z.object({
        dataset: z.string().describe("Name of dataset to look up (e.g. 'sales_q1', 'sales_q2')"),
    }),
    handler: ({ dataset }) => {
        const data: Record<string, number[]> = {
            sales_q1: [120, 145, 132, 167, 189, 201, 178, 156, 143, 165, 190, 210],
            sales_q2: [198, 223, 241, 215, 267, 289, 256, 278, 301, 289, 312, 334],
            sales_q3: [310, 298, 325, 341, 358, 372, 389, 401, 367, 345, 378, 412],
        };
        const values = data[dataset] ?? [];
        console.log(`    [Tool] lookup_data("${dataset}") → ${values.length} values`);
        return JSON.stringify({ dataset, values, count: values.length });
    },
});

// Step 2: Calculate statistics
const calculateStatsTool = defineTool("calculate_stats", {
    description: "Calculate descriptive statistics (mean, min, max, total) for a list of numbers.",
    parameters: z.object({
        numbers: z.array(z.number()).describe("Array of numbers to analyse"),
        label: z.string().describe("Label for these numbers"),
    }),
    handler: ({ numbers, label }) => {
        const total = numbers.reduce((a, b) => a + b, 0);
        const mean = total / numbers.length;
        const min = Math.min(...numbers);
        const max = Math.max(...numbers);
        console.log(`    [Tool] calculate_stats("${label}") → mean=${mean.toFixed(1)}, total=${total}`);
        return JSON.stringify({ label, total, mean: parseFloat(mean.toFixed(2)), min, max });
    },
});

// Step 3: Compare two stat results
const compareResultsTool = defineTool("compare_results", {
    description: "Compare two sets of statistics and determine which is higher and by how much.",
    parameters: z.object({
        a: z.object({ label: z.string(), total: z.number(), mean: z.number() }),
        b: z.object({ label: z.string(), total: z.number(), mean: z.number() }),
    }),
    handler: ({ a, b }) => {
        const diff = a.total - b.total;
        const pctChange = ((diff / b.total) * 100).toFixed(1);
        const higher = diff > 0 ? a.label : b.label;
        console.log(`    [Tool] compare_results("${a.label}" vs "${b.label}") → ${higher} is higher by ${Math.abs(diff)}`);
        return JSON.stringify({
            higher,
            lower: diff > 0 ? b.label : a.label,
            difference: Math.abs(diff),
            percentChange: parseFloat(pctChange),
        });
    },
});

// --- Event tracking ---

type RequestTiming = { type: string; ts: number };

function trackEvents(session: { on: (handler: (event: SessionEvent) => void) => () => void }): {
    events: RequestTiming[];
    toolCalls: string[];
    messageCount: number;
} {
    const tracked = { events: [] as RequestTiming[], toolCalls: [] as string[], messageCount: 0 };

    session.on((event: SessionEvent) => {
        tracked.events.push({ type: event.type, ts: Date.now() });

        if (event.type === "tool.execution_start") {
            const toolName = (event.data as { toolName?: string }).toolName ?? "unknown";
            tracked.toolCalls.push(toolName);
            console.log(`  → tool.execution_start: ${toolName}`);
        } else if (event.type === "tool.execution_complete") {
            const toolName = (event.data as { toolName?: string }).toolName ?? "unknown";
            console.log(`  ← tool.execution_complete: ${toolName}`);
        } else if (event.type === "assistant.message") {
            tracked.messageCount++;
        } else if (event.type === "user.message") {
            console.log("  ↑ user.message sent");
        }
    });

    return tracked;
}

// --- Main ---

async function main() {
    console.log("=".repeat(60));
    console.log("  GitHub Copilot SDK — Premium Request Test");
    console.log("=".repeat(60));
    console.log();
    console.log("Testing whether tool-call follow-ups consume premium requests.");
    console.log("If hypothesis holds: only 1 premium request for the whole run.");
    console.log("If vscode.lm-style: 1 request per LLM call (user prompt + each");
    console.log("  tool-call iteration = N+1 premium requests).");
    console.log();

    const client = new CopilotClient({
        logLevel: "warning",
    });

    await client.start();
    console.log("Client started.\n");

    const session = await client.createSession({
        tools: [lookupDataTool, calculateStatsTool, compareResultsTool],
        onPermissionRequest: approveAll,
    });

    console.log(`Session ID: ${session.sessionId}\n`);

    const tracked = trackEvents(session);

    // A prompt that requires: 2x lookup_data, 2x calculate_stats, 1x compare_results = 5 tool calls
    const prompt =
        "Using the available tools, look up sales_q1 and sales_q2 datasets, " +
        "calculate statistics for each, then compare them. Summarise which quarter " +
        "performed better and by what percentage.";

    console.log("Prompt:");
    console.log(`  "${prompt}"`);
    console.log();
    console.log("--- Events ---");

    const startTime = Date.now();
    const reply = await session.sendAndWait({ prompt });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    console.log();
    console.log("--- Final Response ---");
    console.log(reply?.data.content ?? "(no response)");

    console.log();
    console.log("=".repeat(60));
    console.log("  Summary");
    console.log("=".repeat(60));
    console.log(`  Tool calls made        : ${tracked.toolCalls.length} (${tracked.toolCalls.join(", ")})`);
    console.log(`  LLM API calls (rounds) : ${tracked.messageCount}`);
    console.log(`    Each "assistant message" = 1 HTTP request to the Copilot API.`);
    console.log(`    With vscode.lm: ALL ${tracked.messageCount} would be premium (X-Initiator: user).`);
    console.log(`    With this SDK:  only the 1st should be premium (X-Initiator: user),`);
    console.log(`                    rounds 2–${tracked.messageCount} should be agent (non-premium).`);
    console.log(`  Elapsed                : ${elapsed}s`);
    console.log();
    console.log("To verify premium request consumption:");
    console.log("  1. Check GitHub Copilot usage dashboard (github.com/settings/copilot/usage)");
    console.log(`     If SDK is correct: 1 premium request consumed (not ${tracked.messageCount})`);
    console.log();
    console.log("  2. Inspect X-Initiator headers with mitmproxy:");
    console.log("     a. brew install mitmproxy");
    console.log("     b. mitmweb --listen-port 8080 --web-port 8081  (in another terminal)");
    console.log("     c. Trust the CA: sudo security add-trusted-cert -d -r trustRoot \\");
    console.log("          -k /Library/Keychains/System.keychain ~/.mitmproxy/mitmproxy-ca-cert.pem");
    console.log("     d. npm run demo:proxy");
    console.log("     e. Open http://localhost:8081 and filter for api.github.com/copilot");
    console.log("        Round 1: X-Initiator: user  (premium)");
    console.log(`        Rounds 2–${tracked.messageCount}: X-Initiator: agent (non-premium) ← the key result`);
    console.log("=".repeat(60));

    await session.disconnect();
    await client.stop();
}

main().catch((err) => {
    console.error("Error:", err);
    process.exit(1);
});
