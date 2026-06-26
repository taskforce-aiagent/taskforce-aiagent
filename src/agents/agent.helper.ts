import chalk from "chalk";
import { normalizeOutput } from "../helpers/helper.js";
import { callAIModel, ChatMessage } from "../llm/aiClient.js";
import { Agent } from "./agent.js";
import { AgentRegistry } from "./agentRegistry.js";
import { TFLog } from "../helpers/log.helper.js";
import { OutputFormat, SupportedModel } from "../configs/enum.js";
import { getLLMRouteByModel } from "../configs/aiConfig.js";
import { summarizeOldMessages } from "../helpers/summarize.helper.js";
import { Task } from "../tasks/task.js";

function interpolate(template: string, inputs: Record<string, string>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => inputs[key] || `{${key}}`);
}

function buildAgentDirectory(currentAgent: Agent): string {
  const allAgents = AgentRegistry.getAll();
  return allAgents
    .filter((a) => a.name !== currentAgent.name)
    .map((a) => `- ${a.name}: ${a.role} — ${a.goal}`)
    .join("\n");
}

export function buildUserMessage(
  inputs: Record<string, string>,
  agent: Agent,
  taskDescription: string,
  inputFromPrevious?: string,
  retryReason?: string,
  delegateReason?: string,
  outputFormat?: string,
  rePlanReason?: string
): string {
  const format = (outputFormat || OutputFormat.text).toLowerCase();

  const parts: string[] = [];

  parts.push(`Background:\n${interpolate(agent.backstory, inputs)}`);
  parts.push(`Task:\n${interpolate(taskDescription, inputs)}`);

  if (inputFromPrevious?.trim()) {
    parts.push(`Previous output:\n${inputFromPrevious.trim()}`);
  }
  if (retryReason?.trim()) {
    parts.push(`This task is being re-executed because: ${retryReason.trim()}`);
  }
  if (delegateReason?.trim()) {
    parts.push(
      `This task was delegated to you because: ${delegateReason.trim()}`
    );
  }
  if (rePlanReason?.trim()) {
    parts.push(
      `This task is being re-executed because:\n${rePlanReason.trim()}`
    );
  }

  // Output format block
  if (format === OutputFormat.text) {
    if (outputFormat) {
      parts.push(
        `Expected output format:\nReturn the result in ${outputFormat} format.`
      );
    }
  } else {
    parts.push(
      `Expected output format:\nReturn ONLY the result in ${outputFormat} format.\nDo NOT include any explanations, comments, or additional text.`
    );
  }

  return parts.join("\n\n");
}

// SYSTEM MESSAGE START

function buildIdentityBlock(
  agent: Agent,
  inputs: Record<string, string>
): string {
  const interpolatedGoal = interpolate(agent.goal, inputs);
  const interpolatedPrompt = agent.systemPrompt
    ? interpolate(agent.systemPrompt, inputs)
    : null;

  return (
    interpolatedPrompt ??
    `You are a ${agent.role}.\n\nGoal: ${interpolatedGoal}`
  );
}

function buildToolInstructions(agent: Agent): string {
  const tools = agent.toolExecutor.getTools();
  const toolInfoWithExamples = agent.toolExecutor.buildToolUsageExamples(tools);

  return `You should use tools whenever the topic includes current, missing, or external information. When in doubt, prefer tool usage. If the user prompt includes phrases like "latest", "recent", "trending", or mentions specific time frames (e.g. "last 5 years"), always consider using a tool before answering.

 
Before using any tool, ask yourself:
1. What exactly is my task?
2. Is there any missing information or uncertainty?
3. Would using a tool help me complete the task more accurately?

Available Tools:
${toolInfoWithExamples}

Tool Usage Instructions:
- Use: TOOL(toolName, {"query": "your search query"})
- Tools may be used more than once.
- Do not continue the main task until tool results are returned.
- Tool names are case-sensitive. Use them exactly as listed.

Do NOT:
- Modify the tool name
- Write your own function-like formats (e.g., toolName(...))
- Add explanations or reasoning inside the tool call`;
}

function buildDelegationInstructions(agent: Agent): string {
  return `Delegation Instructions:
  
You may delegate work to another agent ONLY IF:
- You cannot complete the task properly.
- The previous step was incomplete or unclear.
- Another agent is more suitable.
  
Use the format:
DELEGATE(agentName, "task to delegate")
  
Examples:
- Input lacks structure → delegate to someone who can outline better
- Writing is unclear → delegate to an editor
- Content is missing → delegate to someone who can fill the gaps
  
Before completing:
- Is the input ready, or should it go back?
- Are you fixing something that should have been handled earlier?

If yes, delegate instead of fixing.
  
Rules:
- Do not continue after delegating.
- Do not summarize previous work.
- Delegation is optional. Use only if necessary.
  
Available agents for delegation:
${buildAgentDirectory(agent)}`;
}

function buildGuardrails(agent: Agent): string {
  return `Rules you must follow:
  ${
    agent.guardrails
      ? agent.guardrails.map((r, i) => `${i + 1}. ${r}`).join("\n")
      : ""
  }`;
}

function buildTrainingInsights(agent: Agent): string | undefined {
  const summary = agent.trained?.final_summary?.trim();
  if (!summary || summary.length <= 5 || !summary.includes(" ")) return;
  return `Training Insights:\n- ${summary}`;
}

export function buildSystemMessage(
  inputs: Record<string, string>,
  agent: Agent,
  isTraining: boolean = false
): string {
  const today = new Date().toLocaleDateString("tr-TR", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const parts: string[] = [];

  parts.push(buildIdentityBlock(agent, inputs));

  if (agent.canUseTools()) {
    if (!agent.llmModel.supportsTools) {
      parts.push(buildToolInstructions(agent));
    }
  }

  if (
    agent.allowDelegation &&
    !agent.getDelegationDisallowed() &&
    !isTraining
  ) {
    parts.push(buildDelegationInstructions(agent));
  }

  if (agent.guardrails && agent.guardrails.length > 0) {
    parts.push(buildGuardrails(agent));
  }

  const trainingBlock = buildTrainingInsights(agent);
  if (!isTraining && trainingBlock) {
    parts.push(trainingBlock);
  }

  parts.push(`Date: ${today}`);

  return parts.join("\n\n");
}

// SYSTEM MESSAGE END

export async function toolWorker(
  output: string,
  agent: Agent,
  taskDescription: string,
  outputFormat: string,
  context: Record<string, any>
): Promise<ChatMessage[] | undefined> {
  if (agent.toolExecutor && output.includes("TOOL(")) {
    const toolCalls = Array.from(
      output.matchAll(/TOOL\(([^,]+),\s*(\{.*?\})\)/g)
    );

    if (!toolCalls.length) return;

    let enrichedOutput = output;

    const toolResponses: string[] = [];

    for (const match of toolCalls) {
      const [toolCall, toolId, toolArg] = match;
      const toolName = agent.toolExecutor.getToolNameById(toolId);
      let parsedArgs;

      try {
        parsedArgs = JSON.parse(toolArg);
      } catch (e: any) {
        return [
          {
            role: "assistant",
            content: `❌ Failed to parse tool args for ${toolName}: ${e.message}`,
          },
        ];
      }

      const toolResult = await agent.toolExecutor.executeToolById(
        toolId,
        parsedArgs
      );

      const normalized = normalizeOutput(toolResult);
      toolResponses.push(
        `Tool Response for ${toolName}("${toolArg}"):\n${normalized}`
      );
      enrichedOutput = enrichedOutput.replace(
        toolCall,
        toolResponses[toolResponses.length - 1]
      );
      if (agent.getVerbose()) {
        TFLog(
          `🔧 [Tool Used] ${toolName}("${toolArg}") by '${agent.name}'`,
          chalk.white
        );
        TFLog(`🧾 Output:\n${toolResult}\n`, chalk.gray);
      }
    }
    return [
      { role: "system", content: buildSystemMessage(context, agent) },
      {
        role: "user",
        content:
          `Tool results:\n\n${toolResponses.join("\n\n")}` +
          `\n\nNow complete this task:\n\n${taskDescription}` +
          (outputFormat ? `\n\nFormat: ${outputFormat}` : ""),
      },
    ];
  } else {
    return undefined;
  }
}

export async function recursiveToolWorker(
  output: string,
  agent: Agent,
  taskDescription: string,
  outputFormat: string,
  context: Record<string, any>,
  depth = 0,
  maxDepth = 3
): Promise<ChatMessage[] | undefined> {
  if (depth > maxDepth) return;

  const toolMessages = await toolWorker(
    output,
    agent,
    taskDescription,
    outputFormat,
    context
  );

  if (!toolMessages) return;

  const newOutput = await callAIModel(
    agent.name,
    agent.model || process.env.DEFAULT_MODEL!,
    toolMessages,
    agent.getVerbose()
  );

  if (agent.getVerbose()) {
    TFLog(
      `📤 [Agent Output] for '${agent.name}' after tool execution of task "${taskDescription}"`,
      chalk.blue
    );
    TFLog(`${newOutput}\n`, chalk.gray);
  }

  if (newOutput.includes("TOOL(")) {
    return await recursiveToolWorker(
      newOutput,
      agent,
      taskDescription,
      outputFormat,
      context,
      depth + 1,
      maxDepth
    );
  }

  return [{ role: "assistant", content: newOutput }];
}

export async function delegateTo(
  agent: Agent,
  output: string,
  inputs: Record<string, string>
): Promise<{ delegateTo: string; task: string } | null> {
  // Per-agent delegation cap is enforced by canDelegate() (Agent.MAX_DELEGATION).
  if (!agent.canDelegate()) return null;

  const match = output.match(/DELEGATE\(([^,]+),\s*"([^"]+)"\)/);
  if (!match) return null;

  const [, delegateName, delegateTask] = match;
  agent.incrementDelegation();

  if (agent["verbose"]) {
    TFLog(
      `🔁 [Delegation] ${
        agent.name
      } → ${delegateName.trim()} | Task: "${delegateTask}"`,
      chalk.red
    );
  }
  return {
    delegateTo: delegateName.trim(),
    task: delegateTask,
  };
}

export async function truncateMessagesToFitModelLimit(
  modelName: string,
  messages: ChatMessage[]
): Promise<ChatMessage[]> {
  const modelConfig = getLLMRouteByModel(modelName);
  if (!modelConfig) throw new Error(`Model not found: ${modelName}`);

  const maxTokens = modelConfig.model.maxContextTokens || 16000;
  const tokenBuffer = 500;

  const estimateTokenCount = (msg: ChatMessage) =>
    Math.ceil(msg.content.length / 4); // approx

  const systemMessage = messages.find((m) => m.role === "system");
  const otherMessages = messages.filter((m) => m.role !== "system");

  const finalMessages: ChatMessage[] = [];
  const includedMessages: ChatMessage[] = [];
  const truncatedMessages: ChatMessage[] = [];

  let totalTokens = systemMessage ? estimateTokenCount(systemMessage) : 0;
  if (systemMessage) finalMessages.push(systemMessage);

  for (let i = otherMessages.length - 1; i >= 0; i--) {
    const msg = otherMessages[i];
    const tokenCount = estimateTokenCount(msg);

    if (totalTokens + tokenCount <= maxTokens - tokenBuffer) {
      includedMessages.unshift(msg);
      totalTokens += tokenCount;
    } else {
      truncatedMessages.unshift(msg);
    }
  }

  // Only summarize if meaningful chunk was excluded
  if (truncatedMessages.length > 2) {
    const summaryMessage = await summarizeOldMessages(truncatedMessages);
    const summaryTokenCount = estimateTokenCount(summaryMessage);

    if (totalTokens + summaryTokenCount <= maxTokens - tokenBuffer) {
      finalMessages.push(summaryMessage);
      totalTokens += summaryTokenCount;
    } else {
      finalMessages.push({
        role: "system",
        content:
          "📘 Context summary omitted due to token limits. Some older dialogue has been truncated.",
      });
    }
  }

  // Add preserved messages from newest to oldest
  finalMessages.push(...includedMessages);
  return finalMessages;
}

export async function generateDynamicPlan({
  inputs,
  agents,
  model = SupportedModel.GPT_4O_MINI,
  verbose = false,
}: {
  inputs: Record<string, any>;
  agents: any[];
  model?: string;
  verbose?: boolean;
}): Promise<{ tasks: Task[]; agents: any[]; executionMode: string }> {
  const prompt = `
Given the following input:
${JSON.stringify(inputs)}

Available agents (choose agent field EXACTLY as written, case-sensitive):
${agents.map((a) => `- "${a.name}": ${a.role} — ${a.goal}`).join("\n")}

IMPORTANT RULES:
- For each task, the "agent" field MUST be chosen from the above list of agent names, **exactly as shown** (case-sensitive).
- Do NOT invent, translate, or modify agent names.
- If you are unsure, use the closest matching agent in the list.
- If a task does not fit any agent, pick the most appropriate from the list—never make up a new agent.
- AGENT NAMES ARE CASE-SENSITIVE AND MUST MATCH THE LIST EXACTLY.

Your job:
1. Generate an ordered list of tasks, as before.
2. Analyze the task dependencies and recommend the best execution mode for the pipeline:
   - "parallel": all tasks are independent and can be run in parallel.
   - "hierarchical": tasks have dependencies (use inputFromTask field to indicate dependency).
   - "sequential": single step or strictly linear.

Return only a valid JSON object with two fields:
- "executionMode": "parallel" | "hierarchical" | "sequential"
- "tasks": [ ... task array as described above ... ]

Example:
{
  "executionMode": "hierarchical",
  "tasks": [
    { "id": "plan", "name": "Content Plan", "description": "Plan content", "agent": "Planner" },
    { "id": "write", "name": "Write Article", "description": "Write article based on the plan", "agent": "Writer", "inputFromTask": "plan" }
  ]
}
`;

  // LLM call
  const raw = await callAIModel(
    "Task Planner",
    model,
    [
      { role: "system", content: "You are a workflow planner." },
      { role: "user", content: prompt },
    ],
    verbose
  );

  let cleaned = raw;
  if (typeof cleaned === "string" && cleaned.trim().startsWith("```json"))
    cleaned = cleaned.replace(/```json|```/g, "").trim();

  let parsed: any = {};
  try {
    parsed = JSON.parse(cleaned);
    if (
      !parsed ||
      typeof parsed !== "object" ||
      !parsed.tasks ||
      !Array.isArray(parsed.tasks) ||
      !parsed.executionMode
    ) {
      throw new Error("Missing executionMode or tasks");
    }
  } catch (err) {
    throw new Error(
      `[AI-Driven] Could not parse dynamic plan! Output:\n${raw}\n\nError: ${err}`
    );
  }

  // Build tasks
  const tasks = parsed.tasks.map(
    (t: any) =>
      new Task({
        id: t.id,
        name: t.name,
        description: t.description,
        agent: t.agent,
        outputFormat: t.outputFormat || "text",
        inputFromTask: t.inputFromTask,
      })
  );

  // Filter agents actually used in the plan
  const usedAgentNames = new Set(parsed.tasks.map((t: any) => t.agent));
  const agentsUsed = agents.filter((a) => usedAgentNames.has(a.name));

  return {
    tasks,
    agents: agentsUsed,
    executionMode: parsed.executionMode as string,
  };
}
