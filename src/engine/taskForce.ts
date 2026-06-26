import fs from "fs";
import chalk from "chalk";
import { Agent } from "../agents/agent.js";
import { Task } from "../tasks/task.js";
import { delegateTo, generateDynamicPlan } from "../agents/agent.helper.js";

import { SmartManagerAgent } from "../agents/smartManagerAgent.js";
import { AgentRegistry } from "../agents/agentRegistry.js";
import {
  initializeTaskforceLogFile,
  logTaskChaining,
  TFLog,
} from "../helpers/log.helper.js";
import {
  cleanFinalContext,
  getSafeReplacer,
  interpolateTemplate,
} from "../helpers/helper.js";
import { MemoryProvider } from "../memory/memoryFactory.js";
import {
  ExecutionMode,
  MemoryMode,
  MemoryScope,
  SupportedModel,
} from "../configs/enum.js";
import { Retriever } from "../memory/retrievals/retrieval.interface.js";
import {
  AgentTrainingResult,
  TrainingExample,
} from "../agentTraining/agentTraining.types.js";
import readline from "readline-sync";
import { OpenAI } from "openai";
import { EventEmitter } from "events";
import path from "path";
import dotenv from "dotenv";
import {
  checkDelegationScore,
  checkDelegationValidity,
  updateDelegationChain,
} from "../agents/delegation.guard.js";
import { generateMemorySummary } from "../helpers/summarize.helper.js";
dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const emitting = process.env.EMITTING;

interface TaskForceConfig {
  agents: Agent[];
  tasks: Task[];
  verbose?: boolean;
  memory?: boolean;
  executionMode?: ExecutionMode;
  allowParallelProcesses?: boolean;
  managerModel?: SupportedModel;
  maxRetryPerTask?: number;
  maxDelegatePerTask?: number;
  enableReplanning?: boolean;
  enableAIPlanning?: boolean;
  retriever?: Retriever;
}

export class TaskForce extends EventEmitter {
  private agents: Agent[];
  private tasks: Task[];
  private verbose: boolean;
  private memory: boolean;
  private executionMode?: ExecutionMode;
  private allowParallelProcesses?: boolean;
  private managerAgent?: SmartManagerAgent;
  private maxRetryPerTask: number;
  private maxDelegatePerTask: number;
  private globalReplanAttemptCount = 0;
  private readonly maxGlobalReplanLimit = 2;
  private enableReplanning?: boolean;
  private enableAIPlanning?: boolean;
  private retriever?: Retriever;

  constructor(config: TaskForceConfig) {
    super();

    this.agents = config.agents;
    this.tasks = config.tasks;
    this.verbose = config.verbose ?? false;
    this.memory = config.memory ?? false;
    this.executionMode = config.executionMode ?? ExecutionMode.Sequential;
    this.allowParallelProcesses = config.allowParallelProcesses ?? false;
    this.maxRetryPerTask = config.maxRetryPerTask ?? 5;
    this.maxDelegatePerTask = config.maxDelegatePerTask ?? 3;
    this.enableReplanning = config.enableReplanning ?? true;
    this.enableAIPlanning = config.enableAIPlanning ?? true;
    this.retriever = config.retriever;

    // Validate task dependencies
    const allTaskIds = new Set(this.tasks.map((t) => t.id));
    for (const task of this.tasks) {
      if (task.inputFromTask && !allTaskIds.has(task.inputFromTask)) {
        throw new Error(
          `[TaskForce] Task '${task.name}' references missing inputFromTask ID '${task.inputFromTask}'`
        );
      }
    }

    this.agents.forEach((agent) => {
      agent.setVerbose(this.verbose);
      AgentRegistry.register(agent);

      if (agent.toolExecutor?.setTaskForce) {
        agent.toolExecutor.setTaskForce(this);
      }
    });

    // Attach memory to agents if global memory is enabled
    if (this.memory) {
      for (const agent of this.agents) {
        if (!agent.memoryScope || agent.memoryScope === MemoryScope.None)
          continue;

        // Attach memory only if not externally set
        if (!agent.memoryProvider && !(agent as any).memoryInitialized) {
          if (agent.memoryScope === MemoryScope.Short) {
            agent.memoryProvider = MemoryProvider(
              agent.name,
              agent.memoryScope
            );
          } else if (agent.memoryScope === MemoryScope.Long) {
            const mode = (agent as any).memoryMode ?? MemoryMode.Same;
            agent.memoryProvider = MemoryProvider(
              agent.name,
              agent.memoryScope,
              mode
            );
          }
        }
      }
    }

    if (
      this.executionMode === ExecutionMode.Hierarchical ||
      this.executionMode === ExecutionMode.AiDriven
    ) {
      this.managerAgent = new SmartManagerAgent({
        name: "Smart Manager",
        role: "Autonomous Task Coordinator",
        goal: "Dynamically plan, assign, and evaluate tasks based on input context.",
        backstory:
          "You are responsible for delegating and validating the entire task pipeline.",
        model: config.managerModel || process.env.DEFAULT_MANAGER_MODEL!,
        allowDelegation: false,
        taskForce: this,
      });
      this.managerAgent.setVerbose(this.verbose);
    }
  }

  public emitStep(payload: Record<string, any>) {
    this.emit("step", {
      ...payload,
      timestamp: Date.now(),
    });
  }

  async run(inputs: Record<string, any>) {
    initializeTaskforceLogFile();
    if (this.verbose) {
      TFLog(
        `🚀 [Task Force] Running with ${this.agents
          .map((a) => a.name)
          .join(", ")} agents and ${this.tasks
          .map((t) => t.name)
          .join(", ")} tasks`,
        chalk.yellow
      );
    }

    let finalContext: any;
    let executedTaskIds: string[] = [];

    if (this.executionMode === ExecutionMode.Hierarchical) {
      if (!this.managerAgent) {
        throw new Error("Manager agent not set for hierarchical execution.");
      }

      if (this.tasks.length === 1 && this.enableAIPlanning) {
        const mainTask = this.tasks[0];
        const subtasks = await this.decomposeMainTask(mainTask);
        if (this.verbose) {
          TFLog(
            `[TaskForce] Decomposed main task '${mainTask.name}' into ${subtasks.length} subtasks.`,
            chalk.green
          );
        }
        this.tasks = subtasks;
      }

      const { taskPlan, context } = await this.createTaskPlan(inputs);

      executedTaskIds = taskPlan.map((t) => t.id);

      if (this.allowParallelProcesses) {
        finalContext = await this.runParallelHierarchicalFiltered(
          taskPlan,
          context
        );
      } else {
        finalContext = await this.runHierarchicalFiltered(taskPlan, context);
      }

      finalContext = await this.reviewFinalOutput(
        finalContext,
        this.allowParallelProcesses!
      );
    } else if (this.executionMode === ExecutionMode.AiDriven) {
      const plan = await generateDynamicPlan({
        inputs,
        agents: this.agents,
        model: this.managerAgent?.model ?? SupportedModel.GPT_4O_MINI,
        verbose: this.verbose,
      });
      this.tasks = plan.tasks;
      this.agents = plan.agents;

      switch (plan.executionMode) {
        case "parallel":
          finalContext = await this.runParallelHierarchicalFiltered(
            this.tasks,
            inputs
          );
          break;
        case "hierarchical":
          finalContext = await this.runHierarchicalFiltered(this.tasks, inputs);
          break;
        default:
          finalContext = await this.runSequential(inputs);
      }
    } else {
      executedTaskIds = this.tasks.map((t) => t.id);
      finalContext = await this.runSequential(inputs);
    }

    finalContext = cleanFinalContext(finalContext);

    return {
      result: finalContext,
      executedTaskIds,
    };
  }

  async reviewFinalOutput(
    finalContext: Record<string, any>,
    isParallel: boolean
  ) {
    if (!this.enableReplanning) {
      return finalContext;
    }

    for (const [taskId, output] of Object.entries(finalContext)) {
      if (typeof output === "string") {
        if (output.includes("please provide") || output.includes("DELEGATE(")) {
          if (this.verbose) {
            TFLog(
              `🔁 System Check rejected final output due to incomplete or delegated result in task '${taskId}'`,
              chalk.cyan
            );
          }
          finalContext.__replanReason__ = `System Check: Unresolved output in task '${taskId}'`;
          return this.replanRun(finalContext, isParallel);
        }
        try {
          const parsed = JSON.parse(output);
          if (parsed?.__delegate__) {
            if (this.verbose) {
              TFLog(
                `🔁 System Check found unresolved __delegate__ in task '${taskId}'`,
                chalk.cyan
              );
            }
            finalContext.__replanReason__ = `System Check: __delegate__ found in task '${taskId}'`;
            return this.replanRun(finalContext, isParallel);
          }
        } catch {
          // ignore
        }
      }
    }

    // General review by the Manager Agent
    const review = await this.managerAgent!.reviewFinalOutput(finalContext);

    if (review.action === "replan") {
      if (this.verbose) {
        TFLog(
          `🔁 [Manager Agent] rejected final output: ${review.reason}`,
          chalk.magenta
        );
      }
      finalContext.__replanReason__ = `Manager Agent: ${review.reason}`;
      return this.replanRun(finalContext, isParallel);
    }

    if (review.action === "accept" && this.verbose) {
      TFLog(`✅ [Manager Agent] Final output accepted.`, chalk.greenBright);
    }

    return finalContext;
  }

  private computeReplanTasks(
    failedTasks: string[],
    taskList: Task[]
  ): Set<string> {
    const replanTasks = new Set<string>(failedTasks);
    const taskMap = new Map(taskList.map((t) => [t.id, t]));

    let added = true;
    while (added) {
      added = false;
      for (const task of taskList) {
        if (
          task.inputFromTask &&
          replanTasks.has(task.inputFromTask) &&
          !replanTasks.has(task.id)
        ) {
          replanTasks.add(task.id);
          added = true;
        }
      }
    }

    return replanTasks;
  }

  private async replanRun(
    inputs: Record<string, any>,
    isParallel: boolean
  ): Promise<Record<string, any>> {
    if (
      (inputs.__replanCount__ || 0) >= this.maxGlobalReplanLimit ||
      this.globalReplanAttemptCount >= this.maxGlobalReplanLimit
    ) {
      TFLog(
        `🚫 Replan limit (${this.maxGlobalReplanLimit}) reached. Aborting.`,
        chalk.red
      );
      return inputs;
    }

    this.globalReplanAttemptCount++;
    inputs.__replanCount__ = (inputs.__replanCount__ || 0) + 1;

    if (this.verbose) {
      TFLog(
        `🔁 Re-running full task pipeline [attempt ${inputs.__replanCount__}] due to: ${inputs.__replanReason__}`,
        chalk.cyan
      );
    }

    const { taskPlan, context } = await this.createTaskPlan(inputs);

    const failedTasks = Object.entries(inputs)
      .filter(([_, output]) => {
        if (typeof output !== "string") return false;
        if (output.trim() === "") return true;
        if (output.includes("please provide") || output.includes("DELEGATE(")) {
          return true;
        }
        try {
          const parsed = JSON.parse(output);
          return typeof parsed === "object" && parsed?.__delegate__;
        } catch {
          return false;
        }
      })
      .map(([taskId]) => taskId);

    const replanSet = this.computeReplanTasks(failedTasks, taskPlan);

    const filteredTaskPlan = taskPlan.filter((t) => replanSet.has(t.id));
    const filteredContext = { ...context };

    // Save the outputs of successful tasks
    for (const [key, value] of Object.entries(inputs)) {
      if (!replanSet.has(key)) {
        filteredContext[key] = value;

        if (this.verbose) {
          TFLog(`🔁 [Replan] Keeping output for task '${key}'`, chalk.cyan);
        }
      }
    }

    return isParallel
      ? await this.runParallelHierarchicalFiltered(
          filteredTaskPlan,
          filteredContext
        )
      : await this.runHierarchicalFiltered(filteredTaskPlan, filteredContext);
  }

  async runSequential(
    inputs: Record<string, string>
  ): Promise<Record<string, any>> {
    let result = "";
    let taskIndex = 0;
    let results: Record<string, any> = {};

    while (taskIndex < this.tasks.length) {
      const task = this.tasks[taskIndex];
      const agent = this.agents.find((a) => a.name === task.agent);
      if (!agent) {
        taskIndex++;
        continue;
      }
      if (this.verbose) {
        TFLog(
          `📝 [Task] Starting task '${task.name}' assigned to '${agent.name}'`,
          chalk.yellow
        );
      }

      const output = await this.runTask(
        agent,
        task,
        inputs,
        task.description,
        task.outputFormat,
        result,
        undefined,
        undefined,
        ExecutionMode.Sequential,
        this.retriever
      );

      if (this.verbose) {
        TFLog(`✅ [Task] Completed '${task.name}'`, chalk.yellow);
        TFLog(`🧾 Output:\n${output}\n`, chalk.white);
      }

      const delegation = await delegateTo(agent, output, inputs);
      if (delegation) {
        const delegateAgent = this.agents.find(
          (a) => a.name === delegation.delegateTo
        );
        if (!delegateAgent)
          throw new Error(`❌ Agent '${delegation.delegateTo}' not found`);

        if (this.verbose) {
          TFLog(
            `🧭 ${agent.name} delegates to ${delegateAgent.name}`,
            chalk.red
          );
        }
        const delegateResult = await delegateAgent.runTask(
          inputs,
          delegation.task,
          undefined,
          output,
          undefined,
          undefined,
          undefined,
          undefined,
          this.retriever
        );
        result = delegateResult;

        // restart chain from the delegated agent
        taskIndex = this.tasks.findIndex((t) => t.agent === agent.name) + 1;
        continue;
      }
      results[task.id] = output;
      result = output;
      taskIndex++;
    }

    return results;
  }

  private async runHierarchicalFiltered(
    taskPlan: Task[],
    context: Record<string, any>
  ): Promise<Record<string, any>> {
    if (!this.managerAgent) {
      throw new Error("Hierarchical mode requires a managerAgent");
    }

    for (const task of taskPlan) {
      let assignedAgent: Agent | undefined;

      if (typeof task.agent === "string") {
        const agent = this.agents.find((a) => a.name === task.agent);
        if (!agent) throw new Error(`Agent '${task.agent}' not found`);
        assignedAgent = agent;
      } else if (task.agent) {
        assignedAgent = task.agent;
      } else {
        assignedAgent = await this.managerAgent!.assignAgent(task, this.agents);
        if (!assignedAgent) {
          throw new Error(`Failed to assign agent for task '${task.name}'`);
        }
      }

      if (this.verbose) {
        TFLog(
          `🧩 [Task] Running '${task.name}' with '${assignedAgent.name}'`,
          chalk.yellow
        );
      }

      let inputFromPreviousTask = this.getChainedInput(task, context);

      let finalOutput = await this.runTask(
        assignedAgent,
        task,
        context,
        task.description,
        task.outputFormat,
        inputFromPreviousTask,
        undefined,
        undefined,
        undefined,
        this.retriever
      );

      if (this.verbose) {
        TFLog(`✅ [Task] Completed '${task.name}'`, chalk.yellow);
        TFLog(`🧾 Output:\n${finalOutput}\n`, chalk.white);
      }

      finalOutput = await this.evaluateTaskLoop(
        task,
        assignedAgent,
        context,
        finalOutput
      );

      if (finalOutput.includes("DELEGATE(")) {
        if (this.verbose) {
          TFLog(
            `⚠️ [Task '${task.name}'] unresolved delegation output remains`,
            chalk.red
          );
        }
      }
      if (!finalOutput.includes("DELEGATE(")) {
        context[task.id] = finalOutput;
      }
    }

    return context;
  }

  private async runParallelHierarchicalFiltered(
    taskPlan: Task[],
    context: Record<string, any>
  ): Promise<Record<string, any>> {
    // taskId → list of resolve functions (multiple tasks can wait for same input)
    const contextReady = new Map<string, Array<() => void>>();

    const taskPromises = taskPlan.map((task) => {
      return new Promise<{ key: string; value: string }>(async (resolve) => {
        const taskKey = task.inputFromTask;

        if (taskKey && !context[taskKey]) {
          await new Promise<void>((waitForInput) => {
            if (!contextReady.has(taskKey)) {
              contextReady.set(taskKey, []);
            }
            contextReady.get(taskKey)!.push(waitForInput);
          });
        }

        const assignedAgent = this.agents.find((a) => a.name === task.agent);
        if (!assignedAgent) throw new Error(`Agent '${task.agent}' not found`);

        if (this.verbose) {
          TFLog(
            `⚡ [Parallel Task] Running '${task.name}' with '${assignedAgent.name}'`,
            chalk.cyan
          );
        }

        const input = this.getChainedInput(task, context);

        const output = await this.runTask(
          assignedAgent,
          task,
          context,
          task.description,
          task.outputFormat,
          input,
          undefined,
          undefined,
          undefined,
          this.retriever
        );

        const finalOutput = await this.evaluateTaskLoop(
          task,
          assignedAgent,
          context,
          output
        );

        if (finalOutput.includes("DELEGATE(")) {
          if (this.verbose) {
            TFLog(
              `⚠️ [Task '${task.name}'] unresolved delegation output remains`,
              chalk.red
            );
          }
        }
        if (!finalOutput.includes("DELEGATE(")) {
          context[task.id] = finalOutput;
        }

        if (contextReady.has(task.id)) {
          for (const resolveFn of contextReady.get(task.id)!) {
            resolveFn();
          }
          contextReady.delete(task.id);
        }

        resolve({ key: task.id, value: finalOutput });
      });
    });

    const results = await Promise.all(taskPromises);
    const finalContext: Record<string, any> = { ...context };
    for (const { key, value } of results) {
      finalContext[key] = value;
    }

    return finalContext;
  }

  private async handleDelegationAndToolOutput(
    task: Task,
    output: string,
    agent: Agent,
    context: Record<string, any>
  ): Promise<string> {
    // Handle DELEGATE(...)
    const delegateMatch = output.match(/^DELEGATE\((.+?),\s*"(.*)"\)$/);
    if (delegateMatch) {
      const delegateAgentName = delegateMatch[1].trim();
      const delegateTaskDesc = delegateMatch[2].trim();
      const delegateAgent = this.agents.find(
        (a) => a.name === delegateAgentName
      );
      if (!delegateAgent)
        throw new Error(`Agent '${delegateAgentName}' not found`);

      if (delegateAgent.name === agent.name) {
        if (this.verbose) {
          TFLog(`⚠️ Skipping self-delegation: ${agent.name}`, chalk.red);
        }
        return output;
      }

      if (this.verbose) {
        TFLog(
          `📤 DELEGATE() triggered in task '${task.name}': ${agent.name} → ${delegateAgent.name}`,
          chalk.magenta
        );
      }

      if (emitting === "true") {
        this.emitStep({
          agent: agent.name,
          task: task.name,
          action: "delegated",
          to: delegateAgent.name,
          reason: delegateTaskDesc,
        });
      }

      const delegateResult = await this.runTask(
        delegateAgent,
        task,
        context,
        delegateTaskDesc,
        task.outputFormat,
        output,
        undefined,
        undefined,
        undefined,
        this.retriever
      );

      // context[task.id] = delegateResult;
      logTaskChaining(agent.name, delegateAgent.name, delegateResult);
      return delegateResult;
    }

    // Handle TOOL(...)
    const toolMatch = output.match(/^TOOL\((.+?),\s*(\{.*\})\)$/);
    if (toolMatch) {
      const toolName = toolMatch[1].trim();
      const toolInputRaw = toolMatch[2];
      const toolExecutor = this.managerAgent?.toolExecutor;
      const tools = toolExecutor?.getTools();
      const tool = tools?.find((t) => t.name === toolName);
      if (!tool) throw new Error(`Tool '${toolName}' not found`);
      const toolInput = JSON.parse(toolInputRaw);

      if (this.verbose) {
        TFLog(
          `🔧 Running tool '${toolName}' for task '${task.name}' by agent '${agent.name}' with input ${toolInputRaw}`,
          chalk.cyan
        );
      }

      const toolResult = await tool.handler(toolInput);

      let summarized = toolResult;
      if (typeof toolResult === "string" && toolResult.length > 1000) {
        try {
          summarized = await generateMemorySummary(toolResult);
          if (this.verbose) {
            TFLog(
              "📘 Tool result summarized for memory efficiency.",
              chalk.gray
            );
          }
        } catch (err) {
          TFLog(`⚠️ Failed to summarize tool output: ${err}`, chalk.red);
          summarized = toolResult; // fallback
        }
      }

      const rerunResult = await this.runTask(
        agent,
        task,
        context,
        task.description,
        task.outputFormat,
        summarized,
        undefined,
        "Output was tool result",
        undefined,
        this.retriever
      );

      context[task.id] = rerunResult;
      return rerunResult;
    }

    return output;
  }

  private topologicalSortTasks(tasks: Task[]): Task[] {
    const sorted: Task[] = [];
    const visited = new Set<string>();
    const taskMap = new Map<string, Task>();

    for (const task of tasks) {
      taskMap.set(task.id, task);
    }

    function visit(task: Task) {
      if (visited.has(task.id)) return;
      if (task.inputFromTask && taskMap.has(task.inputFromTask)) {
        visit(taskMap.get(task.inputFromTask)!);
      }
      visited.add(task.id);
      sorted.push(task);
    }

    for (const task of tasks) {
      visit(task);
    }

    return sorted;
  }

  private async createTaskPlan(
    inputData: Record<string, any>
    //executeInParallel: boolean
  ): Promise<{ taskPlan: Task[]; context: Record<string, any> }> {
    const context: Record<string, any> = { ...inputData };
    let taskPlan = await this.managerAgent!.planTasks(this.tasks, context);

    taskPlan = this.topologicalSortTasks(taskPlan);

    // Reset internal replan flags
    for (const task of taskPlan) {
      if ("__replanReasonUsed" in task) {
        delete (task as any).__replanReasonUsed;
      }
    }

    return { taskPlan, context };
  }

  async decomposeMainTask(mainTask: Task): Promise<Task[]> {
    if (!this.managerAgent) {
      throw new Error("ManagerAgent not initialized");
    }
    const subtasksData = await this.managerAgent.decomposeTask(
      mainTask,
      this.agents,
      this.verbose
    );

    const subtasks = subtasksData.map((st) => {
      const taskData: Partial<Task> = {
        id: st.id,
        name: st.name,
        description: st.description,
        outputFormat: "text",
        agent: st.agent,
      };

      return new Task(taskData as Task);
    });

    return subtasks;
  }

  private getChainedInput(task: Task, context: Record<string, any>): string {
    const sourceTaskId = task.inputFromTask;
    if (!sourceTaskId) {
      return "";
    }
    const sourceTaskOutput = context[sourceTaskId];
    if (sourceTaskOutput === undefined) return "";

    const sourceTask = this.tasks.find((t) => t.id === sourceTaskId);
    logTaskChaining(
      sourceTask?.name || sourceTaskId,
      task.name,
      sourceTaskOutput
    );

    const rawInput = task.inputMapper
      ? task.inputMapper(sourceTaskOutput)
      : sourceTaskOutput;
    return typeof rawInput === "string" && rawInput.length > 3000
      ? rawInput.slice(0, 3000) + "\n\n[...truncated]"
      : rawInput;
  }

  private async evaluateTaskLoop(
    task: Task,
    agent: Agent,
    context: Record<string, any>,
    initialOutput: string
  ): Promise<string> {
    const guardCheck = checkDelegationValidity(agent, task);
    if (!guardCheck.canDelegate) {
      if (this.verbose) {
        TFLog(
          `🚫 Delegation blocked for '${task.name}': ${guardCheck.reason}`,
          chalk.red
        );
      }
      return `⚠️ Delegation loop or hop limit reached. Accepting output as-is.\n\n${initialOutput}`;
    }

    updateDelegationChain(task, agent);

    let finalOutput = initialOutput;
    let needsEvaluation = true;
    let retryCount = 0;
    let retryReasons: string[] = [];

    while (needsEvaluation) {
      const review = await this.managerAgent!.evaluateTaskOutput(
        task,
        finalOutput,
        this.agents
      );

      // 🚩 Retry loop detected (same agent)
      if (review.action === "retry" && review.retryWith?.name === task.agent) {
        retryCount++;
        if (retryCount >= this.maxRetryPerTask) {
          if (this.verbose) {
            TFLog(
              `🚫 Max retry count (${this.maxRetryPerTask}) reached for task '${task.name}' due to self-retry. Proceeding with last output.`,
              chalk.red
            );
          }
          needsEvaluation = false;
          break;
        }
        if (this.verbose) {
          TFLog(
            `⚠️ [Retry Loop] '${task.name}' retrying on same agent '${agent.name}'. Count: ${retryCount}`,
            chalk.red
          );
        }

        if ((review as any).reason) {
          retryReasons.push((review as any).reason);
        }

        finalOutput = await this.runTask(
          agent,
          task,
          context,
          task.description,
          task.outputFormat,
          finalOutput,
          (review as any).reason,
          undefined,
          undefined,
          this.retriever
        );
        continue;
      }

      if (
        review.action === "delegate" &&
        (review.delegateTo.name === task.agent ||
          finalOutput ===
            `DELEGATE(${review.delegateTo.name}, "${task.description}")`)
      ) {
        if (this.verbose) {
          TFLog(
            `⚠️ [Delegation Loop Detected] '${task.name}' tried to delegate back to itself. Forcing accept.`,
            chalk.red
          );
        }
        needsEvaluation = false;
        break;
      }

      if (this.verbose) {
        TFLog(
          `[Manager Agent] Evaluation decision for task '${
            task.name
          }' by agent '${agent.name}': ${JSON.stringify(
            review,
            getSafeReplacer()
          )}`,
          chalk.blue
        );
      }

      if (review.action === "accept") {
        const delegationScore = checkDelegationScore(finalOutput);
        if (delegationScore.isWeak) {
          if (this.verbose) {
            TFLog(
              `📊 [DelegationScore Metrics] Task '${task.name}' by '${agent.name}' scored ${delegationScore.score}/10 — Reason: ${delegationScore.reason}`,
              chalk.red
            );
          }

          if (emitting === "true") {
            this.emitStep({
              agent: agent.name,
              task: task.name,
              action: "delegationScore",
              score: delegationScore.score,
              reason: delegationScore.reason,
            });
          }

          retryCount++;
          if (retryCount >= this.maxRetryPerTask) {
            TFLog(
              `🚫 Max retry count reached despite weak delegation score.`,
              chalk.red
            );
            break;
          }

          finalOutput = await this.runTask(
            agent,
            task,
            context,
            task.description,
            task.outputFormat,
            finalOutput,
            delegationScore.reason,
            undefined,
            undefined,
            this.retriever
          );
          continue;
        }

        if (agent.memoryScope !== MemoryScope.None && agent.memoryProvider) {
          await agent.memoryProvider.storeMemory({
            taskId: task.description,
            input: JSON.stringify(context),
            output: finalOutput,
            metadata: { agent: agent.name },
          });

          if (this.verbose) {
            TFLog(
              `🧠 [MemoryStore] Saved memory for ${task.name} by ${agent.name}`,
              chalk.gray
            );
          }
        }

        if (this.verbose && retryReasons.length > 0) {
          TFLog(
            `🔁 [Retry Trace] Task '${
              task.name
            }' accepted after ${retryCount} retries. Reasons: ${retryReasons.join(
              " | "
            )}`,
            chalk.gray
          );
        }

        // delegasyon & tool handling
        let delegateDepth = 0;
        let delegated = true;
        while (delegated && delegateDepth < this.maxDelegatePerTask) {
          const processed = await this.handleDelegationAndToolOutput(
            task,
            finalOutput,
            agent,
            context
          );
          if (processed === finalOutput || processed.includes("DELEGATE(")) {
            if (this.verbose && processed.includes("DELEGATE(")) {
              TFLog(
                `⚠️ Unresolved DELEGATE() remains after delegation depth check`,
                chalk.red
              );
            }
            delegated = false;
          } else {
            finalOutput = processed;
            delegateDepth++;
          }
        }
        if (delegateDepth >= this.maxDelegatePerTask) {
          if (this.verbose) {
            TFLog(
              `🚫 Max delegation depth for '${task.name}'. Forcing execution by '${agent.name}'.`,
              chalk.red
            );
          }
          agent.setDelegationDisallowed(true);
          finalOutput = await this.runTask(
            agent,
            task,
            context,
            task.description,
            task.outputFormat,
            finalOutput,
            undefined,
            "Delegation blocked. Forced to execute.",
            undefined,
            this.retriever
          );
        }
        needsEvaluation = false;
        if (emitting === "true") {
          this.emitStep({
            agent: agent.name,
            task: task.name,
            action: "completed",
            output: finalOutput,
          });
        }
      } else if (review.action === "retry") {
        retryCount++;
        if (retryCount >= this.maxRetryPerTask) {
          if (this.verbose) {
            TFLog(
              `🚫 Max retry count (${this.maxRetryPerTask}) reached for task '${task.name}'. Proceeding with last output.`,
              chalk.red
            );
          }
          needsEvaluation = false;
          break;
        }
        const retryAgent = review.retryWith
          ? review.retryWith
          : await this.managerAgent!.assignAgent(task, this.agents);
        finalOutput = await this.runTask(
          retryAgent,
          task,
          context,
          task.description,
          task.outputFormat,
          finalOutput,
          (review as any).reason,
          undefined,
          undefined,
          this.retriever
        );
      } else if (review.action === "delegate") {
        finalOutput = await this.runTask(
          review.delegateTo,
          task,
          context,
          task.description,
          task.outputFormat,
          finalOutput,
          undefined,
          (review as any).reason,
          undefined,
          this.retriever
        );

        if (finalOutput.includes("DELEGATE(")) {
          if (this.verbose) {
            TFLog(
              `🚫 [DELEGATION Validation] Output still contains unresolved DELEGATE(...) for task '${task.name}'`,
              chalk.red
            );
          }

          // If can retry
          retryCount++;
          if (retryCount < this.maxRetryPerTask) {
            finalOutput = await this.runTask(
              agent,
              task,
              context,
              task.description,
              task.outputFormat,
              finalOutput,
              "Unresolved DELEGATE after handling attempt",
              undefined,
              undefined,
              this.retriever
            );
            continue; // evaluateTaskLoop goto top
          }

          // If can't retry, fallback to original output
          finalOutput = `⚠️ Unresolved DELEGATE() in final output. Original content:\n${finalOutput}`;
        }
      }
    }

    return finalOutput;
  }

  private async runTask(
    agent: Agent,
    task: Task,
    inputs: Record<string, any>,
    taskDescription: string,
    outputFormat?: string,
    inputFromPrevious?: string,
    retryReason?: string,
    delegateReason?: string,
    executionMode?: ExecutionMode,
    retriever?: Retriever
  ) {
    if (emitting === "true") {
      this.emitStep({
        agent: agent.name,
        task: task.name,
        action: "start",
      });
    }

    let replanReason =
      inputs.__replanReason__ && !task.__replanReasonUsed
        ? inputs.__replanReason__
        : undefined;

    const resolvedDescription = interpolateTemplate(taskDescription, inputs);

    if (this.verbose && agent.trained) {
      TFLog(
        `📘 [Training Enriched] Agent "${agent.name}" used training suggestions during this run.`,
        chalk.cyan
      );
    }

    const output = await agent.runTask(
      inputs,
      resolvedDescription,
      outputFormat,
      inputFromPrevious,
      retryReason,
      delegateReason,
      replanReason,
      executionMode,
      retriever
    );

    if (replanReason) {
      task.__replanReasonUsed = true;
    }

    if (replanReason && this.verbose) {
      TFLog(`[Manager Agent]🔁 Replan Reason: ${replanReason}`, chalk.magenta);
    }

    return output;
  }

  public async train(
    iterations: number,
    globalInputs: Record<string, any> = {}
  ) {
    for (const agent of this.agents) {
      const task = this.tasks.find((t) => t.agent === agent.name);
      if (!task) {
        console.warn(
          `⚠️ No task found for agent ${agent.name}. Skipping training.`
        );
        continue;
      }

      const trainingsDir = path.join(process.cwd(), "trainings");
      fs.mkdirSync(trainingsDir, { recursive: true });

      const outPath = path.join(
        trainingsDir,
        `${agent.name.toLowerCase().replace(/\s/g, "_")}_trained.json`
      );
      let previousData: AgentTrainingResult = {
        suggestions: [],
        quality: 0,
        final_summary: "",
      };

      if (fs.existsSync(outPath)) {
        const raw = fs.readFileSync(outPath, "utf-8");
        previousData = JSON.parse(raw);
      }

      const examples: TrainingExample[] = [];
      for (let i = 0; i < iterations; i++) {
        console.log(`\n🔁 [${agent.name}] Iteration ${i + 1}/${iterations}`);
        const initialOutput = await agent.runTask(
          globalInputs,
          task.description,
          task.outputFormat,
          "",
          undefined,
          undefined,
          undefined,
          ExecutionMode.Sequential,
          this.retriever,
          true
        );

        console.log(`\n📤 Initial Output:\n${initialOutput}\n`);

        const feedback = readline
          .question(`[${agent.name}] ✍️ Your feedback: `)
          .trim();

        if (!feedback) {
          console.log("⚠️ No feedback provided. Skipping iteration.");
          continue;
        }

        const improvedOutput = await agent.runTask(
          globalInputs,
          task.description,
          task.outputFormat,
          initialOutput + `\n\n[Feedback from user]: ${feedback}`,
          undefined,
          undefined,
          undefined,
          ExecutionMode.Sequential,
          this.retriever,
          true
        );

        console.log(`\n📥 Improved Output:\n${improvedOutput}\n`);

        examples.push({
          initialOutput,
          humanFeedback: feedback,
          improvedOutput,
        });
      }

      const newSuggestions = examples.flatMap((e) =>
        e.humanFeedback ? [`${e.humanFeedback}`] : []
      );

      const mergedSuggestions = [
        ...(previousData.suggestions || []),
        ...newSuggestions,
      ];

      const llmSummaryPrompt = `You are a training compression engine. You are given multiple human feedback comments that have been used to improve an AI agent's outputs.

      Your job is to extract the **underlying improvement instructions** from the feedback list and compress them into a clear and concrete list of what the agent should do better next time.
      
      Respond in the following JSON format:
      {
        "final_summary": "Do this, avoid that, include this... style directives that can be reused in future tasks.",
        "quality": 0-10
      }
      
      Only include distilled guidance in the final_summary. Do NOT write commentary or explanations.
      Strip redundant prefixes like 'Improve based on:' or 'You should...'
      The goal is to turn the list into reusable prompt directives.
      Use simple English.
      Keep the tone neutral and instructive.
      
      Here is the feedback list:
      ${mergedSuggestions
        .map((s) => "- " + s.replace(/^Improve based on:\s*/i, ""))
        .join("\n")}
      `;

      const completion = await openai.chat.completions.create({
        model: SupportedModel.GPT_4O_MINI,
        messages: [{ role: "user", content: llmSummaryPrompt }],
        temperature: 0.7,
      });

      let final_summary = "N/A";
      let quality = 8;

      try {
        const parsed = JSON.parse(
          completion.choices[0].message.content || "{}"
        );
        final_summary = parsed.final_summary || "No summary returned.";
        quality = parsed.quality || 8;
      } catch {
        final_summary = "Failed to parse LLM summary.";
      }

      const result: AgentTrainingResult = {
        suggestions: mergedSuggestions,
        final_summary,
        quality,
      };

      fs.writeFileSync(outPath, JSON.stringify(result, null, 2), "utf-8");

      console.log(
        `\n✅ Training complete for ${agent.name}. Saved to ${outPath}`
      );
    }
  }

  public getTaskById(id: string): Task | undefined {
    return this.tasks.find((t) => t.id === id);
  }

  public listTaskNames(): string[] {
    return this.tasks.map((t) => t.name);
  }

  public getAgentByName(name: string): Agent | undefined {
    return this.agents.find((a) => a.name === name);
  }

  public listAgentNames(): string[] {
    return this.agents.map((a) => a.name);
  }
}
