import {
  callAIModel,
  ChatMessage,
  GenerationOptions,
} from "../llm/aiClient.js";
import { getLLMModelByName, LLMModel } from "../configs/aiConfig.js";
import { ToolExecutor } from "../tools/toolWorker/toolExecutor.js";
import {
  buildSystemMessage,
  buildUserMessage,
  delegateTo,
  recursiveToolWorker,
  truncateMessagesToFitModelLimit,
} from "./agent.helper.js";
import chalk from "chalk";
import { TFLog } from "../helpers/log.helper.js";
import { Tool } from "../tools/base/baseTool.js";
import { ToolRegistry } from "../tools/base/toolRegistry.js";
import { checkDelegate, checkTool } from "../helpers/helper.js";
import { VectorMemoryProvider } from "../memory/vectorMemoryProviders/vectorMemoryProvider.js";
import {
  ExecutionMode,
  MemoryMode,
  MemoryScope,
  SupportedModel,
  VectorMemoryProviderType,
} from "../configs/enum.js";
import { MemoryProvider } from "../memory/memoryFactory.js";
import { Retriever } from "../memory/retrievals/retrieval.interface.js";
import { retrieveAndEnrichPrompt } from "../memory/retrievals/retrieval.helper.js";
import { AgentTrainingResult } from "../agentTraining/agentTraining.types.js";
import { loadTraining } from "../agentTraining/agentTraining.helper.js";
import { TaskForce } from "../engine/taskForce.js";

export class Agent {
  public name: string;
  public role: string;
  public goal: string;
  public backstory: string;
  public model?: SupportedModel | string;
  /**
   * @param modelOptions Generation options for the underlying LLM (e.g., temperature, top_p, etc.)
   * @example
   * modelOptions: {
   *   temperature: 0.2,
   *   top_p: 0.95
   * }
   *
   * @note These settings may be ignored if the selected model does not support them. For best compatibility, use OpenAI models like gpt-4 or gpt-3.5.
   *
   * @see See full compatibility chart in docs/agents/README.md#generation-options-compatibility
   */
  public modelOptions?: GenerationOptions;
  public tools?: (Tool | (new () => Tool) | (() => Tool))[];
  public guardrails?: string[];
  public memoryScope?: MemoryScope;
  public memoryProvider?: VectorMemoryProvider;
  public systemPrompt?: string;
  public allowDelegation: boolean;
  public retriever?: Retriever;
  public readonly llmModel: LLMModel;
  private delegationCount: number = 0;
  private readonly MAX_DELEGATION = 3;
  private agentTools: Tool[];

  public toolExecutor: ToolExecutor;
  private verbose: boolean = false;

  private delegationDisallowed?: boolean;

  trained?: AgentTrainingResult;

  public autoTruncateHistory?: boolean;

  constructor(agent: {
    name: string;
    role: string;
    goal: string;
    backstory: string;
    model?: SupportedModel | string;
    modelOptions?: GenerationOptions;
    tools?: (Tool | (new () => Tool) | (() => Tool))[];
    guardrails?: string[];
    memoryScope?: MemoryScope;
    systemPrompt?: string;
    allowDelegation?: boolean;
    memoryProvider?: VectorMemoryProvider;
    retriever?: Retriever;
    trainingPath?: string;
    taskForce?: TaskForce;
    autoTruncateHistory?: boolean;
  }) {
    this.name = agent.name;
    this.role = agent.role;
    this.goal = agent.goal;
    this.backstory = agent.backstory;
    this.model = agent.model;
    this.modelOptions = agent.modelOptions;
    this.guardrails = agent.guardrails;
    this.systemPrompt = agent.systemPrompt;
    this.allowDelegation = agent.allowDelegation ?? false;
    this.retriever = agent.retriever;
    this.autoTruncateHistory = agent.autoTruncateHistory ?? false;

    this.llmModel = getLLMModelByName(
      this.model || process.env.DEFAULT_MODEL!
    )!;

    // 🆕 Tool sınıflarını instance’a çevir + ToolRegistry’ye kaydet
    const instantiatedTools = (agent.tools || []).map((t) => {
      if (typeof t === "function") {
        try {
          const instance = this.isClassConstructor(t) ? new t() : t(); // 👈 ayrım burada
          ToolRegistry.register(instance);
          return instance;
        } catch (err) {
          throw new Error(`[Agent] Tool initialization failed: ${err}`);
        }
      } else if (t instanceof Tool) {
        ToolRegistry.register(t);
        return t;
      } else {
        throw new Error(`[Agent] Invalid tool type: ${typeof t}`);
      }
    });

    this.agentTools = instantiatedTools;
    this.toolExecutor = new ToolExecutor(
      instantiatedTools,
      agent.taskForce,
      agent.name
    );

    this.memoryScope =
      agent.memoryScope ??
      (this.memoryProvider && this.memoryProvider.getMemoryScope()) ??
      MemoryScope.None;

    this.memoryProvider =
      agent.memoryProvider ??
      MemoryProvider(
        this.name,
        this.memoryScope,
        MemoryMode.Same,
        VectorMemoryProviderType.Local
      );

    if (this.memoryProvider && this.memoryScope != MemoryScope.None) {
      (this as any).memoryInitialized = true;
    }

    const data = loadTraining(agent.name);
    if (data) {
      this.trained = data;
      if (this.verbose) {
        TFLog(
          `📚 Loaded training data for ${agent.name}: ${JSON.stringify(data)}`,
          chalk.magenta
        );
      }
    }
  }

  isClassConstructor(func: any): func is new () => Tool {
    return (
      typeof func === "function" &&
      /^class\s/.test(Function.prototype.toString.call(func))
    );
  }

  setDelegationDisallowed(disallowed: boolean) {
    this.delegationDisallowed = disallowed;
  }

  getDelegationDisallowed() {
    return this.delegationDisallowed;
  }

  public setVerbose(verbose: boolean) {
    this.verbose = verbose;
  }

  public getVerbose(): boolean {
    return this.verbose;
  }

  canUseTools(): boolean {
    return this.toolExecutor.getTools().length > 0;
  }

  canDelegate(): boolean {
    return this.allowDelegation && this.delegationCount < this.MAX_DELEGATION;
  }

  incrementDelegation(): void {
    this.delegationCount++;
  }

  async runTask(
    inputs: Record<string, string>,
    taskDescription: string,
    outputFormat?: string,
    inputFromPrevious?: string,
    retryReason?: string,
    delegateReason?: string,
    rePlanReason?: string,
    executionMode?: ExecutionMode,
    retriever?: Retriever,
    isTraining: boolean = false
  ): Promise<string> {
    if (this.verbose) {
      TFLog(
        `🚀 [Agent Run Task] ${this.name} is executing task: ${taskDescription}`,
        chalk.yellow
      );
    }

    // ❌ MEMORY READ: Skip during training
    if (!isTraining && this.memoryScope !== MemoryScope.None) {
      let enrichedContext = "";
      const retrieverToUse = this.retriever || retriever;
      if (retrieverToUse || this.memoryProvider) {
        enrichedContext = await retrieveAndEnrichPrompt(
          inputFromPrevious || Object.values(inputs).join(" "),
          retrieverToUse,
          this.memoryProvider,
          this.model || process.env.DEFAULT_MODEL!,
          this.verbose,
          {
            agent: this.name,
            taskId: taskDescription,
          }
        );
      }

      if (enrichedContext) {
        inputFromPrevious = inputFromPrevious
          ? `${enrichedContext}\n\n${inputFromPrevious}`
          : enrichedContext;

        if (this.verbose) {
          const providerName =
            this.memoryProvider?.constructor?.name || "UnknownMemory";
          TFLog(
            `Injected Memory from from ${providerName} to ${this.name}:\n${inputFromPrevious}\n`,
            chalk.white
          );
        }
      }
    }

    const systemMessage = buildSystemMessage(inputs, this, isTraining);

    const userMessage = buildUserMessage(
      inputs,
      this,
      taskDescription,
      inputFromPrevious,
      retryReason,
      delegateReason,
      outputFormat,
      rePlanReason
    );

    let chatMessage: ChatMessage[] = [
      {
        role: "system",
        content: systemMessage,
      },
      {
        role: "user",
        content: userMessage,
      },
    ];

    let safeMessages = chatMessage;
    if (this.autoTruncateHistory) {
      safeMessages = await truncateMessagesToFitModelLimit(
        this.model || process.env.DEFAULT_MODEL!,
        chatMessage
      );
    }

    const output = await callAIModel(
      this.name,
      this.model || process.env.DEFAULT_MODEL!,
      safeMessages,
      this.verbose,
      this.llmModel.supportsTools && this.canUseTools() ? this.agentTools : [],
      this.modelOptions
    );

    if (this.verbose) {
      TFLog(`📤 [Agent Output] ${this.name} result:`, chalk.blue);
      TFLog(`${output}\n`, chalk.gray);
    }

    if (
      this.canUseTools() &&
      !this.llmModel.supportsTools &&
      checkTool(output)
    ) {
      const enrichedOutput = await recursiveToolWorker(
        output,
        this,
        taskDescription,
        outputFormat ?? "",
        inputs
      );

      if (enrichedOutput) {
        let safeMessages = enrichedOutput;
        if (this.autoTruncateHistory) {
          safeMessages = await truncateMessagesToFitModelLimit(
            this.model || process.env.DEFAULT_MODEL!,
            enrichedOutput
          );
        }

        const secondOutput = await callAIModel(
          this.name,
          this.model || process.env.DEFAULT_MODEL!,
          safeMessages,
          this.verbose,
          [],
          this.modelOptions
        );

        return secondOutput;
      }
    }

    if (this.getDelegationDisallowed()) {
      if (this.verbose) {
        TFLog(
          `⛔ [Delegation Blocked] ${this.name} is not allowed to delegate further.`,
          chalk.red
        );
      }

      if (checkDelegate(output)) {
        return output.replace(/^DELEGATE\(.*?\)$/, "⚠️ Delegation blocked.");
      }
    }

    if (this.canDelegate() && checkDelegate(output)) {
      const delegation = await delegateTo(this, output, inputs);
      if (delegation) {
        return JSON.stringify({ __delegate__: delegation, __depth__: 1 });
      }
    }

    // ❌ MEMORY WRITE: Skip during training
    if (!isTraining && executionMode === ExecutionMode.Sequential) {
      // 🧠 OUTPUT sonrası: Hafıza kaydı
      if (this.memoryScope !== MemoryScope.None && this.memoryProvider) {
        await this.memoryProvider.storeMemory({
          taskId: taskDescription,
          input: JSON.stringify(inputs),
          output,
          metadata: { agent: this.name },
        });
      }
    }

    return output;
  }
}
