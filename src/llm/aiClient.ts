import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { aiConfig } from "../configs/aiConfig.js";
import { toAIToolSchema } from "../tools/toolWorker/toolAdapter.js";
import { TFLog } from "../helpers/log.helper.js";
import chalk from "chalk";
import { Tool } from "../tools/base/baseTool.js";
import { ChatCompletionToolMessageParam } from "../configs/enum.js";
import { SupportedModel } from "../configs/enum.js";
import { recordLLMCall } from "../helpers/telemetry.helper.js";
import { modelGenerationDefaults } from "./modelsDefaultOptions.js";

/**
 * Options to control LLM generation behavior.
 * Not all options are supported by all models.
 */
export interface GenerationOptions {
  /**
   * Controls randomness of outputs. Lower = more deterministic (e.g., 0.2), higher = more creative (e.g., 0.8)
   *
   * @default 0.7
   * @supportedBy GPT-3.5, GPT-4, GPT-4o, Claude 3, Gemini 1.5 Pro/Flash, Mistral, Local LLaMA, DeepSeek
   */
  temperature?: number;

  /**
   * Controls nucleus sampling (top-p sampling). Value between 0 and 1. Common values: 0.9 or 0.95
   *
   * @default 0.95
   * @supportedBy GPT-3.5, GPT-4, GPT-4o, Claude 3, Gemini 1.5 Pro/Flash, Mistral, Local LLaMA, DeepSeek
   */
  top_p?: number;

  /**
   * Maximum number of tokens in the output.
   *
   * @default Varies by model
   * @supportedBy All models (with model-specific limits)
   */
  max_tokens?: number;

  /**
   * Penalizes repeated content. Positive values (0.1–1.0) encourage diversity.
   *
   * @default 0
   * @supportedBy GPT-3.5, GPT-4, GPT-4o, DeepSeek, Mistral, Local LLaMA
   * @notSupportedBy Claude 3, Gemini
   */
  presence_penalty?: number;

  /**
   * Penalizes tokens based on frequency. Positive values reduce repetition.
   *
   * @default 0
   * @supportedBy GPT-3.5, GPT-4, GPT-4o, DeepSeek, Mistral, Local LLaMA
   * @notSupportedBy Claude 3, Gemini
   */
  frequency_penalty?: number;
}

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

function getMergedGenerationOptions(
  modelName: SupportedModel | string,
  userOptions?: GenerationOptions
): GenerationOptions {
  const defaults = modelGenerationDefaults[modelName] || {};
  return {
    ...defaults,
    ...userOptions,
  };
}

/**
 * Wraps callAIModel with telemetry recording if TELEMETRY=true.
 */
export async function callAIModel(
  agentName: string,
  modelName: SupportedModel | string,
  messages: ChatMessage[],
  verbose: boolean = false,
  tools?: Tool[],
  modelOptions: GenerationOptions = {}
): Promise<string> {
  const mergedOptions = getMergedGenerationOptions(modelName, modelOptions);
  if (process.env.TELEMETRY_MODE === "none") {
    return callAIModelFunc(modelName, messages, verbose, tools, mergedOptions);
  }

  const startTime = Date.now();
  const result = await callAIModelFunc(
    modelName,
    messages,
    verbose,
    tools,
    mergedOptions
  );
  const duration = Date.now() - startTime;

  // Approximate token usage
  const totalTokens = Math.round(JSON.stringify(messages).length / 4);

  recordLLMCall(agentName, totalTokens, duration, modelName);
  return result;
}

async function callAIModelFunc(
  modelName: SupportedModel | string,
  messages: ChatMessage[],
  verbose: boolean = false,
  tools?: Tool[],
  modelOptions: GenerationOptions = {}
): Promise<string> {
  const config = aiConfig[modelName];
  if (!config) throw new Error(`Model '${modelName}' not defined in llmConfig`);

  switch (config.model.provider) {
    case "openai": {
      const openai = new OpenAI({ apiKey: config.apiKey });

      if (config.model.supportsTools && tools?.length) {
        const rawTools = toAIToolSchema(config.model, tools) || [];
        const openAITools = rawTools.map(({ __originalTool__, ...t }) => t);
        const toolMap = rawTools.reduce((acc, t: any) => {
          if (t.function?.name && t.__originalTool__) {
            acc[t.function.name] = t.__originalTool__;
          }
          return acc;
        }, {} as Record<string, Tool>);

        const res = await withRetry(
          () =>
            openai.chat.completions.create({
              model: config.model.name,
              messages,
              temperature: modelOptions.temperature || 0.7,
              top_p: modelOptions.top_p || 1,
              max_tokens: modelOptions.max_tokens || 2048,
              presence_penalty: modelOptions.presence_penalty || 0,
              frequency_penalty: modelOptions.frequency_penalty || 0,
              tools: openAITools,
              tool_choice: "auto",
            }),
          3,
          1000,
          verbose
        );

        const toolCalls = res.choices[0].message.tool_calls;
        let toolResult: ChatCompletionToolMessageParam[] = [];
        if (toolCalls && toolCalls?.length > 0 && toolMap) {
          if (verbose) {
            TFLog(
              `🧠 [LLM] Total ${toolCalls.length} tool call${
                toolCalls.length > 1 ? "s" : ""
              } received.`,
              chalk.yellow
            );
            const names = toolCalls.map((t) => t.function.name).join(", ");
            TFLog(`🧠 [LLM] Tool calls received: ${names}`, chalk.yellow);
          }

          for (const toolCall of toolCalls) {
            const toolName = toolCall.function.name;
            const args = JSON.parse(toolCall.function.arguments);

            const selectedTool = toolMap[toolName];
            if (!selectedTool) {
              throw new Error(`Tool not found: ${toolName}`);
            }

            if (verbose) {
              TFLog(
                `🧠 [LLM] Calling Tool: '${
                  toolCall.function.name
                }' ${JSON.stringify(args, null, 2)}`,
                chalk.yellow
              );
            }

            const result = await selectedTool.handler(args);
            if (verbose) {
              TFLog(
                `🧠 [LLM] Tool Result: '${toolCall.function.name}'`,
                chalk.yellow
              );
              TFLog(`Output:\n${result}\n`, chalk.white);
            }

            toolResult.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: result,
            });
          }

          const followUp = await withRetry(
            () =>
              openai.chat.completions.create({
                model: config.model.name,
                messages: [
                  ...messages,
                  {
                    role: "assistant",
                    content: null,
                    tool_calls: [...toolCalls],
                  },
                  ...toolResult,
                ],
                temperature: modelOptions.temperature || 0.7,
                top_p: modelOptions.top_p || 1,
                max_tokens: modelOptions.max_tokens || 2048,
                presence_penalty: modelOptions.presence_penalty || 0,
                frequency_penalty: modelOptions.frequency_penalty || 0,
              }),
            3,
            1000,
            verbose
          );

          return followUp.choices[0].message.content || "";
        }

        return res.choices[0].message.content || "";
      } else {
        const res = await withRetry(
          () =>
            openai.chat.completions.create({
              model: config.model.name,
              messages,
              temperature: modelOptions.temperature || 0.7,
              top_p: modelOptions.top_p || 1,
              max_tokens: modelOptions.max_tokens || 2048,
              presence_penalty: modelOptions.presence_penalty || 0,
              frequency_penalty: modelOptions.frequency_penalty || 0,
            }),
          3,
          1000,
          verbose
        );
        return res.choices[0].message.content || "";
      }
    }

    case "deepseek": {
      const res = await fetch(config.baseUrl!, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify({
          model: config.model.name,
          messages,
          temperature: modelOptions.temperature || 0.7,
          top_p: modelOptions.top_p || 1,
          max_tokens: modelOptions.max_tokens || 2048,
          presence_penalty: modelOptions.presence_penalty || 0,
          frequency_penalty: modelOptions.frequency_penalty || 0,
        }),
      });
      const json = await res.json();
      return json.choices?.[0]?.message?.content || "";
    }

    case "local": {
      const res = await fetch(config.baseUrl!, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: config.model.name,
          messages,
          temperature: modelOptions.temperature || 0.7,
          top_p: modelOptions.top_p || 1,
          max_tokens: modelOptions.max_tokens || 2048,
          presence_penalty: modelOptions.presence_penalty || 0,
          frequency_penalty: modelOptions.frequency_penalty || 0,
        }),
      });

      const json = await res.json();
      return json.choices?.[0]?.message?.content || "";
    }

    case "anthropic": {
      const res = await fetch(config.baseUrl!, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": config.apiKey!,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: config.model.name,
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          temperature: modelOptions.temperature || 0.7,
          top_p: modelOptions.top_p || 1,
          max_tokens: modelOptions.max_tokens || 2048,
          presence_penalty: modelOptions.presence_penalty || 0,
          frequency_penalty: modelOptions.frequency_penalty || 0,
        }),
      });

      const json = await res.json();
      return (
        json?.content?.[0]?.text || json?.choices?.[0]?.message?.content || ""
      );
    }

    case "gemini": {
      const genAI = new GoogleGenerativeAI(config.apiKey!);
      const model = genAI.getGenerativeModel({ model: config.model.name });

      function mergeSystemAndUserMessages(messages: ChatMessage[]) {
        let systemPrompt = "";

        const nonSystemMessages = messages.filter((msg) => {
          if (msg.role === "system") {
            systemPrompt += msg.content.trim() + "\n\n";
            return false;
          }
          return true;
        });

        const mergedMessages = nonSystemMessages.map((msg, index) => {
          const prefix = index === 0 && systemPrompt ? systemPrompt : "";
          return {
            role: msg.role,
            parts: [{ text: prefix + msg.content }],
          };
        });

        return mergedMessages;
      }

      const geminiMessages = mergeSystemAndUserMessages(messages);

      const generationConfig = {
        temperature: modelOptions.temperature ?? 0.7,
        topP: modelOptions.top_p ?? 1,
        maxOutputTokens: modelOptions.max_tokens ?? 2048,
      };

      // Eğer tools destekleniyorsa
      if (config.model.supportsTools && tools?.length) {
        const rawTools = toAIToolSchema(config.model, tools) || [];
        const functionDeclarations = rawTools.map(
          ({ __originalTool__, ...tool }) => {
            const fn = tool.function!;
            return {
              name: fn.name,
              description: fn.description,
              parameters: fn.parameters,
            };
          }
        );

        const toolMap = rawTools.reduce((acc, tool: any) => {
          if (tool.function?.name && tool.__originalTool__) {
            acc[tool.function.name] = tool.__originalTool__;
          }
          return acc;
        }, {} as Record<string, Tool>);

        if (verbose) {
          TFLog(
            `🛠️ [LLM] Passing ${functionDeclarations.length} tools`,
            chalk.yellow
          );
        }

        const result = await model.generateContent({
          contents: geminiMessages,
          generationConfig,
          tools: [{ functionDeclarations }],
        });

        const parts = result.response?.candidates?.[0]?.content?.parts || [];
        const functionCalls = parts
          .map((p: any) => p.functionCall)
          .filter((fc) => !!fc);
        const responseText = parts
          .map((p: any) => p.text)
          .filter(Boolean)
          .join("\n");

        if (functionCalls.length && toolMap) {
          const outputs: ChatCompletionToolMessageParam[] = [];

          for (const call of functionCalls) {
            const tool = toolMap[call.name];
            if (!tool) continue;

            const args = call.args;
            if (verbose) {
              TFLog(`🧠 [LLM] Calling Tool '${call.name}'`, chalk.yellow);
              TFLog(`Args: ${JSON.stringify(args, null, 2)}`, chalk.white);
            }

            const output = await tool.handler(args);

            if (verbose) {
              TFLog(`🧠 [LLM] Tool Result '${call.name}'`, chalk.yellow);
              TFLog(`Output: ${output}`, chalk.white);
            }

            outputs.push({
              role: "tool",
              tool_call_id: call.name,
              content: output,
            });
          }

          return outputs.map((o) => o.content).join("\n") || responseText;
        }

        return responseText;
      }

      const result = await model.generateContent({
        contents: geminiMessages,
        generationConfig,
      });

      return result.response?.candidates?.[0]?.content?.parts?.[0]?.text || "";
    }

    default:
      throw new Error(`Unsupported LLM provider: ${config.model.provider}`);
  }
}

async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delayMs = 1000,
  verbose = false
): Promise<T> {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      return await fn();
    } catch (err: any) {
      if (
        err?.status === 429 ||
        err?.message?.toLowerCase().includes("rate limit")
      ) {
        if (verbose) {
          console.warn(
            `⏳ OpenAI rate limit hit. Retrying in ${delayMs}ms... (attempt ${
              attempt + 1
            })`
          );
        }
        await new Promise((res) => setTimeout(res, delayMs));
        attempt++;
      } else {
        throw err;
      }
    }
  }
  throw new Error("❌ Retry failed after multiple OpenAI 429 errors.");
}
