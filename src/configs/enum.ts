export enum MemoryScope {
  None = "none",
  Short = "short",
  Long = "long",
}

export enum MemoryMode {
  Same = "same",
  Seperated = "seperated",
}

export enum ExecutionMode {
  Sequential = "sequential",
  Hierarchical = "hierarchical",
  AiDriven = "ai-driven",
}

export enum VectorMemoryProviderType {
  //OpenAI = "openai",
  //Pinecone = "pinecone",
  Chroma = "chroma",
  //Weaviate = "weaviate",
  //Supabase = "supabase",
  Local = "local",
  LangChain = "langchain",
}

export enum OutputFormat {
  json = "json",
  text = "text",
  markdown = "markdown",
  csv = "csv",
  xml = "xml",
}

export enum RetrieverType {
  local = "local",
  langchain = "langchain",
  custom = "custom",
}

export enum SupportedModel {
  GPT_4O_MINI = "gpt-4o-mini",
  GPT_4O = "gpt-4o",
  GPT_3_5_TURBO = "gpt-3.5-turbo",
  DEEPSEEK_CHAT = "deepseek-chat",
  LOCAL_META_LLAMA = "local-meta-llama",
  LOCAL_HERMES_WRITER = "local-hermes-writer",
  CLAUDE_3_HAIKU = "claude-3-haiku",
  CLAUDE_3_SONNET = "claude-3-sonnet",
  CLAUDE_3_OPUS = "claude-3-opus",
  GEMINI_1_5_PRO = "gemini-1.5-pro",
  GEMINI_1_5_FLASH = "gemini-1.5-flash",
  MIXTRAL_8X7B = "mixtral-8x7b",
}

export enum SupportedModelProvider {
  OPENAI = "openai",
  DEEPSEEK = "deepseek",
  LOCAL = "local",
  ANTHROPIC = "anthropic",
  GEMINI = "gemini",
  MISTRAL = "mistral",
}

export enum FineTunableModel {
  GPT_3_5_TURBO = "gpt-3.5-turbo",
  // Eğer başka fine-tune destekli model eklenirse buraya ekle
  // Örn: GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106",
}

export type ChatCompletionTool = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: object;
  };
  [key: string]: any;
};

export type ChatCompletionMessageParam = {
  role: "system" | "user" | "assistant";
  content: string;
};
export type ChatCompletionSystemMessageParam = ChatCompletionMessageParam;
export type ChatCompletionUserMessageParam = ChatCompletionMessageParam;

export type ChatCompletionToolMessageParam = {
  role: "tool";
  tool_call_id: string;
  content: string;
};
