export const baseModelTokenLimits: Record<string, number> = {
  "gpt-3.5-turbo": 16384,
  "gpt-3.5-turbo-0125": 16384,
  "gpt-3.5-turbo-instruct": 4096,
  "gpt-4": 8192,
  "gpt-4-32k": 32768,
  "gpt-4o": 64000,
  "gpt-4o-mini": 16384,
  "meta-llama-3.1-8b-instruct": 8192,
  "nous-hermes-2-mistral-7b-dpo": 8192,
  "deepseek-chat": 16384,
};

export function normalizeOutput(text: string): string {
  return text
    .replace(/\s{2,}/g, " ")
    .replace(/\n{2,}/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .trim();
}

export function checkDelegate(output: string): boolean {
  return output.match(/DELEGATE\(([^,]+),\s*"([^"]+)"\)/) !== null;
}

export function checkTool(output: string): boolean {
  return output.match(/TOOL\(([^,]+),\s*(\{.*?\})\)/g) !== null;
}

export function interpolateTemplate(
  template: string,
  context: Record<string, any>
): string {
  return template.replace(
    /\{(.*?)\}/g,
    (_, key) => context[key.trim()] ?? `{${key}}`
  );
}

export function normalizeInput(input: string): string {
  try {
    const parsed = JSON.parse(input);
    if (typeof parsed === "object") {
      return Object.entries(parsed)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ");
    }
  } catch {
    return input
      .replace(/\{.*?\}/g, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }
  return input;
}

export function cleanMarkdownJson(raw: string): string {
  let cleaned = raw.trim();

  cleaned = cleaned.replace(/^```json\s*\n?/, "");
  cleaned = cleaned.replace(/^```\s*\n?/, "");
  cleaned = cleaned.replace(/\n*```$/, "");
  cleaned = cleaned.replace(/```json/g, "");
  cleaned = cleaned.replace(/```/g, "");

  cleaned = cleaned.trim();

  return cleaned;
}

export function parseCSV(output: string): any[] {
  const lines = output.trim().split("\n");
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => {
      obj[h.trim()] = values[i]?.trim() ?? "";
    });
    return obj;
  });
}

export function parseXML(output: string): any {
  try {
    const parser = new DOMParser();
    const xmlDoc = parser.parseFromString(output, "application/xml");
    return xmlDoc;
  } catch {
    return output;
  }
}

export function cleanFinalContext(finalContext: any): any {
  if (typeof finalContext !== "object" || finalContext === null)
    return finalContext;

  const cleaned = { ...finalContext };
  delete cleaned.__replanReason__;
  delete cleaned.__replanCount__;
  return cleaned;
}

export function getSafeReplacer() {
  const seen = new WeakSet();
  return (_key: string, value: any) => {
    if (typeof value === "object" && value !== null) {
      if (seen.has(value)) return "[Circular]";
      seen.add(value);
    }
    return value;
  };
}
