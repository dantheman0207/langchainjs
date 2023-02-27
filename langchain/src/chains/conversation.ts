import { LLMChain } from "./llm_chain.js";
import { BaseLLM } from "../llms/index.js";
import { BasePromptTemplate, PromptTemplate } from "../prompts/index.js";

import { BaseMemory, BufferMemory, BufferMemoryInput, InputValues, OutputValues, MemoryVariables } from "../memory/index.js";

const defaultTemplate = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:`;

const defaultPrompt = new PromptTemplate({
  template: defaultTemplate,
  inputVariables: ["history", "input"],
});

export class ConversationChain extends LLMChain {
  constructor(fields: {
    llm: BaseLLM;
    prompt?: BasePromptTemplate;
    outputKey?: string;
    memory?: BaseMemory;
  }) {
    super({
      prompt: fields.prompt ?? defaultPrompt,
      llm: fields.llm,
      outputKey: fields.outputKey ?? "response",
    });
    this.memory = fields.memory ?? new BufferMemory();
  }
}

export class ConversationBufferMemory extends BaseMemory implements BufferMemoryInput {
  humanPrefix = "Human";

  aiPrefix = "AI";

  memoryKey = "history";

  outputKey;
  inputKey;
  buffer = "";

  constructor(fields?: Partial<BufferMemoryInput>) {
    super();
    this.humanPrefix = fields?.humanPrefix ?? this.humanPrefix;
    this.aiPrefix = fields?.aiPrefix ?? this.aiPrefix;
    this.memoryKey = fields?.memoryKey ?? this.memoryKey;
  }

  async memoryVariables(): Promise<Array<string>> {
    return [this.memoryKey];
  }

  async loadMemoryVariables(_values: InputValues): Promise<MemoryVariables> {
    const result = { [this.memoryKey]: this.buffer };
    return result;
  }

  async saveContext(
    inputValues: InputValues,
    outputValues: Promise<OutputValues>
  ): Promise<void> {
    let promptInputKey;
    if (!this.inputKey) {
      promptInputKey = await _getPromptInputKey(inputValues, await this.memoryVariables());
    } else {
      promptInputKey = this.inputKey;
    }
    let outputKey;
    if (!this.outputKey) {
      const outputs = await outputValues;
      if (Object.keys(outputs).length != 1) {
        throw new Error(`One output key expected, got ${Object.keys(outputs)}`);
      }
      outputKey = Object.keys(outputs)[0];
    } else {
      outputKey = this.outputKey;
    }
    const values = await outputValues;
    const human = `${this.humanPrefix}: ${inputValues[promptInputKey]}`;
    const ai = `${this.aiPrefix}: ${values[outputKey]}`;
    this.buffer += `\n${[human, ai].join("\n")}`;
  }
}

async function _getPromptInputKey(inputs: Record<string, any>, memory_variables: Array<string>): Promise<string> {
    // "stop" is a special key that can be passed as input but is not used to
    // format the prompt.
    const prompt_input_keys = Object.keys(inputs).filter(key => !memory_variables.includes(key) && key !== "stop");
    if (prompt_input_keys.length !== 1) {
      throw new Error(`One input key expected got ${prompt_input_keys}`);
    }
    return prompt_input_keys[0];
}