import type {
  Configuration as ConfigurationT,
  OpenAIApi as OpenAIApiT,
  CreateEmbeddingRequest,
} from "openai";
import { backOff } from "exponential-backoff";
import type fetchAdapterT from "../util/axios-fetch-adapter.js";
import { chunkArray } from "../util/index.js";
import { Embeddings } from "./base.js";

interface ModelParams {
  modelName: string;
}

export class OpenAIEmbeddings extends Embeddings implements ModelParams {
  modelName = "text-embedding-ada-002";

  batchSize = 20;

  maxRetries = 6;

  private apiKey: string;

  private client: OpenAIApiT;

  constructor(
    fields?: Partial<ModelParams> & {
      verbose?: boolean;
      batchSize?: number;
      maxRetries?: number;
      openAIApiKey?: string;
    }
  ) {
    super();

    const apiKey = fields?.openAIApiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OpenAI API key not found");
    }

    this.modelName = fields?.modelName ?? this.modelName;
    this.batchSize = fields?.batchSize ?? this.batchSize;
    this.apiKey = apiKey;
    this.maxRetries = fields?.maxRetries ?? this.maxRetries;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const subPrompts = chunkArray(texts, this.batchSize);

    const embeddings = [];

    for (let i = 0; i < subPrompts.length; i += 1) {
      const input = subPrompts[i];
      const { data } = await this.embeddingWithRetry({
        model: this.modelName,
        input,
      });
      for (let j = 0; j < input.length; j += 1) {
        embeddings.push(data.data[j].embedding);
      }
    }

    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    const { data } = await this.embeddingWithRetry({
      model: this.modelName,
      input: text,
    });
    return data.data[0].embedding;
  }

  private async embeddingWithRetry(request: CreateEmbeddingRequest) {
    if (!this.client) {
      const { Configuration, OpenAIApi, fetchAdapter } =
        await OpenAIEmbeddings.imports();
      const clientConfig = new Configuration({
        apiKey: this.apiKey,
        baseOptions: { adapter: fetchAdapter },
      });
      this.client = new OpenAIApi(clientConfig);
    }
    const makeCompletionRequest = () => this.client.createEmbedding(request);
    return backOff(makeCompletionRequest, {
      startingDelay: 4,
      maxDelay: 10,
      numOfAttempts: this.maxRetries,
    });
  }

  static async imports(): Promise<{
    Configuration: typeof ConfigurationT;
    OpenAIApi: typeof OpenAIApiT;
    fetchAdapter: typeof fetchAdapterT;
  }> {
    try {
      const { Configuration, OpenAIApi } = await import("openai");
      const { default: fetchAdapter } = await import(
        "../util/axios-fetch-adapter.js"
      );
      return { Configuration, OpenAIApi, fetchAdapter };
    } catch (err) {
      console.error(err);
      throw new Error(
        "Please install openai as a dependency with, e.g. `npm install -S openai`"
      );
    }
  }
}
