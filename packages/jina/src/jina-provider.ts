import { createProvider } from '@ai-sdk/provider/dist/provider';
import { JinaEmbeddingModel } from './jina-embedding-model';
import { JinaEmbeddingModelId } from './jina-embedding-options';

export interface JinaProvider {
  (modelId: JinaEmbeddingModelId, options?: {
    apiKey?: string;
    baseURL?: string;
    headers?: Record<string, string>;
  }): JinaEmbeddingModel;

  embedding(modelId: JinaEmbeddingModelId, options?: {
    apiKey?: string;
    baseURL?: string;
    headers?: Record<string, string>;
  }): JinaEmbeddingModel;
}

export const createJina = createProvider<JinaProvider>({
  provider: 'jina',
  baseURL: 'https://api.jina.ai/v1',
  headers: () => ({
    'Content-Type': 'application/json',
  }),
  generateModels: ({
    defaultObjectGenerationModel,
    defaultTextGenerationModel,
    create,
    config,
  }: { 
    defaultObjectGenerationModel: any;
    defaultTextGenerationModel: any;
    create: any;
    config: any;
  }) => ({
    embedding: create.model(
      (modelId: JinaEmbeddingModelId) =>
        new JinaEmbeddingModel(modelId, {
          provider: config.provider,
          baseURL: config.baseURL,
          headers: config.headers,
          fetch: config.fetch,
        }),
    ),
  }),
});
