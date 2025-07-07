import { z } from 'zod';

export type JinaEmbeddingModelId = 'jina-embeddings-v2-base-en';

export const jinaEmbeddingProviderOptions = z.object({});

export type JinaEmbeddingProviderOptions = z.infer<
  typeof jinaEmbeddingProviderOptions
>;
