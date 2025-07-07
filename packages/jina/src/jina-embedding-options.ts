import { z } from 'zod';

export type JinaEmbeddingModelId = 
  | 'jina-embeddings-v2-base-en'
  | 'jina-embeddings-v4';

// Multimodal input type for Jina embeddings
export type JinaEmbeddingInput = 
  | string
  | { text: string }
  | { image: string };

export const jinaEmbeddingProviderOptions = z.object({
  task: z.enum([
    'retrieval.query',
    'retrieval.passage', 
    'text-matching',
    'classification',
    'separation'
  ]).optional(),
  dimensions: z.number().optional(),
  normalized: z.boolean().optional(),
  late_chunking: z.boolean().optional(),
  embedding_type: z.enum(['float', 'base64', 'binary', 'ubinary']).optional(),
});

export type JinaEmbeddingProviderOptions = z.infer<
  typeof jinaEmbeddingProviderOptions
>;
