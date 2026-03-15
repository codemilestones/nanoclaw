/**
 * NanoClaw Advanced Memory System - Embedding Service
 *
 * Multi-provider embedding service supporting:
 * - Jina AI (task-aware embeddings)
 * - OpenAI (text-embedding-3 models)
 * - SiliconFlow (Chinese models)
 * - Ollama (local models)
 */

import OpenAI from 'openai';

import type { EmbeddingProviderConfig } from './types.js';
import { expandEnv } from './config.js';
import { logger } from '../logger.js';

/**
 * Embedding request/response
 */
export interface EmbedRequest {
  texts: string[];
  /** Task type for task-aware embeddings (jina) */
  task?: 'query' | 'passage';
}

export interface EmbedResponse {
  embeddings: number[][];
  model: string;
  dimensions: number;
}

/**
 * Base embedding provider interface
 */
interface EmbeddingProvider {
  embed(request: EmbedRequest): Promise<EmbedResponse>;
  getDimensions(): number;
}

/**
 * Jina AI embedding provider
 * Supports task-aware embeddings for better retrieval
 */
class JinaEmbeddingProvider implements EmbeddingProvider {
  private client: OpenAI;
  private config: EmbeddingProviderConfig;

  constructor(config: EmbeddingProviderConfig) {
    this.config = config;
    this.client = new OpenAI({
      baseURL: config.baseURL || 'https://api.jina.ai/v1',
      apiKey: expandEnv(config.apiKey || ''),
    });
  }

  getDimensions(): number {
    return this.config.dimensions;
  }

  async embed(request: EmbedRequest): Promise<EmbedResponse> {
    const task = request.task || 'passage';
    const taskParam =
      task === 'query' ? this.config.taskQuery : this.config.taskPassage;

    try {
      const response = await this.client.embeddings.create({
        model: this.config.model,
        input: request.texts,
        encoding_format: 'float',
      } as any); // Jina API uses extra_body which isn't in OpenAI types

      const embeddings = response.data.map((d) => d.embedding);
      return {
        embeddings,
        model: this.config.model,
        dimensions: this.getDimensions(),
      };
    } catch (err) {
      logger.error({ error: err }, 'Jina embedding API error');
      throw new Error(
        `Jina embedding failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
}

/**
 * OpenAI embedding provider
 */
class OpenAIEmbeddingProvider implements EmbeddingProvider {
  private client: OpenAI;
  private config: EmbeddingProviderConfig;

  constructor(config: EmbeddingProviderConfig) {
    this.config = config;
    this.client = new OpenAI({
      baseURL: config.baseURL,
      apiKey: expandEnv(config.apiKey || ''),
    });
  }

  getDimensions(): number {
    return this.config.dimensions;
  }

  async embed(request: EmbedRequest): Promise<EmbedResponse> {
    try {
      const response = await this.client.embeddings.create({
        model: this.config.model,
        input: request.texts,
        dimensions:
          this.config.dimensions > 0 ? this.config.dimensions : undefined,
      });

      const embeddings = response.data.map((d) => d.embedding);
      return {
        embeddings,
        model: this.config.model,
        dimensions: this.getDimensions(),
      };
    } catch (err) {
      logger.error({ error: err }, 'OpenAI embedding API error');
      throw new Error(
        `OpenAI embedding failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
}

/**
 * SiliconFlow embedding provider
 * Supports Chinese models like BAAI/bge-large-zh-v1.5
 */
class SiliconFlowEmbeddingProvider implements EmbeddingProvider {
  private client: OpenAI;
  private config: EmbeddingProviderConfig;

  constructor(config: EmbeddingProviderConfig) {
    this.config = config;
    this.client = new OpenAI({
      baseURL: config.baseURL || 'https://api.siliconflow.cn/v1',
      apiKey: expandEnv(config.apiKey || ''),
    });
  }

  getDimensions(): number {
    return this.config.dimensions;
  }

  async embed(request: EmbedRequest): Promise<EmbedResponse> {
    try {
      const response = await this.client.embeddings.create({
        model: this.config.model,
        input: request.texts,
        encoding_format: 'float',
      });

      const embeddings = response.data.map((d) => d.embedding);
      return {
        embeddings,
        model: this.config.model,
        dimensions: this.getDimensions(),
      };
    } catch (err) {
      logger.error({ error: err }, 'SiliconFlow embedding API error');
      throw new Error(
        `SiliconFlow embedding failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
}

/**
 * Ollama embedding provider (local models)
 */
class OllamaEmbeddingProvider implements EmbeddingProvider {
  private baseURL: string;
  private config: EmbeddingProviderConfig;

  constructor(config: EmbeddingProviderConfig) {
    this.config = config;
    this.baseURL = config.baseURL || 'http://localhost:11434';
  }

  getDimensions(): number {
    return this.config.dimensions;
  }

  async embed(request: EmbedRequest): Promise<EmbedResponse> {
    const embeddings: number[][] = [];

    for (const text of request.texts) {
      try {
        const response = await fetch(`${this.baseURL}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: this.config.model,
            prompt: text,
          }),
        });

        if (!response.ok) {
          throw new Error(`Ollama API error: ${response.statusText}`);
        }

        const data = (await response.json()) as { embedding: number[] };
        embeddings.push(data.embedding);
      } catch (err) {
        logger.error({ error: err }, 'Ollama embedding error');
        throw new Error(
          `Ollama embedding failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    return {
      embeddings,
      model: this.config.model,
      dimensions: this.getDimensions(),
    };
  }
}

/**
 * Main embedding service class
 */
export class EmbeddingService {
  private provider: EmbeddingProvider;
  private config: EmbeddingProviderConfig;
  private cache: Map<string, number[]> = new Map();
  private cacheEnabled: boolean;

  constructor(config: EmbeddingProviderConfig, options?: { cache?: boolean }) {
    this.config = config;
    this.cacheEnabled = options?.cache !== false;

    // Create provider instance based on config
    switch (config.provider) {
      case 'jina':
        this.provider = new JinaEmbeddingProvider(config);
        break;
      case 'openai':
        this.provider = new OpenAIEmbeddingProvider(config);
        break;
      case 'siliconflow':
        this.provider = new SiliconFlowEmbeddingProvider(config);
        break;
      case 'ollama':
        this.provider = new OllamaEmbeddingProvider(config);
        break;
      default:
        // Fallback to jina for unknown providers
        this.provider = new JinaEmbeddingProvider(config);
    }

    logger.info(
      {
        provider: config.provider,
        model: config.model,
        dimensions: config.dimensions,
      },
      'Embedding service initialized',
    );
  }

  /**
   * Get embedding dimensions
   */
  getDimensions(): number {
    return this.provider.getDimensions();
  }

  /**
   * Generate cache key for text
   */
  private getCacheKey(text: string, task?: 'query' | 'passage'): string {
    return `${task || 'passage'}:${text}`;
  }

  /**
   * Embed a single text
   */
  async embed(text: string, task?: 'query' | 'passage'): Promise<number[]> {
    // Check cache
    if (this.cacheEnabled) {
      const key = this.getCacheKey(text, task);
      const cached = this.cache.get(key);
      if (cached) {
        return cached;
      }
    }

    const result = await this.embedBatch([text], task);
    const embedding = result.embeddings[0];

    // Update cache
    if (this.cacheEnabled) {
      const key = this.getCacheKey(text, task);
      this.cache.set(key, embedding);

      // Limit cache size
      if (this.cache.size > 10000) {
        const firstKey = this.cache.keys().next().value;
        if (firstKey) this.cache.delete(firstKey);
      }
    }

    return embedding;
  }

  /**
   * Embed multiple texts in batches
   */
  async embedBatch(
    texts: string[],
    task?: 'query' | 'passage',
  ): Promise<EmbedResponse> {
    if (texts.length === 0) {
      return {
        embeddings: [],
        model: this.config.model,
        dimensions: this.getDimensions(),
      };
    }

    const batchSize = this.config.batchSize || 100;
    const allEmbeddings: number[][] = [];

    // Process in batches
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);

      // Check cache for batch items
      const toEmbed: string[] = [];
      const indices: number[] = [];

      for (let j = 0; j < batch.length; j++) {
        if (this.cacheEnabled) {
          const key = this.getCacheKey(batch[j], task);
          const cached = this.cache.get(key);
          if (cached) {
            allEmbeddings[i + j] = cached;
          } else {
            toEmbed.push(batch[j]);
            indices.push(j);
          }
        } else {
          toEmbed.push(batch[j]);
          indices.push(j);
        }
      }

      if (toEmbed.length > 0) {
        try {
          const response = await this.provider.embed({ texts: toEmbed, task });

          for (let k = 0; k < response.embeddings.length; k++) {
            const idx = i + indices[k];
            allEmbeddings[idx] = response.embeddings[k];

            // Update cache
            if (this.cacheEnabled) {
              const key = this.getCacheKey(toEmbed[k], task);
              this.cache.set(key, response.embeddings[k]);
            }
          }
        } catch (err) {
          logger.error({ error: err, batchStart: i }, 'Embedding batch error');
          throw err;
        }
      }
    }

    // Fill any gaps from cache hits
    const result: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      if (allEmbeddings[i]) {
        result.push(allEmbeddings[i]);
      } else {
        // Shouldn't happen, but fallback to zero vector
        result.push(new Array(this.getDimensions()).fill(0));
      }
    }

    return {
      embeddings: result,
      model: this.config.model,
      dimensions: this.getDimensions(),
    };
  }

  /**
   * Clear the embedding cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; keys: number } {
    return {
      size: this.cache.size,
      keys: this.cache.size,
    };
  }
}

/**
 * Create embedding service from config
 */
export function createEmbeddingService(
  config: EmbeddingProviderConfig,
  options?: { cache?: boolean },
): EmbeddingService {
  return new EmbeddingService(config, options);
}
