/**
 * NanoClaw Advanced Memory System - Hybrid Retrieval Engine
 *
 * Implements:
 * - Hybrid search: Vector + BM25 with RRF fusion
 * - Cross-encoder reranking (Jina/SiliconFlow)
 * - Multi-stage scoring (freshness, importance, time decay)
 * - MMR diversity for result variety
 */

import OpenAI from 'openai';

import type {
  Memory,
  MemorySearchQuery,
  MemorySearchResult,
  RetrievalConfig,
  RerankerConfig,
} from './types.js';
import { EmbeddingService } from './embedding.js';
import { MemoryStore } from './store.js';
import { expandEnv } from './config.js';
import { logger } from '../logger.js';

/**
 * RRF (Reciprocal Rank Fusion) constant
 */
const RRF_K = 60;

/**
 * Calculate RRF score from multiple rankings
 */
function calculateRRF(
  vectorRankings: Array<{ memory: Memory; rank: number }>,
  bm25Rankings: Array<{ memory: Memory; rank: number }>,
  weights: number[],
): Map<string, number> {
  const scores = new Map<string, number>();
  const allRankings = [vectorRankings, bm25Rankings];

  for (let i = 0; i < allRankings.length; i++) {
    const weight = weights[i] || 1;
    const rankings = allRankings[i];
    if (!rankings) continue;

    for (const item of rankings) {
      const rrfScore = weight / (RRF_K + item.rank);
      const currentScore = scores.get(item.memory.id) || 0;
      scores.set(item.memory.id, currentScore + rrfScore);
    }
  }

  return scores;
}

/**
 * Normalize scores to 0-1 range
 */
function normalizeScores(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const max = Math.max(...scores);
  const min = Math.min(...scores);
  const range = max - min;

  if (range === 0) return scores.map(() => 0.5);
  return scores.map((s) => (s - min) / range);
}

/**
 * Calculate time decay factor
 */
function calculateTimeDecay(timestamp: number, decayDays: number = 30): number {
  const ageMs = Date.now() - timestamp;
  const ageDays = ageMs / (1000 * 60 * 60 * 24);
  return Math.exp(-ageDays / decayDays);
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (normA * normB);
}

/**
 * Cross-encoder reranker using Jina or SiliconFlow APIs
 */
class CrossEncoderReranker {
  private client: OpenAI | null = null;
  private config: Required<RerankerConfig>;
  private enabled: boolean;

  constructor(config: RerankerConfig) {
    this.config = {
      provider: config.provider,
      model: config.model || 'jina-reranker-v2-base',
      apiKey: config.apiKey || '',
      topK: config.topK || 10,
      baseURL: config.baseURL || '',
    };
    this.enabled = config.provider !== 'none' && !!this.config.apiKey;

    if (this.enabled) {
      const baseURL =
        config.provider === 'siliconflow'
          ? 'https://api.siliconflow.cn/v1'
          : 'https://api.jina.ai/v1';

      this.client = new OpenAI({
        baseURL,
        apiKey: expandEnv(this.config.apiKey),
      });

      logger.info(
        { provider: config.provider, model: this.config.model },
        'Reranker initialized',
      );
    }
  }

  /**
   * Rerank results based on query relevance
   */
  async rerank(query: string, results: Memory[]): Promise<Map<string, number>> {
    if (!this.enabled || !this.client || results.length === 0) {
      return new Map();
    }

    try {
      const topK = Math.min(this.config.topK || 10, results.length);

      // Use fetch API for reranking
      const baseURL = this.config.baseURL || 'https://api.jina.ai/v1';
      const response = await fetch(`${baseURL}/rerank`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.config.apiKey}`,
        },
        body: JSON.stringify({
          model: this.config.model,
          query,
          documents: results.map((r) => r.text),
          top_n: topK,
        }),
      });

      if (!response.ok) {
        throw new Error(`Reranker API error: ${response.statusText}`);
      }

      const data = (await response.json()) as {
        results: Array<{ index: number; relevance_score: number }>;
      };

      const scores = new Map<string, number>();

      if (data.results && Array.isArray(data.results)) {
        for (const r of data.results) {
          const memory = results[r.index];
          scores.set(memory.id, r.relevance_score);
        }
      }

      logger.debug({ resultCount: scores.size }, 'Reranking completed');
      return scores;
    } catch (err) {
      logger.warn({ error: err }, 'Reranking failed, using original scores');
      return new Map();
    }
  }

  /**
   * Check if reranker is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}

/**
 * MMR (Maximal Marginal Relevance) for diversity
 */
class MMRReranker {
  /**
   * Apply MMR to diversify results
   * lambda: 0 = maximum diversity, 1 = no diversity
   */
  static diversify(
    results: Array<{
      memory: Memory;
      score: number;
      vectorScore?: number;
      bm25Score?: number;
    }>,
    queryEmbedding: number[],
    lambda: number = 0.5,
    limit?: number,
  ): Array<{
    memory: Memory;
    score: number;
    vectorScore?: number;
    bm25Score?: number;
  }> {
    if (results.length <= 1 || lambda >= 1) {
      return results.slice(0, limit);
    }

    const selected: Array<{
      memory: Memory;
      score: number;
      vectorScore?: number;
      bm25Score?: number;
    }> = [];
    const remaining = [...results];

    // Select first result (highest score)
    selected.push(remaining.shift()!);

    const maxIterations = limit || results.length;
    for (let i = 1; i < maxIterations && remaining.length > 0; i++) {
      let bestIdx = 0;
      let bestScore = -Infinity;

      for (let j = 0; j < remaining.length; j++) {
        const candidate = remaining[j];

        // Calculate similarity to already selected items
        let maxSimilarity = 0;
        for (const sel of selected) {
          if (sel.memory.vector && candidate.memory.vector) {
            const sim = cosineSimilarity(
              sel.memory.vector,
              candidate.memory.vector,
            );
            maxSimilarity = Math.max(maxSimilarity, sim);
          }
        }

        // MMR score: lambda * relevance - (1 - lambda) * max_similarity
        const mmrScore =
          lambda * candidate.score - (1 - lambda) * maxSimilarity;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIdx = j;
        }
      }

      selected.push(remaining.splice(bestIdx, 1)[0]);
    }

    return selected;
  }
}

/**
 * Main retrieval engine
 */
export class MemoryRetriever {
  private memoryStore: MemoryStore;
  private embedding: EmbeddingService;
  private config: RetrievalConfig;
  private reranker: CrossEncoderReranker;

  constructor(
    store: MemoryStore,
    embedding: EmbeddingService,
    config: RetrievalConfig,
  ) {
    this.memoryStore = store;
    this.embedding = embedding;
    this.config = config;
    this.reranker = new CrossEncoderReranker(config.rerank);
  }

  /**
   * Search memories with hybrid retrieval
   */
  async search(query: MemorySearchQuery): Promise<MemorySearchResult[]> {
    const {
      query: queryText,
      scope,
      scopeId,
      categories,
      limit = 5,
      minScore = 0,
      rerank = true,
      mmrLambda = 0.3,
      includeGlobal = true,
    } = query;

    // Collect results from different search methods
    const vectorRankings: Array<{ memory: Memory; rank: number }> = [];
    const bm25Rankings: Array<{ memory: Memory; rank: number }> = [];
    const memoryScores = new Map<
      string,
      {
        memory: Memory;
        vectorScore?: number;
        bm25Score?: number;
        timeDecay: number;
      }
    >();

    // Vector search
    if (this.config.mode === 'vector' || this.config.mode === 'hybrid') {
      try {
        const queryEmbed = await this.embedding.embed(queryText, 'query');
        const vectorResults = await this.memoryStore.vectorSearch(queryEmbed, {
          limit: this.config.candidatePoolSize,
          scope,
          scopeId,
          categories,
        });

        for (let i = 0; i < vectorResults.length; i++) {
          const { memory, score } = vectorResults[i];
          vectorRankings.push({ memory, rank: i });

          if (!memoryScores.has(memory.id)) {
            memoryScores.set(memory.id, {
              memory,
              vectorScore: score,
              timeDecay: calculateTimeDecay(memory.timestamp),
            });
          } else {
            const existing = memoryScores.get(memory.id)!;
            existing.vectorScore = score;
          }
        }
      } catch (err) {
        logger.warn({ error: err }, 'Vector search failed');
      }
    }

    // BM25/FTS search
    if (this.config.mode === 'bm25' || this.config.mode === 'hybrid') {
      try {
        const ftsResults = await this.memoryStore.ftsSearch(queryText, {
          limit: this.config.candidatePoolSize,
          scope,
          scopeId,
          categories,
        });

        for (let i = 0; i < ftsResults.length; i++) {
          const { memory, score } = ftsResults[i];
          bm25Rankings.push({ memory, rank: i });

          if (!memoryScores.has(memory.id)) {
            memoryScores.set(memory.id, {
              memory,
              bm25Score: score,
              timeDecay: calculateTimeDecay(memory.timestamp),
            });
          } else {
            const existing = memoryScores.get(memory.id)!;
            existing.bm25Score = score;
          }
        }
      } catch (err) {
        logger.warn({ error: err }, 'FTS search failed');
      }
    }

    // Apply RRF fusion for hybrid mode
    let rrfScores = new Map<string, number>();

    if (
      this.config.mode === 'hybrid' &&
      vectorRankings.length > 0 &&
      bm25Rankings.length > 0
    ) {
      rrfScores = calculateRRF(vectorRankings, bm25Rankings, [
        this.config.vectorWeight,
        this.config.bm25Weight,
      ]);
    } else if (this.config.mode === 'vector' && vectorRankings.length > 0) {
      const scores = vectorRankings.map((r) => (r.memory.vector ? 1 : 0));
      const normalized = normalizeScores(scores);
      for (let i = 0; i < vectorRankings.length; i++) {
        rrfScores.set(vectorRankings[i].memory.id, normalized[i]);
      }
    } else if (this.config.mode === 'bm25' && bm25Rankings.length > 0) {
      const scores = bm25Rankings.map(() => 1);
      const normalized = normalizeScores(scores);
      for (let i = 0; i < bm25Rankings.length; i++) {
        rrfScores.set(bm25Rankings[i].memory.id, normalized[i]);
      }
    }

    // Build combined results
    const combined: Array<{
      memory: Memory;
      score: number;
      vectorScore?: number;
      bm25Score?: number;
    }> = [];

    for (const [id, data] of memoryScores) {
      const baseScore =
        rrfScores.get(id) ?? data.vectorScore ?? data.bm25Score ?? 0;
      const importanceBoost = (data.memory.importance ?? 0.5) * 0.2;
      const timeDecay = data.timeDecay;
      const finalScore = baseScore * (1 - timeDecay * 0.1) + importanceBoost;

      combined.push({
        memory: data.memory,
        score: Math.max(0, Math.min(1, finalScore)),
        vectorScore: data.vectorScore,
        bm25Score: data.bm25Score,
      });
    }

    // Apply cross-encoder reranking
    let rerankedScores = new Map<string, number>();
    if (rerank && this.reranker.isEnabled()) {
      const memoriesToRerank = combined
        .sort((a, b) => b.score - a.score)
        .slice(0, this.config.rerank.topK || 10)
        .map((r) => r.memory);

      rerankedScores = await this.reranker.rerank(queryText, memoriesToRerank);
    }

    // Apply reranker scores
    if (rerankedScores.size > 0) {
      for (const item of combined) {
        const rerankScore = rerankedScores.get(item.memory.id);
        if (rerankScore !== undefined) {
          item.score = item.score * 0.3 + rerankScore * 0.7;
        }
      }
    }

    // Sort by final score
    combined.sort((a, b) => b.score - a.score);

    // Apply MMR for diversity
    const queryEmbed = await this.embedding.embed(queryText, 'query');
    const diversified =
      mmrLambda > 0 && mmrLambda < 1
        ? MMRReranker.diversify(combined, queryEmbed, mmrLambda)
        : combined;

    // Filter by min score
    let results = diversified.filter((r) => r.score >= minScore);

    // Include global memories if requested and scope is group
    if (includeGlobal && scope === 'group' && scopeId) {
      const globalLimit = Math.floor(limit / 2);
      const globalResults = await this.search({
        query: queryText,
        scope: 'global',
        limit: globalLimit,
        minScore: minScore * 0.8,
        rerank: false,
        includeGlobal: false,
      });

      const existingIds = new Set(results.map((r) => r.memory.id));
      for (const gr of globalResults) {
        if (!existingIds.has(gr.memory.id)) {
          results.push({
            memory: gr.memory,
            score: gr.score * 0.9, // Slightly lower score for global memories
            vectorScore: gr.vectorScore,
            bm25Score: gr.bm25Score,
          });
        }
      }

      results.sort((a, b) => b.score - a.score);
    }

    // Apply hard minimum score
    const hardMin = this.config.hardMinScore || this.config.minScore;
    results = results.filter((r) => r.score >= hardMin);

    // Apply limit
    results = results.slice(0, limit);

    // Update access count for retrieved memories
    for (const result of results) {
      await this.memoryStore
        .update(result.memory.id, {
          accessCount: result.memory.accessCount + 1,
          lastAccessed: Date.now(),
        })
        .catch(() => {
          // Ignore update errors
        });
    }

    return results.map((r) => ({
      memory: r.memory,
      score: r.score,
      vectorScore: r.vectorScore,
      bm25Score: r.bm25Score,
      rerankScore: rerankedScores.get(r.memory.id),
      timeDecay: calculateTimeDecay(r.memory.timestamp),
    }));
  }

  /**
   * Get more memories like a given memory
   */
  async findSimilar(
    memoryId: string,
    options: {
      limit?: number;
      scope?: string;
      minScore?: number;
    } = {},
  ): Promise<MemorySearchResult[]> {
    const memory = await this.memoryStore.get(memoryId);
    if (!memory || !memory.vector) {
      return [];
    }

    const similarMemories = await this.memoryStore.vectorSearch(memory.vector, {
      limit: options.limit || 5,
      scope: options.scope as any,
      minScore: options.minScore,
    });

    return similarMemories
      .filter((r) => r.memory.id !== memoryId)
      .map((r) => ({
        memory: r.memory,
        score: r.score,
      }));
  }
}

/**
 * Create a memory retriever instance
 */
export function createMemoryRetriever(
  store: MemoryStore,
  embedding: EmbeddingService,
  config: RetrievalConfig,
): MemoryRetriever {
  return new MemoryRetriever(store, embedding, config);
}
