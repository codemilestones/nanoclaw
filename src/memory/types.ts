/**
 * NanoClaw Advanced Memory System - Type Definitions
 *
 * Core types for the memory system, including memory entries,
 * search queries, embedding providers, and configuration.
 */

/**
 * Memory scope determines visibility and isolation
 */
export type MemoryScope = 'global' | 'group' | 'agent' | 'custom';

/**
 * Memory category for classification and retrieval
 */
export type MemoryCategory =
  | 'preference' // User preferences, likes/dislikes
  | 'fact' // Factual information about the user
  | 'decision' // Decisions made or conclusions reached
  | 'entity' // Named entities (people, places, organizations)
  | 'context' // Contextual information for conversations
  | 'other'; // Uncategorized

/**
 * Core memory entry stored in LanceDB
 */
export interface Memory {
  /** Unique identifier (UUID) */
  id: string;
  /** The memory text content (also FTS indexed) */
  text: string;
  /** Embedding vector for semantic search */
  vector?: number[];
  /** Category classification */
  category: MemoryCategory;
  /** Scope for isolation */
  scope: MemoryScope;
  /** Group folder or agent ID for scoped memories */
  scopeId: string;
  /** Importance score 0-1 */
  importance: number;
  /** Creation timestamp (milliseconds) */
  timestamp: number;
  /** Access count for boosting */
  accessCount: number;
  /** Last access timestamp */
  lastAccessed: number;
  /** Extended metadata as JSON string */
  metadata?: string;
}

/**
 * Search query for memory retrieval
 */
export interface MemorySearchQuery {
  /** Search query text */
  query: string;
  /** Scope filter */
  scope?: MemoryScope;
  /** Scope ID filter (e.g., group folder) */
  scopeId?: string;
  /** Category filter */
  categories?: MemoryCategory[];
  /** Maximum results to return */
  limit?: number;
  /** Minimum relevance score (0-1) */
  minScore?: number;
  /** Whether to rerank results */
  rerank?: boolean;
  /** MMR diversity factor (0-1, higher = more diverse) */
  mmrLambda?: number;
  /** Whether to include global memories in group scope */
  includeGlobal?: boolean;
}

/**
 * Single memory search result with scoring
 */
export interface MemorySearchResult {
  /** The memory entry */
  memory: Memory;
  /** Combined relevance score (0-1) */
  score: number;
  /** Vector search score (0-1) */
  vectorScore?: number;
  /** BM25 keyword score (0-1) */
  bm25Score?: number;
  /** Reranker score (0-1) if reranking was applied */
  rerankScore?: number;
  /** Time decay factor applied */
  timeDecay?: number;
}

/**
 * Embedding provider configuration
 */
export interface EmbeddingProviderConfig {
  /** Provider name */
  provider: 'jina' | 'openai' | 'ollama' | 'siliconflow';
  /** API key (can use ${ENV_VAR} syntax) */
  apiKey?: string;
  /** Model name */
  model: string;
  /** Base URL for API */
  baseURL?: string;
  /** Embedding dimensions */
  dimensions: number;
  /** Task type for query embeddings */
  taskQuery?: string;
  /** Task type for passage embeddings */
  taskPassage?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Batch size for embeddings */
  batchSize?: number;
}

/**
 * Reranker configuration
 */
export interface RerankerConfig {
  /** Provider for reranking */
  provider: 'jina' | 'siliconflow' | 'none';
  /** Model name */
  model?: string;
  /** API key */
  apiKey?: string;
  /** Base URL */
  baseURL?: string;
  /** Top K results to rerank */
  topK?: number;
}

/**
 * Retrieval configuration
 */
export interface RetrievalConfig {
  /** Search mode: vector, bm25, or hybrid */
  mode: 'vector' | 'bm25' | 'hybrid';
  /** Weight for vector search in hybrid mode (0-1) */
  vectorWeight: number;
  /** Weight for BM25 in hybrid mode (0-1) */
  bm25Weight: number;
  /** Minimum score threshold */
  minScore: number;
  /** Hard minimum score (strict cutoff) */
  hardMinScore?: number;
  /** Candidate pool size for hybrid search */
  candidatePoolSize: number;
  /** Reranking configuration */
  rerank: RerankerConfig;
}

/**
 * Auto-capture configuration
 */
export interface AutoCaptureConfig {
  /** Enable automatic memory extraction */
  enabled: boolean;
  /** Categories to auto-capture */
  categories: MemoryCategory[];
  /** Minimum importance threshold */
  minImportance: number;
  /** Maximum memories per conversation */
  maxPerConversation: number;
}

/**
 * Auto-recall configuration
 */
export interface AutoRecallConfig {
  /** Enable automatic memory injection */
  enabled: boolean;
  /** Top K memories to inject */
  topK: number;
  /** Categories to recall */
  categories: MemoryCategory[];
}

/**
 * Scope configuration
 */
export interface ScopeConfig {
  /** Default scope for new memories */
  default: MemoryScope;
  /** Scope definitions */
  definitions: Record<
    string,
    {
      description: string;
      isolated?: boolean;
    }
  >;
}

/**
 * Main memory system configuration
 */
export interface MemoryConfig {
  /** Master enable/disable */
  enabled: boolean;
  /** LanceDB storage path */
  lancedbPath: string;
  /** Embedding configuration */
  embedding: EmbeddingProviderConfig;
  /** Retrieval configuration */
  retrieval: RetrievalConfig;
  /** Auto-capture configuration */
  autoCapture: AutoCaptureConfig;
  /** Auto-recall configuration */
  autoRecall: AutoRecallConfig;
  /** Scope configuration */
  scopes: ScopeConfig;
}

/**
 * Memory statistics
 */
export interface MemoryStats {
  /** Total memory count */
  totalMemories: number;
  /** Memories by category */
  byCategory: Record<MemoryCategory, number>;
  /** Memories by scope */
  byScope: Record<MemoryScope, number>;
  /** Oldest memory timestamp */
  oldestTimestamp?: number;
  /** Newest memory timestamp */
  newestTimestamp?: number;
  /** Average importance */
  avgImportance: number;
  /** Total access count */
  totalAccessCount: number;
}

/**
 * Memory store result
 */
export interface MemoryStoreResult {
  /** Success status */
  success: boolean;
  /** Stored memory ID */
  id?: string;
  /** Error message if failed */
  error?: string;
}

/**
 * Memory delete result
 */
export interface MemoryDeleteResult {
  /** Success status */
  success: boolean;
  /** Number of memories deleted */
  count: number;
  /** Error message if failed */
  error?: string;
}

/**
 * Extraction result from conversation
 */
export interface ExtractionResult {
  /** Extracted memories */
  memories: Array<{
    text: string;
    category: MemoryCategory;
    importance: number;
  }>;
  /** Confidence scores */
  confidence: number[];
}
