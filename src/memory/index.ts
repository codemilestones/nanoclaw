/**
 * NanoClaw Advanced Memory System - Main Entry Point
 *
 * Exports all public APIs and provides initialization.
 */

import type {
  AutoCaptureConfig,
  AutoRecallConfig,
  EmbeddingProviderConfig,
  Memory,
  MemoryCategory,
  MemoryConfig,
  MemoryDeleteResult,
  MemoryScope,
  MemorySearchQuery,
  MemorySearchResult,
  MemoryStats,
  MemoryStoreResult,
  RetrievalConfig,
  RerankerConfig,
  ScopeConfig,
} from './types.js';
import {
  loadMemoryConfig,
  saveMemoryConfig,
  validateConfig,
  expandEnv,
} from './config.js';
import { createEmbeddingService, EmbeddingService } from './embedding.js';
import { createMemoryStore, MemoryStore } from './store.js';
import { createMemoryRetriever, MemoryRetriever } from './retriever.js';
import {
  applyQualityFilters,
  isWorthStoring,
  categorizeText,
  shouldSkipRetrieval,
  adaptiveRetrievalParams,
  extractEntities,
} from './filters.js';
import { logger } from '../logger.js';
import fs from 'fs';
import path from 'path';

/**
 * Main memory system class
 */
export class MemorySystem {
  private config: MemoryConfig;
  private memoryStore: MemoryStore | null = null;
  private embedding: EmbeddingService | null = null;
  private retriever: MemoryRetriever | null = null;
  private initialized: boolean = false;
  private projectRoot: string;

  constructor(config: MemoryConfig, projectRoot: string) {
    this.config = config;
    this.projectRoot = projectRoot;
  }

  /**
   * Initialize the memory system
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    if (!this.config.enabled) {
      logger.info('Memory system is disabled');
      return false;
    }

    try {
      // Validate configuration
      const validation = validateConfig(this.config);
      if (!validation.valid) {
        logger.warn(
          { errors: validation.errors },
          'Invalid memory configuration',
        );
        return false;
      }

      // Ensure data directory exists
      const memoryDir = path.dirname(this.config.lancedbPath);
      if (!fs.existsSync(memoryDir)) {
        fs.mkdirSync(memoryDir, { recursive: true });
      }

      // Initialize embedding service
      this.embedding = createEmbeddingService(this.config.embedding);

      // Initialize storage
      this.memoryStore = await createMemoryStore(this.config.lancedbPath);

      // Initialize retriever
      this.retriever = createMemoryRetriever(
        this.memoryStore,
        this.embedding,
        this.config.retrieval,
      );

      this.initialized = true;
      logger.info('Memory system initialized successfully');
      return true;
    } catch (err) {
      logger.error({ error: err }, 'Failed to initialize memory system');
      return false;
    }
  }

  /**
   * Store a new memory
   */
  async store(options: {
    text: string;
    category?: MemoryCategory;
    scope?: MemoryScope;
    scopeId?: string;
    importance?: number;
    metadata?: Record<string, unknown>;
  }): Promise<MemoryStoreResult> {
    if (!this.initialized) {
      return { success: false, error: 'Memory system not initialized' };
    }

    const {
      text,
      category,
      scope = this.config.scopes.default,
      scopeId,
      importance = 0.5,
      metadata,
    } = options;

    // Apply quality filters
    if (!isWorthStoring(text)) {
      logger.debug(
        { text: text.slice(0, 50) },
        'Memory filtered by quality check',
      );
      return { success: false, error: 'Content quality too low' };
    }

    try {
      // Auto-categorize if not provided
      const finalCategory = category || categorizeText(text);

      // Generate embedding
      const vector = await this.embedding!.embed(text, 'passage');

      // Create memory
      const memory: Memory = {
        id: `mem-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`,
        text,
        vector,
        category: finalCategory,
        scope,
        scopeId: scopeId || 'default',
        importance: Math.max(0, Math.min(1, importance)),
        timestamp: Date.now(),
        accessCount: 0,
        lastAccessed: Date.now(),
        metadata: metadata ? JSON.stringify(metadata) : undefined,
      };

      const storeResult = await this.memoryStore!.store(memory);

      if (storeResult.success) {
        logger.debug(
          { id: memory.id, category: finalCategory },
          'Memory stored',
        );
      }

      return storeResult;
    } catch (err) {
      logger.error({ error: err }, 'Failed to store memory');
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  /**
   * Search memories
   */
  async search(query: MemorySearchQuery): Promise<MemorySearchResult[]> {
    if (!this.initialized || !this.retriever) {
      return [];
    }

    // Check if we should skip retrieval for this query
    if (shouldSkipRetrieval(query.query)) {
      logger.debug(
        { query: query.query.slice(0, 50) },
        'Skipping retrieval for low-value query',
      );
      return [];
    }

    try {
      return await this.retriever.search(query);
    } catch (err) {
      logger.error({ error: err }, 'Memory search failed');
      return [];
    }
  }

  /**
   * Get a memory by ID
   */
  async get(id: string): Promise<Memory | null> {
    if (!this.initialized || !this.memoryStore) {
      return null;
    }

    return await this.memoryStore.get(id);
  }

  /**
   * Update a memory
   */
  async update(id: string, updates: Partial<Memory>): Promise<boolean> {
    if (!this.initialized || !this.memoryStore) {
      return false;
    }

    return await this.memoryStore.update(id, updates);
  }

  /**
   * Delete a memory by ID
   */
  async delete(id: string): Promise<boolean> {
    if (!this.initialized || !this.memoryStore) {
      return false;
    }

    return await this.memoryStore.delete(id);
  }

  /**
   * Delete memories by query
   */
  async deleteByQuery(query: {
    scope?: MemoryScope;
    scopeId?: string;
    category?: MemoryCategory;
  }): Promise<MemoryDeleteResult> {
    if (!this.initialized || !this.memoryStore) {
      return { success: false, count: 0, error: 'Not initialized' };
    }

    return await this.memoryStore.deleteFilter(query);
  }

  /**
   * List memories
   */
  async list(options: {
    scope?: MemoryScope;
    scopeId?: string;
    category?: MemoryCategory;
    limit?: number;
  }): Promise<Memory[]> {
    if (!this.initialized || !this.memoryStore) {
      return [];
    }

    return await this.memoryStore.list(options);
  }

  /**
   * Get memory statistics
   */
  async getStats(options?: {
    scope?: MemoryScope;
    scopeId?: string;
  }): Promise<MemoryStats> {
    if (!this.initialized || !this.memoryStore) {
      return {
        totalMemories: 0,
        byCategory: {
          preference: 0,
          fact: 0,
          decision: 0,
          entity: 0,
          context: 0,
          other: 0,
        },
        byScope: {
          global: 0,
          group: 0,
          agent: 0,
          custom: 0,
        },
        avgImportance: 0,
        totalAccessCount: 0,
      };
    }

    return await this.memoryStore.getStats(options);
  }

  /**
   * Find similar memories
   */
  async findSimilar(
    memoryId: string,
    options?: { limit?: number; scope?: string },
  ): Promise<MemorySearchResult[]> {
    if (!this.initialized || !this.retriever) {
      return [];
    }

    return await this.retriever.findSimilar(memoryId, options);
  }

  /**
   * Clear all memories
   */
  async clear(): Promise<boolean> {
    if (!this.initialized || !this.memoryStore) {
      return false;
    }

    return await this.memoryStore.clear();
  }

  /**
   * Close the memory system
   */
  async close(): Promise<void> {
    if (this.memoryStore) {
      await this.memoryStore.close();
      this.memoryStore = null;
    }
    this.embedding = null;
    this.retriever = null;
    this.initialized = false;
  }

  /**
   * Check if memory system is enabled
   */
  isEnabled(): boolean {
    return this.config.enabled;
  }

  /**
   * Check if auto-capture is enabled
   */
  isAutoCaptureEnabled(): boolean {
    return this.config.autoCapture.enabled;
  }

  /**
   * Check if auto-recall is enabled
   */
  isAutoRecallEnabled(): boolean {
    return this.config.autoRecall.enabled;
  }

  /**
   * Get auto-recall configuration
   */
  getAutoRecallConfig(): AutoRecallConfig {
    return this.config.autoRecall;
  }

  /**
   * Get auto-capture configuration
   */
  getAutoCaptureConfig(): AutoCaptureConfig {
    return this.config.autoCapture;
  }

  /**
   * Get the embedding service
   */
  getEmbeddingService(): EmbeddingService | null {
    return this.embedding;
  }

  /**
   * Update configuration
   */
  updateConfig(updates: Partial<MemoryConfig>): void {
    this.config = { ...this.config, ...updates };

    // Re-initialize if needed
    if (updates.enabled !== undefined && !updates.enabled) {
      this.close();
    }
  }
}

/**
 * Global memory system instance
 */
let globalMemorySystem: MemorySystem | null = null;

/**
 * Initialize the global memory system
 */
export async function initializeMemorySystem(
  projectRoot: string,
  customConfigPath?: string,
): Promise<MemorySystem | null> {
  if (globalMemorySystem) {
    return globalMemorySystem;
  }

  const config = loadMemoryConfig(customConfigPath, projectRoot);
  const system = new MemorySystem(config, projectRoot);

  const success = await system.initialize();
  if (!success) {
    return null;
  }

  globalMemorySystem = system;
  return system;
}

/**
 * Get the global memory system instance
 */
export function getMemorySystem(): MemorySystem | null {
  return globalMemorySystem;
}

/**
 * Close the global memory system
 */
export async function closeMemorySystem(): Promise<void> {
  if (globalMemorySystem) {
    await globalMemorySystem.close();
    globalMemorySystem = null;
  }
}

// Export all types
export * from './types.js';

// Export config functions
export {
  loadMemoryConfig,
  saveMemoryConfig,
  validateConfig,
  expandEnv,
} from './config.js';

// Export embedding service
export { createEmbeddingService, EmbeddingService } from './embedding.js';

// Export storage
export { createMemoryStore, MemoryStore } from './store.js';

// Export retriever
export { createMemoryRetriever, MemoryRetriever } from './retriever.js';

// Export filters
export {
  applyQualityFilters,
  isWorthStoring,
  categorizeText,
  shouldSkipRetrieval,
  adaptiveRetrievalParams,
  extractEntities,
  isGreeting,
  isConfirmation,
  isLowContent,
  isNoise,
  calculateContentQuality,
} from './filters.js';
