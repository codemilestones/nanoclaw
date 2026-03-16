/**
 * NanoClaw Advanced Memory System - Configuration Management
 *
 * Loads configuration from environment variables and config files.
 * Supports environment variable expansion in config values.
 */

import fs from 'fs';
import os from 'os';
import path from 'path';

import type {
  AutoCaptureConfig,
  AutoRecallConfig,
  EmbeddingProviderConfig,
  MemoryConfig,
  RetrievalConfig,
  RerankerConfig,
  ScopeConfig,
} from './types.js';
import { logger } from '../logger.js';

/** Default config directory path */
const DEFAULT_CONFIG_DIR = path.join(os.homedir(), '.config', 'nanoclaw');
/** Default config file path */
const DEFAULT_CONFIG_PATH = path.join(DEFAULT_CONFIG_DIR, 'memory.json');

/**
 * Expand environment variables in a string
 * Supports ${VAR_NAME} syntax
 */
export function expandEnv(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, varName) => {
    return process.env[varName] || '';
  });
}

/**
 * Load and parse a JSON config file
 */
function loadConfigFile(filePath: string): Record<string, unknown> | null {
  try {
    if (!fs.existsSync(filePath)) {
      logger.debug({ path: filePath }, 'Config file not found, using defaults');
      return null;
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    const config = JSON.parse(content);
    logger.info({ path: filePath }, 'Config file loaded');
    return config;
  } catch (err) {
    logger.warn({ path: filePath, error: err }, 'Failed to load config file');
    return null;
  }
}

/**
 * Get the default embedding provider config
 */
function getDefaultEmbeddingConfig(): EmbeddingProviderConfig {
  const apiKey = process.env.JINA_API_KEY || process.env.OPENAI_API_KEY;
  const provider = (process.env.MEMORY_EMBEDDING_PROVIDER ||
    (process.env.JINA_API_KEY ? 'jina' : 'openai')) as 'jina' | 'openai';

  const configs: Record<string, EmbeddingProviderConfig> = {
    jina: {
      provider: 'jina',
      apiKey: apiKey || '${JINA_API_KEY}',
      model: process.env.MEMORY_EMBEDDING_MODEL || 'jina-embeddings-v3',
      baseURL: 'https://api.jina.ai/v1',
      dimensions: parseInt(
        process.env.MEMORY_EMBEDDING_DIMENSIONS || '1024',
        10,
      ),
      taskQuery: 'retrieval.query',
      taskPassage: 'retrieval.passage',
      timeout: 30000,
      batchSize: 100,
    },
    openai: {
      provider: 'openai',
      apiKey: '${OPENAI_API_KEY}',
      model: process.env.MEMORY_EMBEDDING_MODEL || 'text-embedding-3-small',
      baseURL: process.env.OPENAI_BASE_URL,
      dimensions: parseInt(
        process.env.MEMORY_EMBEDDING_DIMENSIONS || '1536',
        10,
      ),
      timeout: 30000,
      batchSize: 100,
    },
    siliconflow: {
      provider: 'siliconflow',
      apiKey: '${SILICONFLOW_API_KEY}',
      model: process.env.MEMORY_EMBEDDING_MODEL || 'BAAI/bge-large-zh-v1.5',
      baseURL: 'https://api.siliconflow.cn/v1',
      dimensions: parseInt(
        process.env.MEMORY_EMBEDDING_DIMENSIONS || '1024',
        10,
      ),
      timeout: 30000,
      batchSize: 50,
    },
    ollama: {
      provider: 'ollama',
      model: process.env.MEMORY_EMBEDDING_MODEL || 'nomic-embed-text',
      baseURL: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
      dimensions: parseInt(
        process.env.MEMORY_EMBEDDING_DIMENSIONS || '768',
        10,
      ),
      timeout: 60000,
      batchSize: 10,
    },
  };

  return configs[provider] || configs.jina;
}

/**
 * Get default retrieval config
 */
function getDefaultRetrievalConfig(): RetrievalConfig {
  return {
    mode: 'hybrid',
    vectorWeight: parseFloat(process.env.MEMORY_VECTOR_WEIGHT || '0.7'),
    bm25Weight: parseFloat(process.env.MEMORY_BM25_WEIGHT || '0.3'),
    minScore: parseFloat(process.env.MEMORY_MIN_SCORE || '0.35'),
    hardMinScore: parseFloat(process.env.MEMORY_HARD_MIN_SCORE || '0.45'),
    candidatePoolSize: parseInt(
      process.env.MEMORY_CANDIDATE_POOL_SIZE || '20',
      10,
    ),
    rerank: {
      provider:
        (process.env.MEMORY_RERANK_PROVIDER as
          | 'jina'
          | 'siliconflow'
          | 'none') || 'none',
      model: process.env.MEMORY_RERANK_MODEL || 'jina-reranker-v2-base',
      apiKey: '${JINA_API_KEY}',
      topK: parseInt(process.env.MEMORY_RERANK_TOP_K || '10', 10),
    },
  };
}

/**
 * Get default auto-capture config
 */
function getDefaultAutoCaptureConfig(): AutoCaptureConfig {
  return {
    enabled: process.env.MEMORY_AUTO_CAPTURE !== 'false',
    categories: ['preference', 'fact', 'decision', 'entity'],
    minImportance: parseFloat(process.env.MEMORY_MIN_IMPORTANCE || '0.3'),
    maxPerConversation: parseInt(
      process.env.MEMORY_MAX_PER_CONVERSATION || '5',
      10,
    ),
  };
}

/**
 * Get default auto-recall config
 */
function getDefaultAutoRecallConfig(): AutoRecallConfig {
  return {
    enabled: process.env.MEMORY_AUTO_RECALL !== 'false',
    topK: parseInt(process.env.MEMORY_RECALL_TOP_K || '3', 10),
    categories: ['preference', 'fact', 'decision', 'entity'],
  };
}

/**
 * Get default scope config
 */
function getDefaultScopeConfig(): ScopeConfig {
  return {
    default: 'group',
    definitions: {
      global: {
        description: 'Shared knowledge across all groups',
        isolated: false,
      },
      group: {
        description: 'Group-specific memories',
        isolated: true,
      },
      agent: {
        description: 'Agent-specific knowledge',
        isolated: true,
      },
      custom: {
        description: 'Custom scoped memories',
        isolated: true,
      },
    },
  };
}

/**
 * Deep merge two objects
 */
function deepMerge<T>(target: T, source: Partial<T>): T {
  const result = { ...target };

  for (const key of Object.keys(source) as Array<keyof T>) {
    const sourceValue = source[key];
    const targetValue = result[key];

    if (
      sourceValue &&
      typeof sourceValue === 'object' &&
      !Array.isArray(sourceValue) &&
      targetValue &&
      typeof targetValue === 'object' &&
      !Array.isArray(targetValue)
    ) {
      result[key] = deepMerge(
        targetValue as T[keyof T],
        sourceValue as Partial<T[keyof T]>,
      ) as T[keyof T];
    } else if (sourceValue !== undefined) {
      result[key] = sourceValue as T[keyof T];
    }
  }

  return result;
}

/**
 * Expand environment variables recursively in an object
 */
function expandEnvRecursive<T>(obj: T): T {
  if (typeof obj === 'string') {
    return expandEnv(obj) as T;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) => expandEnvRecursive(item)) as T;
  }

  if (obj && typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = expandEnvRecursive(value);
    }
    return result as T;
  }

  return obj;
}

/**
 * Load memory configuration from file and environment variables
 */
export function loadMemoryConfig(
  customPath?: string,
  projectRoot?: string,
): MemoryConfig {
  // Ensure config directory exists
  const configPath = customPath || DEFAULT_CONFIG_PATH;
  const configDir = path.dirname(configPath);
  if (!fs.existsSync(configDir)) {
    fs.mkdirSync(configDir, { recursive: true });
  }

  // Load default configuration
  const defaultConfig: MemoryConfig = {
    enabled:
      process.env.MEMORY_ENABLED === 'true' ||
      process.env.MEMORY_ENABLED !== 'false',
    lancedbPath:
      process.env.MEMORY_LANCEDB_PATH ||
      (projectRoot
        ? path.join(projectRoot, 'data', 'memory')
        : path.join(process.cwd(), 'data', 'memory')),
    embedding: getDefaultEmbeddingConfig(),
    retrieval: getDefaultRetrievalConfig(),
    autoCapture: getDefaultAutoCaptureConfig(),
    autoRecall: getDefaultAutoRecallConfig(),
    scopes: getDefaultScopeConfig(),
  };

  // Load and merge file config
  const fileConfig = loadConfigFile(configPath);
  if (fileConfig) {
    try {
      // Expand env vars in file config before merging
      const expandedConfig = expandEnvRecursive(fileConfig);
      return deepMerge(defaultConfig, expandedConfig as Partial<MemoryConfig>);
    } catch (err) {
      logger.warn({ error: err }, 'Failed to merge config, using defaults');
    }
  }

  return defaultConfig;
}

/**
 * Save memory configuration to file
 */
export function saveMemoryConfig(
  config: MemoryConfig,
  filePath?: string,
): boolean {
  const configPath = filePath || DEFAULT_CONFIG_PATH;
  try {
    const configDir = path.dirname(configPath);
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }

    fs.writeFileSync(configPath, JSON.stringify(config, null, 2) + '\n');
    logger.info({ path: configPath }, 'Config saved');
    return true;
  } catch (err) {
    logger.error({ path: configPath, error: err }, 'Failed to save config');
    return false;
  }
}

/**
 * Validate memory configuration
 */
export function validateConfig(config: MemoryConfig): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.enabled) {
    return { valid: true, errors: [] };
  }

  // Validate embedding config
  if (!config.embedding.model) {
    errors.push('embedding.model is required');
  }
  if (config.embedding.dimensions <= 0) {
    errors.push('embedding.dimensions must be positive');
  }

  // Validate retrieval config
  const { vectorWeight, bm25Weight } = config.retrieval;
  if (config.retrieval.mode === 'hybrid') {
    if (Math.abs(vectorWeight + bm25Weight - 1.0) > 0.01) {
      errors.push(
        `retrieval.vectorWeight + retrieval.bm25Weight must equal 1.0, got ${vectorWeight + bm25Weight}`,
      );
    }
  }

  // Validate scores are in range
  if (config.retrieval.minScore < 0 || config.retrieval.minScore > 1) {
    errors.push('retrieval.minScore must be between 0 and 1');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}
