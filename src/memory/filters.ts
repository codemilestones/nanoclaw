/**
 * NanoClaw Advanced Memory System - Quality Filters
 *
 * Implements:
 * - Adaptive retrieval: skip greetings, simple confirmations
 * - Noise filtering: remove low-quality content
 * - Content validation: check for meaningful content
 */

import type { MemoryCategory } from './types.js';

/**
 * Simple greeting patterns
 */
const GREETING_PATTERNS = [
  /^(hi|hello|hey|yo|sup)[!.,\s]*$/i,
  /^(good morning|good evening|good night)[!.,\s]*$/i,
  /^(howdy|greetings|salutations)[!.,\s]*$/i,
  /^哈[喽罗]/i,
  /^你好[啊吗嘛]*/,
  /^早上好|晚上好|晚安/i,
];

/**
 * Simple confirmation patterns
 */
const CONFIRMATION_PATTERNS = [
  /^(ok|okay|sure|alright|got it|understood|sounds good)[!.,\s]*$/i,
  /^(yes|no|yep|nope|yeah|nah)[!.,\s]*$/i,
  /^(thanks?|thx|ty)[!.,\s]*$/i,
  /^(cool|awesome|great|perfect)[!.,\s]*$/i,
  /^好的|没问题|可以|行/i,
  /^谢谢|多谢|感谢/i,
];

/**
 * Very short/low-content patterns
 */
const LOW_CONTENT_PATTERNS = [
  /^[.?!]{1,5}$/,
  /^(uh+|oh+|um+|ah+)[!.,\s]*$/i,
  /^(lol|lmao|rofl|haha)[!.,\s]*$/i,
  /^(嗯|啊|哦|呃|额)[！。，\s]*$/,
];

/**
 * Noise keywords (don't store as memories)
 */
const NOISE_KEYWORDS = [
  'ping',
  'pong',
  'test',
  'testing',
  'hello world',
  '测试',
  'ping',
  'pong',
];

/**
 * Check if text is a simple greeting
 */
export function isGreeting(text: string): boolean {
  const trimmed = text.trim().toLowerCase();
  return GREETING_PATTERNS.some((pattern) => pattern.test(trimmed));
}

/**
 * Check if text is a simple confirmation
 */
export function isConfirmation(text: string): boolean {
  const trimmed = text.trim().toLowerCase();
  return CONFIRMATION_PATTERNS.some((pattern) => pattern.test(trimmed));
}

/**
 * Check if text has very low content value
 */
export function isLowContent(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length < 3) return true;

  const lower = trimmed.toLowerCase();
  return LOW_CONTENT_PATTERNS.some((pattern) => pattern.test(lower));
}

/**
 * Check if text is noise (testing, etc.)
 */
export function isNoise(text: string): boolean {
  const lower = text.toLowerCase();
  return NOISE_KEYWORDS.some((keyword) => lower.includes(keyword));
}

/**
 * Check if query should skip memory retrieval
 * Returns true if the query is too simple to benefit from memory
 */
export function shouldSkipRetrieval(query: string): boolean {
  const trimmed = query.trim();

  // Skip very short queries
  if (trimmed.length < 5) return true;

  // Skip greetings and confirmations
  if (isGreeting(trimmed)) return true;
  if (isConfirmation(trimmed)) return true;
  if (isLowContent(trimmed)) return true;

  // Skip queries that are just questions without context
  const questionOnlyPattern =
    /^(how|what|when|where|why|who|which|can|could|would|should|is|are|do|does|did)[\s?]*$/i;
  if (questionOnlyPattern.test(trimmed)) return true;

  return false;
}

/**
 * Calculate content quality score
 * Returns 0-1, higher is better quality
 */
export function calculateContentQuality(text: string): number {
  let score = 0.5;

  const trimmed = text.trim();

  // Length factor (up to 0.3 points)
  const lengthScore = Math.min(trimmed.length / 200, 1) * 0.3;
  score += lengthScore;

  // Unique words factor (up to 0.2 points)
  const words = trimmed.toLowerCase().split(/\s+/);
  const uniqueWords = new Set(words);
  const uniqueScore = Math.min(uniqueWords.size / words.length, 1) * 0.2;
  score += uniqueScore;

  // Contains numbers/dates (0.1 points)
  if (/\d+/.test(trimmed)) {
    score += 0.1;
  }

  // Contains entities (capitalized words) (0.1 points)
  if (/[A-Z][a-z]+/.test(trimmed)) {
    score += 0.1;
  }

  // Penalize for being mostly noise patterns
  if (isGreeting(trimmed) || isConfirmation(trimmed)) {
    score -= 0.3;
  }

  if (isLowContent(trimmed)) {
    score -= 0.2;
  }

  return Math.max(0, Math.min(1, score));
}

/**
 * Check if text is worth storing as a memory
 */
export function isWorthStoring(text: string): boolean {
  const quality = calculateContentQuality(text);
  return quality >= 0.3;
}

/**
 * Extract meaningful entities from text
 * Returns potential named entities
 */
export function extractEntities(text: string): string[] {
  const entities: string[] = [];

  // Extract capitalized words (potential proper nouns)
  const capitalizedPattern = /\b[A-Z][a-z]+\b/g;
  const matches = text.match(capitalizedPattern);
  if (matches) {
    for (const match of matches) {
      // Filter out common words that start sentences
      if (
        !['The', 'This', 'That', 'These', 'Those', 'A', 'An'].includes(match)
      ) {
        entities.push(match);
      }
    }
  }

  return [...new Set(entities)];
}

/**
 * Categorize text into memory category
 * Returns the most likely category
 */
export function categorizeText(text: string): MemoryCategory {
  const lower = text.toLowerCase();

  // Preference indicators
  if (
    /\b(prefer|like|love|enjoy|favorite|hate|dislike|want|wish|hope)\b/i.test(
      text,
    ) ||
    /\b(喜欢|爱|讨厌|想要|希望|偏好|最爱)\b/.test(text)
  ) {
    return 'preference';
  }

  // Decision indicators
  if (
    /\b(decided|chose|selected|picked|going to|will|shall|plan to)\b/i.test(
      text,
    ) ||
    /\b(决定|选择|计划|打算)\b/.test(text)
  ) {
    return 'decision';
  }

  // Entity indicators (names, places)
  if (
    /[A-Z][a-z]+(\s+[A-Z][a-z]+)+/.test(text) ||
    /\b(在|去|到)\s+\S+/.test(text)
  ) {
    return 'entity';
  }

  // Fact indicators
  if (
    /\b(is|are|was|were|has|have|had|can|could|should)\b/i.test(text) ||
    /\b(是|有|能|会)\b/.test(text)
  ) {
    return 'fact';
  }

  // Default to other
  return 'other';
}

/**
 * Adaptive retrieval options
 */
export interface AdaptiveRetrievalOptions {
  /** Maximum memories to retrieve */
  maxMemories?: number;
  /** Minimum relevance score */
  minScore?: number;
  /** Whether to include context memories */
  includeContext?: boolean;
  /** Categories to prioritize */
  priorityCategories?: MemoryCategory[];
}

/**
 * Determine retrieval parameters based on query
 * Adapts retrieval strategy based on query characteristics
 */
export function adaptiveRetrievalParams(
  query: string,
  defaults: AdaptiveRetrievalOptions = {},
): AdaptiveRetrievalOptions {
  const trimmed = query.trim();
  const options: AdaptiveRetrievalOptions = { ...defaults };

  // Short queries: fewer results, higher threshold
  if (trimmed.length < 20) {
    options.maxMemories = options.maxMemories || 3;
    options.minScore = options.minScore !== undefined ? options.minScore : 0.5;
  }
  // Long queries: more results, lower threshold
  else if (trimmed.length > 100) {
    options.maxMemories = options.maxMemories || 7;
    options.minScore = options.minScore !== undefined ? options.minScore : 0.3;
  }
  // Medium queries: balanced
  else {
    options.maxMemories = options.maxMemories || 5;
    options.minScore = options.minScore !== undefined ? options.minScore : 0.4;
  }

  // Question queries: prioritize facts and context
  if (/\?(？)?$/.test(trimmed)) {
    options.priorityCategories = options.priorityCategories || [
      'fact',
      'context',
      'entity',
    ];
    options.includeContext = true;
  }
  // Preference queries: prioritize preferences and decisions
  else if (
    /\b(prefer|like|want|should|recommend|best)\b/i.test(trimmed) ||
    /\b(推荐|最好|应该|喜欢)\b/.test(trimmed)
  ) {
    options.priorityCategories = options.priorityCategories || [
      'preference',
      'decision',
    ];
  }

  return options;
}

/**
 * Filter and deduplicate memories
 */
export function filterUniqueMemories<T extends { id: string }>(
  memories: T[],
  maxCount?: number,
): T[] {
  const seen = new Set<string>();
  const unique: T[] = [];

  for (const memory of memories) {
    if (!seen.has(memory.id)) {
      seen.add(memory.id);
      unique.push(memory);

      if (maxCount && unique.length >= maxCount) {
        break;
      }
    }
  }

  return unique;
}

/**
 * Sort memories by relevance and importance
 */
export function sortByRelevance<
  T extends { score?: number; importance?: number },
>(memories: T[]): T[] {
  return [...memories].sort((a, b) => {
    const scoreA = (a.score || 0) + (a.importance || 0) * 0.2;
    const scoreB = (b.score || 0) + (b.importance || 0) * 0.2;
    return scoreB - scoreA;
  });
}

/**
 * Extract key phrases from text for memory summarization
 */
export function extractKeyPhrases(text: string, maxPhrases = 3): string[] {
  const phrases: string[] = [];

  // Extract quoted text
  const quotedPattern = /"([^"]+)"/g;
  const quotes = text.match(quotedPattern);
  if (quotes) {
    phrases.push(...quotes.map((q) => q.slice(1, -1)));
  }

  // Extract sentences with keywords
  const sentences = text.split(/[.!?。！？]/);
  const keywordPattern =
    /\b(prefer|like|want|decided|important|remember|note|key)\b/i;
  const keywordPatternCn = /(偏好|喜欢|想要|决定|重要|记住|注意|关键)/;

  for (const sentence of sentences) {
    if (keywordPattern.test(sentence) || keywordPatternCn.test(sentence)) {
      phrases.push(sentence.trim());
    }
  }

  return phrases.slice(0, maxPhrases);
}

/**
 * Quality filter configuration
 */
export interface QualityFilterConfig {
  /** Minimum content quality to store */
  minQuality?: number;
  /** Skip greetings */
  skipGreetings?: boolean;
  /** Skip confirmations */
  skipConfirmations?: boolean;
  /** Skip low content */
  skipLowContent?: boolean;
  /** Skip noise */
  skipNoise?: boolean;
}

/**
 * Apply quality filters to text
 * Returns true if text passes all filters
 */
export function applyQualityFilters(
  text: string,
  config: QualityFilterConfig = {},
): boolean {
  const {
    minQuality = 0.3,
    skipGreetings = true,
    skipConfirmations = true,
    skipLowContent = true,
    skipNoise = true,
  } = config;

  // Check quality score
  const quality = calculateContentQuality(text);
  if (quality < minQuality) {
    return false;
  }

  // Apply filters
  if (skipGreetings && isGreeting(text)) {
    return false;
  }

  if (skipConfirmations && isConfirmation(text)) {
    return false;
  }

  if (skipLowContent && isLowContent(text)) {
    return false;
  }

  if (skipNoise && isNoise(text)) {
    return false;
  }

  return true;
}
