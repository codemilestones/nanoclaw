/**
 * NanoClaw Advanced Memory System - Fact Extractor
 *
 * Extracts meaningful facts, preferences, decisions, and entities
 * from conversations for memory storage.
 */

import type { Memory, MemoryCategory, ExtractionResult } from './types.js';
import { categorizeText, extractEntities, isWorthStoring } from './filters.js';
import { logger } from '../logger.js';

/**
 * Extraction patterns for different memory types
 */
const EXTRACTION_PATTERNS = {
  preference: [
    // English patterns
    /\b(I|you)(?:\s+(?:really\s+)?(?:prefer|like|love|enjoy|hate|dislike))\s+(.+?)(?:\.|$)/gi,
    /\b(my|your)\s+(?:favorite|preference)(?:\s+is)?\s+(.+?)(?:\.|$)/gi,
    // Chinese patterns
    /[\u4e00-\u9fff]+(?:喜欢|爱|讨厌|偏好|不爱)(?:\s*[\u4e00-\u9fff]+)?/g,
  ],
  decision: [
    // English patterns
    /\b(I|we|you)\s+(?:have\s+)?decided\s+to\s+(.+?)(?:\.|$)/gi,
    /\b(the\s+)?decision\s+(?:is\s+)?(?:to\s+)?(.+?)(?:\.|$)/gi,
    /\bgoing\s+to\s+will\s+(.+?)(?:\.|$)/gi,
    // Chinese patterns
    /[\u4e00-\u9fff]+(?:决定|选择)(?:\s*[\u4e00-\u9fff]+){1,5}/g,
  ],
  fact: [
    // English patterns
    /\b(I am|You are|He is|She is|They are|It is)\s+(.+?)(?:\.|$)/gi,
    /\b(I have|You have)\s+(.+?)(?:\.|$)/gi,
    // Chinese patterns
    /[\u4e00-\u9fff]+(?:是|有)(?:\s*[\u4e00-\u9fff]+){1,5}/g,
  ],
  entity: [
    // Names (consecutive capitalized words)
    /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g,
    // Chinese names
    /[\u4e00-\u9fff]{2,4}/g,
  ],
  context: [
    // Contextual information
    /\b(context|background|info|information)\b\s*[:\s]+(.+?)(?:\.|$)/gi,
    // Chinese patterns
    /[\u4e00-\u9fff]+(?:背景|信息)(?:\s*[\u4e00-\u9fff]+){1,5}/g,
  ],
  other: [
    // Generic catch-all pattern
    /.+/g,
  ],
};

/**
 * Extracted memory candidate
 */
interface MemoryCandidate {
  text: string;
  category: MemoryCategory;
  importance: number;
  confidence: number;
}

/**
 * Extract facts from a message
 */
export class FactExtractor {
  private maxMemoriesPerMessage: number;
  private minConfidence: number;

  constructor(options?: {
    maxMemoriesPerMessage?: number;
    minConfidence?: number;
  }) {
    this.maxMemoriesPerMessage = options?.maxMemoriesPerMessage || 3;
    this.minConfidence = options?.minConfidence || 0.3;
  }

  /**
   * Extract memories from a single message
   */
  extractFromMessage(message: {
    text: string;
    sender?: string;
    isFromUser?: boolean;
  }): MemoryCandidate[] {
    const { text, isFromUser = true } = message;

    // Only extract from user messages (not agent responses)
    if (!isFromUser) {
      return [];
    }

    const candidates: MemoryCandidate[] = [];

    // Extract using pattern matching
    const categoriesToCheck: MemoryCategory[] = [
      'preference',
      'decision',
      'fact',
      'entity',
      'context',
      'other',
    ];
    for (const category of categoriesToCheck) {
      const patterns =
        EXTRACTION_PATTERNS[category as keyof typeof EXTRACTION_PATTERNS] || [];

      for (const pattern of patterns) {
        const matches = text.matchAll(pattern);
        for (const match of matches) {
          const extractedText = match[1] || match[0];

          // Clean up the extracted text
          const cleaned = this.cleanExtractedText(extractedText);

          if (cleaned && cleaned.length >= 3 && isWorthStoring(cleaned)) {
            candidates.push({
              text: cleaned,
              category,
              importance: this.calculateImportance(cleaned, category),
              confidence: this.calculateConfidence(cleaned, category, text),
            });
          }
        }
      }
    }

    // Extract entities
    const entities = extractEntities(text);
    for (const entity of entities) {
      if (entity.length >= 2) {
        candidates.push({
          text: `${entity} was mentioned`,
          category: 'entity',
          importance: 0.3,
          confidence: 0.6,
        });
      }
    }

    // Remove duplicates
    const unique = this.deduplicateCandidates(candidates);

    // Sort by confidence and importance
    unique.sort((a, b) => {
      const scoreA = a.confidence * a.importance;
      const scoreB = b.confidence * b.importance;
      return scoreB - scoreA;
    });

    // Return top N
    return unique.slice(0, this.maxMemoriesPerMessage);
  }

  /**
   * Extract memories from a conversation
   */
  extractFromConversation(
    messages: Array<{
      text: string;
      sender?: string;
      isFromUser?: boolean;
    }>,
  ): MemoryCandidate[] {
    const allCandidates: MemoryCandidate[] = [];

    for (const message of messages) {
      const candidates = this.extractFromMessage(message);
      allCandidates.push(...candidates);
    }

    // Deduplicate across all messages
    const unique = this.deduplicateCandidates(allCandidates);

    // Sort by combined score (frequency * confidence * importance)
    const frequencyMap = new Map<string, number>();
    for (const candidate of unique) {
      const key = this.getCanonicalKey(candidate.text, candidate.category);
      frequencyMap.set(key, (frequencyMap.get(key) || 0) + 1);
    }

    unique.sort((a, b) => {
      const keyA = this.getCanonicalKey(a.text, a.category);
      const keyB = this.getCanonicalKey(b.text, b.category);
      const freqA = frequencyMap.get(keyA) || 1;
      const freqB = frequencyMap.get(keyB) || 1;

      const scoreA = freqA * a.confidence * a.importance;
      const scoreB = freqB * b.confidence * b.importance;

      return scoreB - scoreA;
    });

    return unique.slice(0, this.maxMemoriesPerMessage);
  }

  /**
   * Clean extracted text
   */
  private cleanExtractedText(text: string): string {
    return (
      text
        .trim()
        // Remove trailing punctuation
        .replace(/[.!?,;:。！？，；：]+$/, '')
        // Remove quotes
        .replace(/^["']|["']$/g, '')
        // Normalize whitespace
        .replace(/\s+/g, ' ')
    );
  }

  /**
   * Calculate importance score
   */
  private calculateImportance(text: string, category: MemoryCategory): number {
    let importance = 0.5;

    // Length factor
    if (text.length > 20) importance += 0.1;
    if (text.length > 50) importance += 0.1;

    // Category-specific adjustments
    switch (category) {
      case 'preference':
        importance += 0.2;
        break;
      case 'decision':
        importance += 0.3;
        break;
      case 'fact':
        importance += 0.1;
        break;
      case 'entity':
        importance -= 0.1;
        break;
    }

    // Keywords indicating importance
    const importantKeywords = [
      'important',
      'remember',
      'note',
      'key',
      'crucial',
      '重要',
      '记住',
      '注意',
      '关键',
    ];
    for (const keyword of importantKeywords) {
      if (text.toLowerCase().includes(keyword)) {
        importance += 0.15;
        break;
      }
    }

    return Math.max(0, Math.min(1, importance));
  }

  /**
   * Calculate confidence score
   */
  private calculateConfidence(
    text: string,
    category: MemoryCategory,
    context: string,
  ): number {
    let confidence = 0.5;

    // Pattern match confidence
    if (category === 'preference') {
      if (/\bprefer|like|love|hate|dislike\b/i.test(text)) {
        confidence = 0.8;
      } else if (/\b喜欢|爱|讨厌|偏好/.test(text)) {
        confidence = 0.8;
      }
    } else if (category === 'decision') {
      if (/\bdecided|chosen|going to\b/i.test(text)) {
        confidence = 0.85;
      } else if (/\b决定|选择|打算/.test(text)) {
        confidence = 0.85;
      }
    } else if (category === 'fact') {
      if (/\bis|are|have|has\b/i.test(text)) {
        confidence = 0.7;
      } else if (/\b是|有|能|会/.test(text)) {
        confidence = 0.7;
      }
    }

    // Contextual confidence
    const isFirstPerson =
      /\b(I|my|me|mine)\b/i.test(context) ||
      /[\u4e00-\u9fff]{0,2}[我我的]/.test(context);
    if (isFirstPerson) {
      confidence += 0.1;
    }

    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Get canonical key for deduplication
   */
  private getCanonicalKey(text: string, category: MemoryCategory): string {
    return `${category}:${text.toLowerCase().trim()}`;
  }

  /**
   * Deduplicate candidates
   */
  private deduplicateCandidates(
    candidates: MemoryCandidate[],
  ): MemoryCandidate[] {
    const seen = new Set<string>();
    const unique: MemoryCandidate[] = [];

    for (const candidate of candidates) {
      const key = this.getCanonicalKey(candidate.text, candidate.category);

      // Check for similar existing candidates
      let isDuplicate = false;
      for (const existingKey of seen) {
        if (this.isSimilar(key, existingKey)) {
          isDuplicate = true;
          break;
        }
      }

      if (!isDuplicate) {
        seen.add(key);
        unique.push(candidate);
      }
    }

    return unique;
  }

  /**
   * Check if two keys are similar
   */
  private isSimilar(key1: string, key2: string): boolean {
    const [cat1, text1] = key1.split(':');
    const [cat2, text2] = key2.split(':');

    if (cat1 !== cat2) return false;

    // Check for exact match
    if (text1 === text2) return true;

    // Check for high similarity (simple Levenshtein approximation)
    const similarity = this.stringSimilarity(text1, text2);
    return similarity > 0.85;
  }

  /**
   * Simple string similarity
   */
  private stringSimilarity(s1: string, s2: string): number {
    if (s1 === s2) return 1;

    const longer = s1.length > s2.length ? s1 : s2;
    const shorter = s1.length > s2.length ? s2 : s1;

    if (longer.length === 0) return 1;

    const editDistance = this.levenshteinDistance(longer, shorter);
    return (longer.length - editDistance) / longer.length;
  }

  /**
   * Levenshtein distance
   */
  private levenshteinDistance(s1: string, s2: string): number {
    const m = s1.length;
    const n = s2.length;
    const dp: number[][] = [];

    for (let i = 0; i <= m; i++) {
      dp[i] = [i];
    }
    for (let j = 0; j <= n; j++) {
      dp[0][j] = j;
    }

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (s1[i - 1] === s2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = Math.min(
            dp[i - 1][j] + 1,
            dp[i][j - 1] + 1,
            dp[i - 1][j - 1] + 1,
          );
        }
      }
    }

    return dp[m][n];
  }
}

/**
 * Create a fact extractor instance
 */
export function createFactExtractor(options?: {
  maxMemoriesPerMessage?: number;
  minConfidence?: number;
}): FactExtractor {
  return new FactExtractor(options);
}

/**
 * Extract and store memories from a conversation
 */
export async function extractAndStoreMemories(
  messages: Array<{ text: string; sender?: string; isFromUser?: boolean }>,
  storeFunction: (options: {
    text: string;
    category: MemoryCategory;
    importance: number;
    scopeId: string;
  }) => Promise<{ success: boolean }>,
  scopeId: string,
  options?: { maxMemories?: number },
): Promise<{ stored: number; failed: number }> {
  const extractor = new FactExtractor({
    maxMemoriesPerMessage: options?.maxMemories || 5,
  });

  const candidates = extractor.extractFromConversation(messages);

  let stored = 0;
  let failed = 0;

  for (const candidate of candidates) {
    try {
      const result = await storeFunction({
        text: candidate.text,
        category: candidate.category,
        importance: candidate.importance,
        scopeId,
      });

      if (result.success) {
        stored++;
        logger.debug(
          { text: candidate.text.slice(0, 50), category: candidate.category },
          'Memory extracted and stored',
        );
      } else {
        failed++;
      }
    } catch (err) {
      logger.warn(
        { error: err, candidate },
        'Failed to store extracted memory',
      );
      failed++;
    }
  }

  return { stored, failed };
}
