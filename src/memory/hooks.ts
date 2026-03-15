/**
 * NanoClaw Advanced Memory System - Auto Capture/Recall Hooks
 *
 * Integrates with the message loop to:
 * - Auto-recall: Inject relevant memories before agent processes messages
 * - Auto-capture: Extract and store memories after agent responds
 */

import type { NewMessage } from '../types.js';
import type { MemorySearchResult, MemoryCategory } from './types.js';
import {
  getMemorySystem,
  shouldSkipRetrieval,
  adaptiveRetrievalParams,
} from './index.js';
import { extractAndStoreMemories } from './extractor.js';
import { logger } from '../logger.js';

/**
 * Format recall results for injection into messages
 */
export function formatRecallResults(
  results: MemorySearchResult[],
  options: {
    includeScore?: boolean;
    includeCategory?: boolean;
    maxLength?: number;
  } = {},
): string {
  if (results.length === 0) {
    return '';
  }

  const {
    includeScore = false,
    includeCategory = true,
    maxLength = 200,
  } = options;

  const lines: string[] = [];
  lines.push('<relevant_memories>');

  // Group by category for better organization
  const byCategory = new Map<MemoryCategory, MemorySearchResult[]>();
  for (const result of results) {
    const category = result.memory.category;
    if (!byCategory.has(category)) {
      byCategory.set(category, []);
    }
    byCategory.get(category)!.push(result);
  }

  // Prioritize categories
  const categoryOrder: MemoryCategory[] = [
    'preference',
    'fact',
    'decision',
    'entity',
    'context',
    'other',
  ];

  for (const category of categoryOrder) {
    const categoryResults = byCategory.get(category);
    if (!categoryResults || categoryResults.length === 0) continue;

    for (const result of categoryResults) {
      let text = result.memory.text;

      // Truncate if too long
      if (text.length > maxLength) {
        text = text.slice(0, maxLength) + '...';
      }

      let line = `- ${text}`;

      if (includeCategory) {
        line += ` [${category}]`;
      }

      if (includeScore && result.score !== undefined) {
        line += ` (relevance: ${result.score.toFixed(2)})`;
      }

      lines.push(line);
    }
  }

  lines.push('</relevant_memories>');

  return lines.join('\n');
}

/**
 * Auto-recall: Inject relevant memories before processing
 */
export async function autoRecall(
  messages: NewMessage[],
  groupFolder: string,
  options?: {
    enabled?: boolean;
    topK?: number;
    categories?: MemoryCategory[];
    minScore?: number;
  },
): Promise<string> {
  const memorySystem = getMemorySystem();
  if (
    !memorySystem ||
    !memorySystem.isEnabled() ||
    !memorySystem.isAutoRecallEnabled()
  ) {
    return '';
  }

  // Check if auto-recall is disabled
  if (options?.enabled === false) {
    return '';
  }

  // Get the last user message for querying
  const userMessages = messages.filter((m) => m.is_from_me);
  if (userMessages.length === 0) {
    return '';
  }

  const lastMessage = userMessages[userMessages.length - 1];
  const query = lastMessage.content;

  // Check if we should skip retrieval for this query
  if (shouldSkipRetrieval(query)) {
    logger.debug(
      { query: query.slice(0, 50) },
      'Skipping auto-recall for low-value query',
    );
    return '';
  }

  try {
    const autoRecallConfig = memorySystem.getAutoRecallConfig();
    const adaptiveParams = adaptiveRetrievalParams(query);

    const results = await memorySystem.search({
      query,
      scope: 'group',
      scopeId: groupFolder,
      categories: options?.categories || autoRecallConfig.categories,
      limit: options?.topK || autoRecallConfig.topK,
      minScore: options?.minScore || autoRecallConfig.topK > 3 ? 0.4 : 0.35,
      includeGlobal: true,
    });

    if (results.length > 0) {
      logger.debug(
        { count: results.length, scopeId: groupFolder },
        'Auto-recall: retrieved memories',
      );
      return formatRecallResults(results);
    }

    return '';
  } catch (err) {
    logger.warn({ error: err }, 'Auto-recall failed');
    return '';
  }
}

/**
 * Auto-capture: Extract and store memories after agent response
 */
export async function autoCapture(
  messages: NewMessage[],
  agentResponse: string,
  groupFolder: string,
  options?: {
    enabled?: boolean;
    maxPerConversation?: number;
  },
): Promise<{ stored: number; failed: number }> {
  const memorySystem = getMemorySystem();
  if (
    !memorySystem ||
    !memorySystem.isEnabled() ||
    !memorySystem.isAutoCaptureEnabled()
  ) {
    return { stored: 0, failed: 0 };
  }

  // Check if auto-capture is disabled
  if (options?.enabled === false) {
    return { stored: 0, failed: 0 };
  }

  try {
    const autoCaptureConfig = memorySystem.getAutoCaptureConfig();

    // Prepare messages for extraction
    const messagesForExtraction = messages.map((m) => ({
      text: m.content,
      sender: m.sender_name,
      isFromUser: m.is_from_me,
    }));

    // Extract and store memories
    const result = await extractAndStoreMemories(
      messagesForExtraction,
      async (options) => {
        // Cast to any to bypass the private method access check
        const sys = memorySystem as any;
        const storeResult = await sys.store({
          text: options.text,
          category: options.category,
          importance: options.importance,
          scope: 'group',
          scopeId: groupFolder,
        });
        return storeResult;
      },
      groupFolder,
      {
        maxMemories:
          options?.maxPerConversation || autoCaptureConfig.maxPerConversation,
      },
    );

    if (result.stored > 0) {
      logger.info(
        { stored: result.stored, scopeId: groupFolder },
        'Auto-capture: memories extracted and stored',
      );
    }

    return result;
  } catch (err) {
    logger.warn({ error: err }, 'Auto-capture failed');
    return { stored: 0, failed: 0 };
  }
}

/**
 * Conversation context for memory extraction
 */
export interface ConversationContext {
  messages: NewMessage[];
  agentResponse?: string;
  groupFolder: string;
}

/**
 * Process both auto-recall and auto-capture
 */
export async function processMemoryHooks(
  context: ConversationContext,
  phase: 'recall' | 'capture',
): Promise<string | { stored: number; failed: number }> {
  switch (phase) {
    case 'recall':
      return await autoRecall(context.messages, context.groupFolder);

    case 'capture':
      return await autoCapture(
        context.messages,
        context.agentResponse || '',
        context.groupFolder,
      );

    default:
      logger.warn({ phase }, 'Unknown memory hook phase');
      return phase === 'recall' ? '' : { stored: 0, failed: 0 };
  }
}

/**
 * Initialize memory hooks for a group
 */
export function initializeMemoryHooks(groupFolder: string): void {
  logger.debug({ groupFolder }, 'Memory hooks initialized');
  // Placeholder for any per-group initialization
}

/**
 * Check if memory hooks are enabled for a group
 */
export function areMemoryHooksEnabled(groupFolder?: string): boolean {
  const memorySystem = getMemorySystem();
  if (!memorySystem || !memorySystem.isEnabled()) {
    return false;
  }

  // Check if auto-recall or auto-capture is enabled
  return (
    memorySystem.isAutoRecallEnabled() || memorySystem.isAutoCaptureEnabled()
  );
}

/**
 * Get memory statistics for a group
 */
export async function getGroupMemoryStats(groupFolder: string): Promise<{
  totalMemories: number;
  byCategory: Record<string, number>;
} | null> {
  const memorySystem = getMemorySystem();
  if (!memorySystem || !memorySystem.isEnabled()) {
    return null;
  }

  try {
    const stats = await memorySystem.getStats({
      scope: 'group',
      scopeId: groupFolder,
    });
    return {
      totalMemories: stats.totalMemories,
      byCategory: stats.byCategory as Record<string, number>,
    };
  } catch (err) {
    logger.warn(
      { error: err, groupFolder },
      'Failed to get group memory stats',
    );
    return null;
  }
}
