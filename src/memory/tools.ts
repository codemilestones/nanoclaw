/**
 * NanoClaw Advanced Memory System - IPC Tools Handler
 *
 * Defines IPC task handlers for memory operations.
 * These are called from the host's IPC watcher when containers request memory operations.
 */

import type { IpcDeps } from '../ipc.js';
import type {
  Memory,
  MemoryCategory,
  MemoryScope,
  MemorySearchResult,
} from './types.js';
import { getMemorySystem } from './index.js';
import { logger } from '../logger.js';

/**
 * IPC task data for memory_store
 */
export interface MemoryStoreIpcData {
  type: 'memory_store';
  content: string;
  category?: MemoryCategory;
  scope?: MemoryScope;
  scopeId?: string;
  importance?: number;
  metadata?: Record<string, unknown>;
}

/**
 * IPC task data for memory_recall
 */
export interface MemoryRecallIpcData {
  type: 'memory_recall';
  query: string;
  scope?: MemoryScope;
  scopeId?: string;
  categories?: MemoryCategory[];
  limit?: number;
  minScore?: number;
  includeGlobal?: boolean;
}

/**
 * IPC task data for memory_forget
 */
export interface MemoryForgetId {
  type: 'memory_forget';
  id?: string;
  query?: string;
  scope?: MemoryScope;
  scopeId?: string;
}

/**
 * IPC task data for memory_update
 */
export interface MemoryUpdateIpcData {
  type: 'memory_update';
  id: string;
  text?: string;
  category?: MemoryCategory;
  importance?: number;
}

/**
 * IPC task data for memory_list
 */
export interface MemoryListIpcData {
  type: 'memory_list';
  scope?: MemoryScope;
  scopeId?: string;
  category?: MemoryCategory;
  limit?: number;
}

/**
 * IPC task data for memory_stats
 */
export interface MemoryStatsIpcData {
  type: 'memory_stats';
  scope?: MemoryScope;
  scopeId?: string;
}

/**
 * Process memory-related IPC tasks
 */
export async function processMemoryIpc(
  data: {
    type: string;
    [key: string]: unknown;
  },
  sourceGroup: string,
  isMain: boolean,
  memorySystem: ReturnType<typeof getMemorySystem>,
): Promise<{ success: boolean; result?: unknown; error?: string }> {
  if (!memorySystem || !memorySystem.isEnabled()) {
    return {
      success: false,
      error: 'Memory system is not enabled',
    };
  }

  switch (data.type) {
    case 'memory_store': {
      const storeData = data as unknown as MemoryStoreIpcData;

      // Authorization: non-main can only store to their own scope
      if (!isMain && storeData.scopeId && storeData.scopeId !== sourceGroup) {
        return {
          success: false,
          error: 'Unauthorized: can only store to own scope',
        };
      }

      const sys = memorySystem as any;
      const storeResult = await sys.store({
        text: storeData.content,
        category: storeData.category,
        scope: storeData.scope || 'group',
        scopeId: storeData.scopeId || sourceGroup,
        importance: storeData.importance,
        metadata: storeData.metadata,
      });

      if (storeResult.success) {
        logger.info(
          { id: storeResult.id, category: storeData.category, sourceGroup },
          'Memory stored via IPC',
        );
      }

      return {
        success: storeResult.success,
        result: storeResult.id,
        error: storeResult.error,
      };
    }

    case 'memory_recall': {
      const recallData = data as unknown as MemoryRecallIpcData;

      // Authorization: non-main can only query their scope
      if (!isMain && recallData.scopeId && recallData.scopeId !== sourceGroup) {
        return {
          success: false,
          error: 'Unauthorized: can only query own scope',
        };
      }

      const results = await memorySystem.search({
        query: recallData.query,
        scope: recallData.scope || 'group',
        scopeId: recallData.scopeId || sourceGroup,
        categories: recallData.categories,
        limit: recallData.limit || 5,
        minScore: recallData.minScore,
        includeGlobal: recallData.includeGlobal !== false,
      });

      // Format results for IPC
      const formatted = results.map((r) => ({
        id: r.memory.id,
        text: r.memory.text,
        category: r.memory.category,
        score: r.score,
        importance: r.memory.importance,
      }));

      return {
        success: true,
        result: formatted,
      };
    }

    case 'memory_forget': {
      const forgetData = data as unknown as MemoryForgetId;

      if (forgetData.id) {
        // Delete by ID - only the owner scope can delete
        const memory = await memorySystem.get(forgetData.id);
        if (memory && !isMain && memory.scopeId !== sourceGroup) {
          return {
            success: false,
            error: 'Unauthorized: can only delete own memories',
          };
        }

        const deleted = await memorySystem.delete(forgetData.id);
        if (deleted) {
          logger.info(
            { id: forgetData.id, sourceGroup },
            'Memory deleted via IPC',
          );
        }

        return {
          success: deleted,
          result: deleted ? 1 : 0,
        };
      } else if (forgetData.query) {
        // Delete by query - only affects own scope
        const deleteResult = await memorySystem.deleteByQuery({
          scope: forgetData.scope || 'group',
          scopeId: forgetData.scopeId || sourceGroup,
        });

        logger.info(
          { query: forgetData.query, count: deleteResult.count, sourceGroup },
          'Memories deleted by query via IPC',
        );

        return {
          success: deleteResult.success,
          result: deleteResult.count,
          error: deleteResult.error,
        };
      }

      return {
        success: false,
        error: 'Either id or query must be provided',
      };
    }

    case 'memory_update': {
      const updateData = data as unknown as MemoryUpdateIpcData;

      // Check authorization
      const memory = await memorySystem.get(updateData.id);
      if (!memory) {
        return {
          success: false,
          error: 'Memory not found',
        };
      }

      if (!isMain && memory.scopeId !== sourceGroup) {
        return {
          success: false,
          error: 'Unauthorized: can only update own memories',
        };
      }

      const updated = await memorySystem.update(updateData.id, {
        text: updateData.text,
        category: updateData.category,
        importance: updateData.importance,
      });

      if (updated) {
        logger.info(
          { id: updateData.id, sourceGroup },
          'Memory updated via IPC',
        );
      }

      return {
        success: updated,
      };
    }

    case 'memory_list': {
      const listData = data as unknown as MemoryListIpcData;

      // Authorization: non-main can only list their scope
      if (!isMain && listData.scopeId && listData.scopeId !== sourceGroup) {
        return {
          success: false,
          error: 'Unauthorized: can only list own scope',
        };
      }

      const memories = await memorySystem.list({
        scope: listData.scope || 'group',
        scopeId: listData.scopeId || sourceGroup,
        category: listData.category,
        limit: listData.limit || 20,
      });

      const formatted = memories.map((m) => ({
        id: m.id,
        text: m.text.slice(0, 100) + (m.text.length > 100 ? '...' : ''),
        category: m.category,
        importance: m.importance,
        timestamp: m.timestamp,
      }));

      return {
        success: true,
        result: formatted,
      };
    }

    case 'memory_stats': {
      const statsData = data as unknown as MemoryStatsIpcData;

      // Authorization: non-main can only get their scope stats
      if (!isMain && statsData.scopeId && statsData.scopeId !== sourceGroup) {
        return {
          success: false,
          error: 'Unauthorized: can only get own scope stats',
        };
      }

      const stats = await memorySystem.getStats({
        scope: statsData.scope,
        scopeId: statsData.scopeId || sourceGroup,
      });

      return {
        success: true,
        result: stats,
      };
    }

    default:
      return {
        success: false,
        error: `Unknown memory task type: ${data.type}`,
      };
  }
}

/**
 * Write memory IPC response to a file
 */
export async function writeMemoryIpcResponse(
  groupFolder: string,
  response: { success: boolean; result?: unknown; error?: string },
): Promise<void> {
  const fs = await import('fs');
  const path = await import('path');

  const ipcDir = path.join(process.cwd(), 'data', 'ipc', groupFolder, 'input');
  fs.mkdirSync(ipcDir, { recursive: true });

  const filename = `memory-response-${Date.now()}.json`;
  const filepath = path.join(ipcDir, filename);

  fs.writeFileSync(filepath, JSON.stringify(response) + '\n');
}
