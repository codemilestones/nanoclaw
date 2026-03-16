/**
 * Performance monitoring utilities for diagnosing bottlenecks
 */

/**
 * Simple hierarchical performance timer
 */
export class PerfTimer {
  private checkpoints: Map<string, { start: number; end?: number; parent?: string }> = new Map();
  private enabled: boolean;

  constructor(enabled: boolean = true) {
    this.enabled = enabled;
  }

  /**
   * Start timing a labeled operation
   */
  start(label: string, parent?: string): void {
    if (!this.enabled) return;
    this.checkpoints.set(label, { start: performance.now(), parent });
  }

  /**
   * End timing and return duration in milliseconds
   */
  end(label: string): number {
    if (!this.enabled) return 0;
    const checkpoint = this.checkpoints.get(label);
    if (!checkpoint) return 0;
    const duration = performance.now() - checkpoint.start;
    this.checkpoints.set(label, { ...checkpoint, end: performance.now() });
    return duration;
  }

  /**
   * Get duration for a label (if ended)
   */
  getDuration(label: string): number {
    if (!this.enabled) return 0;
    const checkpoint = this.checkpoints.get(label);
    if (!checkpoint || checkpoint.end === undefined) return 0;
    return checkpoint.end - checkpoint.start;
  }

  /**
   * Report all timings as a formatted string
   */
  report(): string {
    if (!this.enabled) return '';
    const lines: string[] = [];
    lines.push('[PERF] Performance Report:');

    // Build tree structure
    const rootItems = Array.from(this.checkpoints.entries())
      .filter(([_, data]) => !data.parent)
      .sort((a, b) => a[1].start - b[1].start);

    for (const [label, data] of rootItems) {
      this._reportNode(label, data, '', lines);
    }

    return lines.join('\n');
  }

  /**
   * Recursively report a node and its children
   */
  private _reportNode(
    label: string,
    data: { start: number; end?: number; parent?: string },
    indent: string,
    lines: string[],
  ): void {
    const duration = data.end !== undefined ? (data.end - data.start).toFixed(0) : '...';
    lines.push(`${indent}${label}: ${duration}ms`);

    // Find children
    const children = Array.from(this.checkpoints.entries())
      .filter(([_, childData]) => childData.parent === label)
      .sort((a, b) => a[1].start - b[1].start);

    for (const [childLabel, childData] of children) {
      this._reportNode(childLabel, childData, indent + '  ', lines);
    }
  }

  /**
   * Log a timing value immediately
   */
  log(label: string, durationMs: number, extra?: Record<string, unknown>): void {
    if (!this.enabled) return;
    const extraStr = extra ? ` ${JSON.stringify(extra)}` : '';
    console.log(`[PERF] ${label}: ${durationMs.toFixed(0)}ms${extraStr}`);
  }

  /**
   * Clear all checkpoints
   */
  clear(): void {
    this.checkpoints.clear();
  }
}

/**
 * Global performance timer instance
 */
let globalTimer: PerfTimer | null = null;

/**
 * Get or create global timer
 */
export function getPerfTimer(): PerfTimer {
  if (!globalTimer) {
    const enabled = process.env.PERF_DEBUG === 'true';
    globalTimer = new PerfTimer(enabled);
  }
  return globalTimer;
}

/**
 * Check if performance debugging is enabled
 */
export function isPerfDebugEnabled(): boolean {
  return process.env.PERF_DEBUG === 'true';
}

/**
 * Measure an async function and log its duration
 */
export async function measure<T>(
  label: string,
  fn: () => Promise<T>,
  timer?: PerfTimer,
): Promise<T> {
  const t = timer ?? getPerfTimer();
  t.start(label);
  try {
    const result = await fn();
    t.end(label);
    return result;
  } catch (err) {
    t.end(label);
    throw err;
  }
}

/**
 * Measure a sync function and log its duration
 */
export function measureSync<T>(
  label: string,
  fn: () => T,
  timer?: PerfTimer,
): T {
  const t = timer ?? getPerfTimer();
  t.start(label);
  try {
    const result = fn();
    t.end(label);
    return result;
  } catch (err) {
    t.end(label);
    throw err;
  }
}
