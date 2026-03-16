import { Channel, NewMessage } from './types.js';
import { formatLocalTime } from './timezone.js';
import { getPerfTimer, isPerfDebugEnabled } from './utils/performance.js';

export function escapeXml(s: string): string {
  if (!s) return '';
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function formatMessages(
  messages: NewMessage[],
  timezone: string,
): string {
  const lines = messages.map((m) => {
    const displayTime = formatLocalTime(m.timestamp, timezone);
    return `<message sender="${escapeXml(m.sender_name)}" time="${escapeXml(displayTime)}">${escapeXml(m.content)}</message>`;
  });

  const header = `<context timezone="${escapeXml(timezone)}" />\n`;

  return `${header}<messages>\n${lines.join('\n')}\n</messages>`;
}

export function stripInternalTags(text: string): string {
  return text.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
}

export function formatOutbound(rawText: string): string {
  const text = stripInternalTags(rawText);
  if (!text) return '';
  return text;
}

export function routeOutbound(
  channels: Channel[],
  jid: string,
  text: string,
): Promise<void> {
  const channel = channels.find((c) => c.ownsJid(jid) && c.isConnected());
  if (!channel) throw new Error(`No channel for JID: ${jid}`);
  return channel.sendMessage(jid, text);
}

export function findChannel(
  channels: Channel[],
  jid: string,
): Channel | undefined {
  return channels.find((c) => c.ownsJid(jid));
}

/**
 * Format messages with memory recall injected.
 * This is an async version that includes relevant memories from the memory system.
 */
export async function formatMessagesWithMemory(
  messages: NewMessage[],
  timezone: string,
  groupFolder: string,
): Promise<string> {
  // Import memory hooks dynamically to avoid circular dependencies
  const { autoRecall } = await import('./memory/hooks.js');

  // Performance tracking
  const timer = getPerfTimer();
  const outerLabel = `formatMessagesWithMemory[${groupFolder.split('/').pop()}]`;
  timer.start(outerLabel);

  // Get auto-recall results
  timer.start(`${outerLabel}/autoRecall`);
  const memoryBlock = await autoRecall(messages, groupFolder);
  timer.end(`${outerLabel}/autoRecall`);

  // Format messages
  timer.start(`${outerLabel}/formatMessages`);
  const formattedMessages = formatMessages(messages, timezone);
  timer.end(`${outerLabel}/formatMessages`);

  // Inject memory block if we have results
  if (memoryBlock) {
    timer.end(outerLabel);
    return `${formattedMessages}\n\n${memoryBlock}`;
  }

  timer.end(outerLabel);
  return formattedMessages;
}
