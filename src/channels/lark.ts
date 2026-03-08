import * as lark from '@larksuiteoapi/node-sdk';

import { ASSISTANT_NAME, TRIGGER_PATTERN } from '../config.js';
import { readEnvFile } from '../env.js';
import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import {
  Channel,
  OnChatMetadata,
  OnInboundMessage,
  RegisteredGroup,
} from '../types.js';

export class LarkChannel implements Channel {
  name = 'lark';

  private client: lark.Client;
  private wsClient: lark.WSClient | null = null;
  private eventDispatcher: lark.EventDispatcher;
  private opts: ChannelOpts;
  private botOpenId = '';
  private connected = false;
  private userNameCache = new Map<string, string>();

  constructor(appId: string, appSecret: string, opts: ChannelOpts) {
    this.opts = opts;
    this.client = new lark.Client({
      appId,
      appSecret,
      appType: lark.AppType.SelfBuild,
    });
    this.eventDispatcher = new lark.EventDispatcher({}).register({
      'im.message.receive_v1': (data: any) => this.handleMessageEvent(data),
    });
    this.wsClient = new lark.WSClient({
      appId,
      appSecret,
      loggerLevel: lark.LoggerLevel.warn,
    });
  }

  async connect(): Promise<void> {
    // Get bot open_id by sending a request to the bot info API
    try {
      const resp = await (this.client as any).request({
        method: 'GET',
        url: 'https://open.feishu.cn/open-apis/bot/v3/info',
      });
      this.botOpenId = resp?.bot?.open_id || '';
      logger.info(
        { botOpenId: this.botOpenId },
        'Lark bot info retrieved',
      );
    } catch (err) {
      logger.warn({ err }, 'Failed to get Lark bot info, continuing anyway');
    }

    // Start WebSocket long connection
    await this.wsClient!.start({ eventDispatcher: this.eventDispatcher });
    this.connected = true;
    logger.info('Lark channel connected via WebSocket');
    console.log('\n  Lark bot connected via WebSocket');
    console.log(
      '  Send any message to the bot to get a chat\'s registration ID\n',
    );
  }

  private async handleMessageEvent(data: any): Promise<void> {
    logger.info({ keys: data ? Object.keys(data) : null }, 'Lark event received');
    try {
      const event = data;
      const message = event?.message;
      const sender = event?.sender;

      if (!message || !sender) {
        logger.info({ data: JSON.stringify(data).slice(0, 500) }, 'Lark: missing message or sender');
        return;
      }

      // Skip messages from the bot itself
      const senderOpenId = sender?.sender_id?.open_id || '';
      if (senderOpenId === this.botOpenId) return;

      const chatId = message.chat_id;
      const chatJid = `lark:${chatId}`;
      const msgId = message.message_id;
      const messageType: string = message.message_type;
      const timestamp = new Date(
        parseInt(message.create_time, 10) * 1000,
      ).toISOString();

      // Parse message content
      const content = this.parseMessageContent(
        messageType,
        message.content,
        message.mentions,
      );

      // Resolve sender name
      const senderName = await this.resolveSenderName(senderOpenId, sender);

      // Determine if group chat
      const isGroup = message.chat_type === 'group';

      // Report chat metadata
      this.opts.onChatMetadata(chatJid, timestamp, undefined, 'lark', isGroup);

      // Only deliver messages for registered groups
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) {
        logger.debug(
          { chatJid, chatId },
          'Message from unregistered Lark chat',
        );
        return;
      }

      this.opts.onMessage(chatJid, {
        id: msgId,
        chat_jid: chatJid,
        sender: senderOpenId,
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
      });

      logger.info(
        { chatJid, sender: senderName, messageType },
        'Lark message stored',
      );
    } catch (err) {
      logger.error({ err }, 'Error handling Lark message event');
    }
  }

  /**
   * Parse message content based on message_type.
   * For text messages, also handle @bot mentions → trigger pattern.
   */
  private parseMessageContent(
    messageType: string,
    rawContent: string,
    mentions?: any[],
  ): string {
    let parsed: any;
    try {
      parsed = JSON.parse(rawContent);
    } catch {
      return rawContent || '';
    }

    switch (messageType) {
      case 'text': {
        let text: string = parsed.text || '';

        // Replace @_user_N placeholders from mentions
        if (mentions && mentions.length > 0) {
          for (const mention of mentions) {
            const key = mention.key; // e.g., "@_user_1"
            if (mention.id?.open_id === this.botOpenId) {
              // Bot mention → replace with trigger pattern
              text = text.replace(key, `@${ASSISTANT_NAME}`);
            } else {
              // Other user mention → use their name
              const name = mention.name || 'someone';
              text = text.replace(key, `@${name}`);
            }
          }
        }

        return text.trim();
      }
      case 'post': {
        // Rich text — extract plain text from all paragraphs
        const title = parsed.title || '';
        const paragraphs: string[] = [];
        // post content can be in zh_cn, en_us, or other locales
        const localeContent =
          parsed.content || (Object.values(parsed)[0] as any);
        if (Array.isArray(localeContent)) {
          for (const paragraph of localeContent) {
            if (!Array.isArray(paragraph)) continue;
            const parts = paragraph
              .map((el: any) => {
                if (el.tag === 'text') return el.text || '';
                if (el.tag === 'a') return el.text || el.href || '';
                if (el.tag === 'at') {
                  if (el.user_id === this.botOpenId) return `@${ASSISTANT_NAME}`;
                  return `@${el.user_name || 'someone'}`;
                }
                if (el.tag === 'img') return '[Image]';
                if (el.tag === 'media') return '[Media]';
                return '';
              })
              .filter(Boolean);
            paragraphs.push(parts.join(''));
          }
        }
        const body = paragraphs.join('\n');
        return title ? `${title}\n${body}` : body;
      }
      case 'image':
        return '[Image]';
      case 'file':
        return `[File: ${parsed.file_name || 'file'}]`;
      case 'audio':
        return '[Audio]';
      case 'media':
        return '[Video]';
      case 'sticker':
        return '[Sticker]';
      case 'share_chat':
        return '[Shared Chat]';
      case 'share_user':
        return '[Shared Contact]';
      case 'location':
        return `[Location: ${parsed.name || ''}]`;
      case 'merge_forward':
        return '[Forwarded Messages]';
      default:
        return `[${messageType}]`;
    }
  }

  /**
   * Resolve sender's display name via contact API, with cache.
   */
  private async resolveSenderName(
    openId: string,
    sender: any,
  ): Promise<string> {
    // Check cache first
    const cached = this.userNameCache.get(openId);
    if (cached) return cached;

    // Try sender_id.name from event data (not always present)
    // Fall back to contact API
    try {
      const resp = await this.client.contact.v3.user.get({
        path: { user_id: openId },
        params: { user_id_type: 'open_id' },
      });
      const name = (resp as any)?.data?.user?.name || openId;
      this.userNameCache.set(openId, name);
      return name;
    } catch {
      // If contact API fails, use openId as fallback
      const fallback = openId.slice(0, 8);
      this.userNameCache.set(openId, fallback);
      return fallback;
    }
  }

  async sendMessage(jid: string, text: string): Promise<void> {
    try {
      const chatId = jid.replace(/^lark:/, '');

      // Lark has a ~30KB limit per message; split on 4000 chars to be safe
      const MAX_LENGTH = 4000;
      const chunks =
        text.length <= MAX_LENGTH
          ? [text]
          : text.match(new RegExp(`[\\s\\S]{1,${MAX_LENGTH}}`, 'g')) || [text];

      for (const chunk of chunks) {
        await this.client.im.v1.message.create({
          params: { receive_id_type: 'chat_id' },
          data: {
            receive_id: chatId,
            msg_type: 'text',
            content: JSON.stringify({ text: chunk }),
          },
        });
      }

      logger.info({ jid, length: text.length }, 'Lark message sent');
    } catch (err) {
      logger.error({ jid, err }, 'Failed to send Lark message');
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('lark:');
  }

  async disconnect(): Promise<void> {
    this.connected = false;
    // WSClient doesn't expose a stop method — setting to null lets GC clean up
    this.wsClient = null;
    logger.info('Lark channel disconnected');
  }
}

registerChannel('lark', (opts: ChannelOpts) => {
  const envVars = readEnvFile(['LARK_APP_ID', 'LARK_APP_SECRET']);
  const appId = process.env.LARK_APP_ID || envVars.LARK_APP_ID || '';
  const appSecret =
    process.env.LARK_APP_SECRET || envVars.LARK_APP_SECRET || '';
  if (!appId || !appSecret) {
    logger.warn('Lark: LARK_APP_ID or LARK_APP_SECRET not set');
    return null;
  }
  return new LarkChannel(appId, appSecret, opts);
});
