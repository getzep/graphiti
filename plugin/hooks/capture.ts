import type { GraphitiClient, GraphitiMessage } from '../client.ts';
import { normalizeConfig } from '../config.ts';
import type { PluginConfig } from '../config.ts';
import type { PackInjectorContext } from './pack-injector.ts';

export interface AgentEndEvent {
  success?: boolean;
  messages?: Array<{ role?: string; content: string; name?: string }>;
}

export type CaptureHook = (event: AgentEndEvent, ctx: PackInjectorContext) => Promise<void>;

export interface CaptureHookDeps {
  client: GraphitiClient;
  config?: Partial<PluginConfig>;
}

const GRAPHITI_CONTEXT_RE = /<graphiti-context>[\s\S]*?<\/graphiti-context>/gi;
const PACK_CONTEXT_RE = /<pack-context[\s\S]*?<\/pack-context>/gi;
const FALLBACK_CONTEXT_RE = /<graphiti-fallback>[\s\S]*?<\/graphiti-fallback>/gi;

export const stripInjectedContext = (content: string): string => {
  return content
    .replace(GRAPHITI_CONTEXT_RE, '')
    .replace(PACK_CONTEXT_RE, '')
    .replace(FALLBACK_CONTEXT_RE, '')
    .trim();
};

const resolveGroupId = (ctx: PackInjectorContext): string | null => {
  return ctx.messageProvider?.groupId ?? ctx.sessionKey ?? null;
};

const extractTurn = (messages: Array<{ role?: string; content: string }>): GraphitiMessage[] => {
  const reversed = [...messages].reverse();
  const assistant = reversed.find((message) => message.role === 'assistant');
  const user = reversed.find((message) => message.role === 'user');

  const cleaned: GraphitiMessage[] = [];
  if (user) {
    cleaned.push({
      content: stripInjectedContext(user.content),
      role_type: 'user',
    });
  }
  if (assistant) {
    cleaned.push({
      content: stripInjectedContext(assistant.content),
      role_type: 'assistant',
    });
  }
  return cleaned;
};

export const createCaptureHook = (deps: CaptureHookDeps): CaptureHook => {
  const config = normalizeConfig(deps.config);
  const logger = config.debug ? (message: string) => console.log(message) : () => undefined;

  return async (event, ctx) => {
    if (!event.success) {
      return;
    }

    const groupId = resolveGroupId(ctx);
    if (!groupId) {
      logger('Capture skipped: missing group ID.');
      return;
    }

    const messages = event.messages ?? [];
    if (messages.length === 0) {
      return;
    }

    const turnMessages = extractTurn(messages);
    if (turnMessages.length === 0) {
      return;
    }

    void deps.client
      .ingestMessages(groupId, turnMessages)
      .catch((error) => logger(`Graphiti capture failed: ${(error as Error).message}`));
  };
};
