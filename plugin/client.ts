export interface GraphitiFact {
  uuid?: string;
  name?: string;
  fact: string;
}

export interface GraphitiEntity {
  uuid?: string;
  name?: string;
  summary: string;
}

export interface GraphitiSearchResults {
  facts: GraphitiFact[];
  entities?: GraphitiEntity[];
}

export interface GraphitiMessage {
  content: string;
  role_type: 'user' | 'assistant' | 'system';
  role?: string | null;
  name?: string;
  uuid?: string | null;
  timestamp?: string;
  source_description?: string;
}

export interface GraphitiClientConfig {
  baseUrl: string;
  apiKey?: string;
  recallTimeoutMs: number;
  captureTimeoutMs: number;
  maxFacts: number;
}

export class GraphitiClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly recallTimeoutMs: number;
  private readonly captureTimeoutMs: number;
  private readonly maxFacts: number;

  constructor(config: GraphitiClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.recallTimeoutMs = config.recallTimeoutMs;
    this.captureTimeoutMs = config.captureTimeoutMs;
    this.maxFacts = config.maxFacts;
  }

  async search(query: string, groupIds?: string[]): Promise<GraphitiSearchResults> {
    const body = {
      query,
      group_ids: groupIds && groupIds.length > 0 ? groupIds : undefined,
      max_facts: this.maxFacts,
    };

    const response = await this.request('/search', body, this.recallTimeoutMs);
    const payload = (await response.json()) as { facts?: GraphitiFact[] };
    return {
      facts: payload.facts ?? [],
      entities: [],
    };
  }

  async ingestMessages(groupId: string, messages: GraphitiMessage[]): Promise<void> {
    const body = {
      group_id: groupId,
      messages,
    };
    await this.request('/messages', body, this.captureTimeoutMs);
  }

  private async request(path: string, body: object, timeoutMs: number): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Graphiti API error ${response.status}`);
      }

      return response;
    } finally {
      clearTimeout(timeout);
    }
  }
}
